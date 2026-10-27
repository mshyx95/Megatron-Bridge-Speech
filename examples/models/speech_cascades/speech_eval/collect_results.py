# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate retained MPI8 evidence and aggregate existing metrics, without judging."""

import argparse
import hashlib
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

from .mpi import SUITE_OUTPUT_COUNTS, digest, file_digest, merge_records, read_shard


logger = logging.getLogger(__name__)
LONG_STANDARD = {"generate_earnings21.txt", "generate_earnings22.txt", "generate_coraal.txt"}


class StoredMetrics(TypedDict, total=False):
    """Optional metrics emitted by the existing WER and BLEU scorers."""

    wer_percent: float
    reference_words: int
    bleu: float
    final_score: str


class DatasetResult(StoredMetrics):
    """Validated dataset evidence and denominators, without private runtime paths."""

    output: str
    samples: int
    requests: int
    output_cap_hits: int
    empty_predictions: int
    sha256: str
    audit_sha256: str
    contract_sha256: str


def _metric(score: str) -> StoredMetrics:
    match = re.fullmatch(r"WER:\s*([\d.]+),\s*#REF:\s*(\d+)", score)
    if match:
        wer, words = float(match[1]), int(match[2])
        if not math.isfinite(wer) or words <= 0:
            raise ValueError("Invalid WER or reference-word denominator")
        return {"wer_percent": wer, "reference_words": words}
    match = re.match(r"BLEU:\s*(?:BLEU\s*=\s*)?([\d.]+)(?:,|\s)", score)
    if match and math.isfinite(float(match[1])):
        return {"bleu": float(match[1])}
    raise ValueError(f"Unrecognized stored metric: {score}")


def aggregate_metrics(datasets: list[DatasetResult]) -> dict[str, int | float | str]:
    """Aggregate stored per-dataset metrics, preserving their reported precision."""
    if not datasets:
        raise ValueError("Cannot aggregate empty datasets")
    if all("wer_percent" in row for row in datasets):
        words = sum(row["reference_words"] for row in datasets)
        return {
            "datasets": len(datasets),
            "samples": sum(row["samples"] for row in datasets),
            "reference_words": words,
            "wer_macro_percent": sum(row["wer_percent"] for row in datasets) / len(datasets),
            "wer_reference_weighted_percent": sum(row["wer_percent"] * row["reference_words"] for row in datasets)
            / words,
            "precision": "aggregated from stored per-dataset WER; not a fresh scorer run",
        }
    if all("bleu" in row for row in datasets):
        return {"directions": len(datasets), "bleu_macro": sum(row["bleu"] for row in datasets) / len(datasets)}
    raise ValueError("Cannot aggregate missing or mixed metrics")


def collect_suite(root: Path, *, model: str, stage: str, step: int, protocol: str) -> dict[str, object]:
    """Validate hashes, eight-rank coverage and request counts before collecting.

    Args:
        root: Downloaded suite directory, including all rank receipts and outputs.
        model: Public model identity supplied by the operator.
        stage: Training-stage label; never inferred from a directory name.
        step: Exact checkpoint iteration supplied by the operator.
        protocol: Public protocol identifier resolving to the retained run manifest.

    Returns:
        A report with source hashes, denominators and stored-metric aggregation.
        Model/stage/step labels must still be checked against the run manifest.
    """
    if step < 1 or not all(value.strip() for value in (model, stage, protocol)):
        raise ValueError("Explicit model, stage, positive step and protocol are required")
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    suite = summary["suite"]
    smoke = summary.get("smoke_samples", 0)
    if suite not in SUITE_OUTPUT_COUNTS or summary.get("merged") is not True or summary.get("world_size") != 8:
        raise ValueError("Expected a complete MPI8 suite summary")
    if not isinstance(smoke, int) or smoke < 0 or bool(smoke) != summary.get("partial_dataset", False):
        raise ValueError("Inconsistent partial-dataset marker")
    entries = summary["datasets"]
    names = [entry["output"] for entry in entries]
    expected = 1 if smoke else SUITE_OUTPUT_COUNTS[suite]
    if len(names) != expected or len(set(names)) != len(names):
        raise ValueError("Missing or duplicate suite outputs")
    if smoke and (suite != "openasr" or names != ["generate_librispeech-clean.txt"]):
        raise ValueError("Unexpected smoke dataset")
    datasets: list[DatasetResult] = []
    for entry in entries:
        name = entry["output"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Output must be a local filename")
        output = root / "results" / name
        receipt = json.loads(output.with_suffix(output.suffix + ".complete.json").read_text())
        if any(entry.get(key) != value for key, value in receipt.items()):
            raise ValueError("Summary and completion receipt disagree")
        if receipt["run_key"] != summary["run_key"] or receipt["world_size"] != 8:
            raise ValueError("Stale or mismatched suite run identity")
        audit_path = output.with_suffix(".requests.jsonl")
        if file_digest(output) != receipt["sha256"] or file_digest(audit_path) != receipt["audit_sha256"]:
            raise ValueError("Modified merged predictions or audit")
        rank_name = "responses.worker0.json" if suite == "bfcl" else name
        rows = json.loads(output.read_text())
        predictions = [row for row in rows if "final_score" not in row]
        audit_lines = audit_path.read_bytes().splitlines(keepends=True)
        shards = []
        audit_offset = 0
        for rank in range(8):
            rank_output = root / "ranks" / str(rank) / "results" / rank_name
            if rank_output.exists():
                shard = read_shard(rank_output)
            else:
                shard = json.loads(rank_output.with_suffix(rank_output.suffix + ".complete.json").read_text())
                # Historical downloads retained rank receipts, not duplicate prediction files.
                records = predictions[rank::8]
                encoded = (json.dumps(records, ensure_ascii=False, indent=2) + "\n").encode()
                if hashlib.sha256(encoded).hexdigest() != shard["sha256"]:
                    raise ValueError("Reconstructed rank predictions disagree with receipt")
                shard["records"] = records
            count = sum(shard["request_counts"].values())
            rank_audit = b"".join(audit_lines[audit_offset : audit_offset + count])
            if hashlib.sha256(rank_audit).hexdigest() != shard["audit_sha256"]:
                raise ValueError("Merged audit does not preserve complete rank request evidence")
            audit_offset += count
            shards.append(shard)
        if audit_offset != len(audit_lines):
            raise ValueError("Unexpected extra merged audit records")
        if [shard["sha256"] for shard in shards] != receipt["rank_sha256"]:
            raise ValueError("Rank hashes disagree with merged receipt")
        if any(
            shard["rank"] != rank or shard["contract"]["run_key"] != summary["run_key"]
            for rank, shard in enumerate(shards)
        ):
            raise ValueError("Stale or misplaced rank receipt")
        ordered = merge_records(shards, 8)
        if predictions != ordered or len(predictions) != receipt["source_count"]:
            raise ValueError("Merged predictions do not match complete ordered rank coverage")
        if digest([row["audio_ids"][0] for row in predictions]) != receipt["source_sha256"]:
            raise ValueError("Merged source IDs disagree")
        if smoke and len(predictions) != smoke:
            raise ValueError("Smoke sample denominator disagrees")
        audit = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        counts = Counter(row["audio_id"] for row in audit)
        expected_counts: Counter[str] = Counter()
        for shard in shards:
            expected_counts.update(shard["request_counts"])
        if counts != expected_counts or set(counts) != {row["audio_ids"][0] for row in predictions}:
            raise ValueError("Missing or extra recording/segment requests")
        if any(row["status"] != "ok" or row.get("finish_reason") not in {"stop", "length"} for row in audit):
            raise ValueError("Failed or incomplete generation request")
        dataset: DatasetResult = {
            "output": name,
            "samples": len(predictions),
            "requests": len(audit),
            "output_cap_hits": sum(row["finish_reason"] == "length" for row in audit),
            "empty_predictions": sum(not row["generated_texts"][0].strip() for row in predictions),
            "sha256": receipt["sha256"],
            "audit_sha256": receipt["audit_sha256"],
            "contract_sha256": digest(shards[0]["contract"]),
        }
        scores = [row["final_score"] for row in rows if "final_score" in row]
        if suite in {"openasr", "openasr_ml", "openasr_long", "fleurs"}:
            if len(scores) != 1:
                raise ValueError("Expected exactly one stored dataset score")
            dataset["final_score"] = scores[0]
            metric = _metric(scores[0])
            if suite == "fleurs":
                if "bleu" not in metric:
                    raise ValueError("Fleurs requires a stored BLEU score")
                dataset["bleu"] = metric["bleu"]
            else:
                if "wer_percent" not in metric or "reference_words" not in metric:
                    raise ValueError("ASR requires stored WER and reference-word counts")
                dataset["wer_percent"] = metric["wer_percent"]
                dataset["reference_words"] = metric["reference_words"]
        datasets.append(dataset)
    report: dict[str, object] = {
        "schema_version": 1,
        "model": model,
        "checkpoint_stage": stage,
        "checkpoint_step": step,
        "protocol": protocol,
        "identity_source": "operator labels; verify against retained run manifest",
        "suite": suite,
        "run_key": summary["run_key"],
        "summary_sha256": file_digest(summary_path),
        "partial_dataset": bool(smoke),
        "samples": sum(row["samples"] for row in datasets),
        "requests": sum(row["requests"] for row in datasets),
        "datasets": datasets,
    }
    if suite in {"openasr", "openasr_ml", "openasr_long", "fleurs"}:
        report["aggregate"] = aggregate_metrics(datasets)
    else:
        report["scoring_status"] = "generation validated only; use the existing suite scorer/judge"
    if suite == "openasr_long":
        standard = [row for row in datasets if row["output"] in LONG_STANDARD]
        if len(standard) != 3:
            raise ValueError("Missing standard three-set long-form evidence")
        report["standard_three_set"] = aggregate_metrics(standard)
        report["aggregate_scope"] = "four-set diagnostic; compare standard_three_set separately"
    return report


def main() -> None:
    """Write a fresh local collection report; never download or invoke a judge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    report = collect_suite(args.root, model=args.model, stage=args.stage, step=args.step, protocol=args.protocol)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Collected %s samples=%d partial=%s -> %s",
        report["suite"],
        report["samples"],
        report["partial_dataset"],
        output,
    )


if __name__ == "__main__":
    main()
