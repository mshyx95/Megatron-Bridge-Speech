# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler MPI ranks, whole-recording shards and validated evaluation merge."""

import argparse
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path


logger = logging.getLogger(__name__)
SUITE_OUTPUT_COUNTS = {
    "openasr": 7,
    "openasr_ml": 14,
    "openasr_long": 4,
    "bbh": 1,
    "voicebench": 1,
    "bfcl": 1,
    "fleurs": 56,
}


def rank_context(environment: Mapping[str, str]) -> tuple[int, int, str]:
    """Require eight scheduler ranks on one node and select the assigned GPU."""
    required = (
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
    )
    if any(key not in environment for key in required):
        raise ValueError("Launch with mpi: true and process_count_per_node: 8")
    rank, local_rank, world_size, local_size = (int(environment[key]) for key in required)
    if world_size != 8 or local_size != 8 or rank != local_rank or not 0 <= rank < 8:
        raise ValueError("Expected one node with exactly eight MPI ranks")
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    devices = [item.strip() for item in visible.split(",")] if visible is not None else list(map(str, range(8)))
    if not all(devices) or "-1" in devices or len(set(devices)) != len(devices) or len(devices) not in (1, 8):
        raise ValueError("MPI GPU visibility must contain one assigned GPU or eight distinct GPUs")
    return rank, world_size, devices[0] if len(devices) == 1 else devices[local_rank]


def digest(value: object) -> str:
    """Hash a structured contract or ordered source inventory."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class StridedDataset:
    """Select complete recordings without padding, including legitimate empty shards."""

    def __init__(self, dataset, source_ids: list[str], *, rank: int, world_size: int):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid shard rank or world size")
        if len(source_ids) != len(dataset) or not source_ids or len(set(source_ids)) != len(source_ids):
            raise ValueError("Empty source or missing/duplicate source IDs")
        self.dataset = dataset
        self.source_ids = source_ids
        self.rank = rank
        self.world_size = world_size
        self.indices = list(range(rank, len(source_ids), world_size))
        self.visited = []
        self.request_counts = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = self.indices[index]
        example = self.dataset[source_index]
        chunks = example if isinstance(example, list) else [example]
        if not chunks or any(chunk["audio_id"] != self.source_ids[source_index] for chunk in chunks):
            raise ValueError("Empty or misaligned whole-recording chunks")
        self.visited.append(source_index)
        self.request_counts[self.source_ids[source_index]] = len(chunks)
        return example

    def receipt(self, records: list[dict], contract: dict) -> dict:
        """Validate traversal and output IDs before declaring the shard complete."""
        expected_ids = [self.source_ids[index] for index in self.indices]
        if self.visited != self.indices or [row["audio_ids"][0] for row in records] != expected_ids:
            raise ValueError("Incomplete or reordered shard traversal/predictions")
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "source_count": len(self.source_ids),
            "source_sha256": digest(self.source_ids),
            "indices": self.indices,
            "contract": contract,
        }


def merge_records(shards: list[dict], world_size: int) -> list[dict]:
    """Validate all ranks and recover the original order, never averaging shard metrics."""
    if len(shards) != world_size or {shard["rank"] for shard in shards} != set(range(world_size)):
        raise ValueError("Missing or duplicate shard ranks")
    totals = {shard["source_count"] for shard in shards}
    if len(totals) != 1 or next(iter(totals)) < 1:
        raise ValueError("Inconsistent or empty source dataset")
    if (
        len({shard["source_sha256"] for shard in shards}) != 1
        or len({digest(shard["contract"]) for shard in shards}) != 1
    ):
        raise ValueError("Shard source ordering or evaluation contract differs")
    total = totals.pop()
    ordered = {}
    ids = set()
    for shard in shards:
        expected = list(range(shard["rank"], total, world_size))
        if shard["world_size"] != world_size or shard["indices"] != expected:
            raise ValueError("Incorrect shard partition")
        if len(shard["records"]) != len(expected):
            raise ValueError("Incomplete shard records")
        for index, row in zip(expected, shard["records"]):
            if len(row.get("audio_ids", [])) != 1 or len(row.get("generated_texts", [])) != 1:
                raise ValueError("Malformed shard prediction")
            audio_id = row["audio_ids"][0]
            if audio_id in ids or index in ordered or not isinstance(row["generated_texts"][0], str):
                raise ValueError("Duplicate IDs or malformed prediction")
            ids.add(audio_id)
            ordered[index] = row
    if set(ordered) != set(range(total)):
        raise ValueError("Missing source indices")
    result = [ordered[index] for index in range(total)]
    if digest([row["audio_ids"][0] for row in result]) != shards[0]["source_sha256"]:
        raise ValueError("Merged IDs differ from source order")
    return result


def atomic_json(path: Path, value: object) -> None:
    """Publish a JSON artifact only after it has been fully written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def file_digest(path: Path) -> str:
    """Hash an output, audit, or small identity file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_shard(output: Path, records: list[dict], shard: StridedDataset, contract: dict) -> None:
    """Write a completion receipt only for fully audited predictions."""
    metadata = shard.receipt(records, contract)
    audit_path = output.with_suffix(".requests.jsonl")
    if not records and not audit_path.exists():
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("")
    with audit_path.open() as stream:
        audit = [json.loads(line) for line in stream if line.strip()]
    counts = Counter(row["audio_id"] for row in audit)
    if counts != shard.request_counts or any(row["status"] != "ok" for row in audit):
        raise ValueError("Incomplete or failed shard request audit")
    atomic_json(output, records)
    metadata.update(sha256=file_digest(output), audit_sha256=file_digest(audit_path), request_counts=dict(counts))
    atomic_json(output.with_suffix(output.suffix + ".complete.json"), metadata)


def read_shard(output: Path) -> dict:
    """Reject modified outputs or request audits before any global merge."""
    metadata = json.loads(output.with_suffix(output.suffix + ".complete.json").read_text())
    if (
        file_digest(output) != metadata["sha256"]
        or file_digest(output.with_suffix(".requests.jsonl")) != metadata["audit_sha256"]
    ):
        raise ValueError("Modified shard output or audit")
    return {**metadata, "records": json.loads(output.read_text())}


def child_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Keep scheduler GPU selection but isolate each single-GPU server from MPI."""
    distributed = {
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    }
    return {
        key: value
        for key, value in environment.items()
        if key not in distributed and not key.startswith(("OMPI_", "PMI_", "PMIX_", "TORCHELASTIC_"))
    }


def validate_ranks(records: list[dict]) -> None:
    """Reject duplicated GPUs, suites, model identities or output locations."""
    if len(records) != 8 or {record["rank"] for record in records} != set(range(8)):
        raise ValueError("Missing or duplicate MPI ranks")
    if len({record["device"] for record in records}) != 8:
        raise ValueError("Multiple MPI ranks selected the same GPU")
    for field in ("output", "suite", "run_key"):
        if len({record[field] for record in records}) != 1:
            raise ValueError(f"MPI ranks disagree on {field}")


def merge_suite(root: Path, *, suite: str, run_key: str, environment: dict[str, str]) -> None:
    """Validate complete shards and rescore each dataset once, then run postprocessing."""
    from .speech_task import score_records

    inventories = [
        set(path.name for path in (root / "ranks" / str(rank) / "results").glob("*.complete.json"))
        for rank in range(8)
    ]
    smoke_samples = int(environment.get("SPEECH_SMOKE_SAMPLES", "0"))
    if smoke_samples and (suite != "openasr" or not 1 <= smoke_samples <= 128):
        raise ValueError("Smoke requires OpenASR with 1..128 samples")
    expected_count = 1 if smoke_samples else SUITE_OUTPUT_COUNTS[suite]
    if len(inventories[0]) != expected_count or any(names != inventories[0] for names in inventories):
        raise ValueError("Missing or inconsistent dataset shard receipts")
    if smoke_samples and inventories[0] != {"generate_librispeech-clean.txt.complete.json"}:
        raise ValueError("Unexpected smoke dataset")
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=False)
    summary = []
    for receipt_name in sorted(inventories[0]):
        name = receipt_name.removesuffix(".complete.json")
        paths = [root / "ranks" / str(rank) / "results" / name for rank in range(8)]
        shards = [read_shard(path) for path in paths]
        if any(shard["rank"] != rank or shard["contract"]["run_key"] != run_key for rank, shard in enumerate(shards)):
            raise ValueError("Stale or misplaced rank receipt")
        rows = merge_records(shards, 8)
        contract = shards[0]["contract"]
        if smoke_samples and (
            shards[0]["source_count"] != smoke_samples or contract.get("smoke_samples") != smoke_samples
        ):
            raise ValueError("Smoke receipt sample count mismatch")
        destination = result_dir / ("responses.json" if suite == "bfcl" else name)
        if suite != "bfcl":
            score = score_records(rows, task=contract["task"], sub_task=contract["sub_task"])
            if score is not None:
                rows.append({"final_score": score})
        atomic_json(destination, rows)
        audit_path = destination.with_suffix(".requests.jsonl")
        with audit_path.open("w") as output:
            for path in paths:
                with path.with_suffix(".requests.jsonl").open() as audit:
                    for line in audit:
                        output.write(line)
        metadata = {
            "run_key": run_key,
            "world_size": 8,
            "source_count": shards[0]["source_count"],
            "source_sha256": shards[0]["source_sha256"],
            "sha256": file_digest(destination),
            "audit_sha256": file_digest(audit_path),
            "rank_sha256": [shard["sha256"] for shard in shards],
        }
        atomic_json(destination.with_suffix(destination.suffix + ".complete.json"), metadata)
        logger.info(
            "MERGE_DATASET suite=%s samples=%d score=%s output=%s audit=%s receipt=%s",
            suite,
            metadata["source_count"],
            rows[-1].get("final_score", "generation-only"),
            destination,
            audit_path,
            destination.with_suffix(destination.suffix + ".complete.json"),
        )
        summary.append({"output": destination.name, **metadata})
    scripts = Path(__file__).parent / "scripts/audio_eval"
    if suite == "bbh":
        subprocess.run(
            [
                sys.executable,
                str(scripts / "format_bigbench_audio_judge_input.py"),
                "--manifest",
                environment["BBH_DATASET"],
                "--predictions",
                str(result_dir / "generate_bigbench_audio_s2t.json"),
                "--output",
                str(result_dir / "eval_results.json"),
            ],
            env=environment,
            check=True,
        )
    elif suite == "bfcl":
        subprocess.run(
            [
                sys.executable,
                str(scripts / "report_bfcl.py"),
                str(result_dir / "responses.json"),
                str(root / "ranks/0/data"),
                str(result_dir / "report"),
            ],
            env=environment,
            check=True,
        )
    atomic_json(
        root / "summary.json",
        {
            "suite": suite,
            "world_size": 8,
            "run_key": run_key,
            "merged": True,
            "datasets": summary,
            "partial_dataset": bool(smoke_samples),
            "smoke_samples": smoke_samples,
        },
    )
    logger.info(
        "MERGE_COMPLETE suite=%s datasets=%d samples=%d partial_dataset=%s summary=%s results=%s",
        suite,
        len(summary),
        sum(item["source_count"] for item in summary),
        bool(smoke_samples),
        root / "summary.json",
        result_dir,
    )


def _run_key(environment: Mapping[str, str], suite: str) -> str:
    model = Path(environment["MODEL"])
    tokenizer = Path(environment["TOKENIZER"])
    files = [model / "config.json", model / "model.safetensors.index.json", tokenizer / "tokenizer_config.json"]
    return digest(
        {
            "suite": suite,
            "model": str(model.resolve()),
            "tokenizer": str(tokenizer.resolve()),
            "metadata": [file_digest(path) for path in files],
            "world_size": 8,
            "smoke_samples": int(environment.get("SPEECH_SMOKE_SAMPLES", "0")),
            "source": file_digest(Path(__file__)),
        }
    )


def _check_peers(store, deadline: float) -> None:
    if store.check(["failure"]):
        raise RuntimeError(store.get("failure").decode())
    if time.monotonic() >= deadline:
        raise TimeoutError("MPI evaluation coordination deadline exceeded")


def _run_child(command: list[str], environment: dict[str, str], store, deadline: float) -> None:
    log_path = Path(environment["SPEECH_EVAL_LOG_DIR"]) / "client.log"
    logger.info(
        "WORKER_BEGIN rank=%s gpu=%s port=%s output=%s client_log=%s server_log=%s",
        environment["SPEECH_EVAL_RANK"],
        environment["CUDA_VISIBLE_DEVICES"],
        environment["PORT"],
        environment["OUTPUT_DIR"],
        log_path,
        log_path.with_name("server.log"),
    )
    process = subprocess.Popen(
        command,
        env={**environment, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    relay_errors = []

    def relay():
        try:
            with log_path.open("w") as destination:
                for line in process.stdout:
                    destination.write(line)
                    destination.flush()
                    logger.info("worker | %s", line.rstrip("\n"))
        except Exception as error:
            relay_errors.append(error)

    reader = threading.Thread(target=relay, daemon=True)
    reader.start()
    started = time.monotonic()
    next_heartbeat = started + 30
    try:
        while process.poll() is None:
            _check_peers(store, deadline)
            if relay_errors:
                raise RuntimeError(f"Unable to relay worker log: {relay_errors[0]}")
            if time.monotonic() >= next_heartbeat:
                logger.info(
                    "WORKER_HEARTBEAT rank=%s elapsed_seconds=%.0f client_log=%s output=%s",
                    environment["SPEECH_EVAL_RANK"],
                    time.monotonic() - started,
                    log_path,
                    environment["OUTPUT_DIR"],
                )
                next_heartbeat = time.monotonic() + 30
            threading.Event().wait(0.2)
        if process.returncode:
            raise RuntimeError(
                f"Rank {environment['SPEECH_EVAL_RANK']} exited with {process.returncode}; log: {log_path}"
            )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        reader.join(timeout=10)
    if reader.is_alive() or relay_errors:
        raise RuntimeError(f"Worker log relay did not finish cleanly: {log_path}")
    logger.info(
        "WORKER_DONE rank=%s elapsed_seconds=%.2f output=%s",
        environment["SPEECH_EVAL_RANK"],
        time.monotonic() - started,
        environment["OUTPUT_DIR"],
    )


def main(argv: list[str] | None = None) -> None:
    """Coordinate scheduler ranks, execute one shard each and merge only on rank zero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=tuple(SUITE_OUTPUT_COUNTS))
    parser.add_argument("--coordination-port", type=int, default=29681)
    parser.add_argument("--coordination-timeout", type=int, default=72000)
    parser.add_argument("--check-ranks", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    rank, world_size, device = rank_context(os.environ)
    smoke_samples = int(os.environ.get("SPEECH_SMOKE_SAMPLES", "0"))
    if smoke_samples and (args.suite != "openasr" or not 1 <= smoke_samples <= 128):
        raise ValueError("Smoke requires OpenASR with 1..128 samples")
    if not 1 <= args.coordination_port <= 65535 or args.coordination_timeout < 1:
        raise ValueError("Invalid MPI coordination settings")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command and not args.check_ranks:
        raise ValueError("Missing rank worker command")
    root = Path(os.environ["AMLT_OUTPUT_DIR"]).resolve() / args.suite
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [mpi rank={rank}] %(levelname)s %(message)s")
    logger.info(
        "MPI_RANK_START suite=%s rank=%d/%d gpu=%s model=%s tokenizer=%s output_root=%s",
        args.suite,
        rank,
        world_size,
        device,
        os.environ.get("MODEL"),
        os.environ.get("TOKENIZER"),
        root,
    )
    import torch.distributed as dist

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Rank {rank} received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    timeout = timedelta(seconds=args.coordination_timeout)
    store = dist.TCPStore("127.0.0.1", args.coordination_port, world_size, rank == 0, timeout)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size, timeout=timeout)
    try:
        run_key = "rank-check" if args.check_ranks else _run_key(os.environ, args.suite)
        records = [None] * world_size
        dist.all_gather_object(
            records, {"rank": rank, "device": device, "output": str(root), "suite": args.suite, "run_key": run_key}
        )
        validate_ranks(records)
        logger.info(
            "MPI_READY suite=%s ranks=%d distinct_gpus=%d output_root=%s",
            args.suite,
            world_size,
            len({record["device"] for record in records}),
            root,
        )
        if args.check_ranks:
            if rank == 0:
                atomic_json(root / "rank-check.json", records)
            dist.barrier()
            return
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            (root / "ranks").mkdir(exist_ok=False)
            atomic_json(root / "rank-plan.json", records)
        dist.barrier()
        rank_root = root / "ranks" / str(rank)
        rank_root.mkdir(exist_ok=False)
        handler = logging.FileHandler(rank_root / "rank.log")
        handler.setFormatter(logging.Formatter(f"%(asctime)s [mpi rank={rank}] %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.info(
            "MPI_OUTPUTS rank_log=%s rank_dir=%s merged_results=%s summary=%s",
            rank_root / "rank.log",
            rank_root,
            root / "results",
            root / "summary.json",
        )
        environment = child_environment(os.environ)
        environment.pop("SPEECH_BASE_URL", None)
        environment.pop("SPEECH_BASE_URLS", None)
        environment.update(
            CUDA_VISIBLE_DEVICES=device,
            SPEECH_NUM_SERVERS="1",
            SPEECH_EVAL_RANK=str(rank),
            SPEECH_EVAL_WORLD_SIZE=str(world_size),
            SPEECH_EVAL_RUN_KEY=run_key,
            SPEECH_EVAL_LOG_DIR=str(rank_root),
            OUTPUT_DIR=str(rank_root / "results"),
            PORT=str(int(os.environ.get("PORT", "18281")) + rank),
            POSTPROCESS="0",
        )
        atomic_json(rank_root / "state.json", {"rank": rank, "status": "running"})
        deadline = time.monotonic() + args.coordination_timeout
        try:
            _run_child(command, environment, store, deadline)
        except BaseException as error:
            atomic_json(rank_root / "state.json", {"rank": rank, "status": "failed", "error": str(error)})
            raise
        atomic_json(rank_root / "state.json", {"rank": rank, "status": "complete"})
        store.set(f"done/{rank}", "1")
        logger.info("MPI_SHARD_DONE rank=%d state=%s; waiting for remaining ranks", rank, rank_root / "state.json")
        while not store.check([f"done/{index}" for index in range(world_size)]):
            _check_peers(store, deadline)
            threading.Event().wait(0.2)
        if rank == 0:
            logger.info("MERGE_BEGIN suite=%s rank_root=%s results=%s", args.suite, root / "ranks", root / "results")
            merge_suite(
                root, suite=args.suite, run_key=run_key, environment={**environment, "CUDA_VISIBLE_DEVICES": ""}
            )
            store.set("merged", "1")
        while not store.check(["merged"]):
            _check_peers(store, deadline)
            threading.Event().wait(0.2)
        dist.barrier()
        logger.info("MPI_COMPLETE suite=%s summary=%s", args.suite, root / "summary.json")
    except BaseException as error:
        logger.exception("MPI_FAILED suite=%s rank=%d output_root=%s", args.suite, rank, root)
        store.set("failure", f"Rank {rank}: {error}")
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
