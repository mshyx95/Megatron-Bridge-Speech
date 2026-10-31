import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
ENTRY = ROOT / "examples/models/speech_cascades"
pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def local_package(monkeypatch):
    monkeypatch.syspath_prepend(str(ENTRY))


@pytest.fixture
def export_files(tmp_path):
    source = tmp_path / "source"
    checkpoint = tmp_path / "iter_0004000"
    output = tmp_path / "export"
    for directory in (source, checkpoint, output):
        directory.mkdir()
    for name in (".metadata", "run_config.yaml"):
        (checkpoint / name).write_text("source metadata")
    for directory in (source, output):
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3_5AudioForCausalLM"],
                    "model_type": "qwen3_5_text",
                }
            )
        )
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "model.safetensors"}}))
    (output / "model.safetensors").write_bytes(b"fixture weight bytes")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (output / name).write_text("{}")
    return output, {"source_hf": source, "checkpoint": checkpoint, "profile": "qwen35_9b", "step": 4000}


def test_export_receipt_is_idempotent_and_never_overwrites(export_files):
    from speech_eval.export_receipt import check_export

    output, args = export_files
    assert check_export(output, action="preflight", **args)["iteration"] == 4000
    report = check_export(output, action="write", **args)
    assert check_export(output, action="verify", **args) == report
    with pytest.raises(FileExistsError):
        check_export(output, action="write", **args)


@pytest.mark.parametrize("change", ["source", "tokenizer", "shard", "profile", "step", "legacy", "traversal"])
def test_export_receipt_rejects_stale_or_incomplete_exports(export_files, change):
    from speech_eval.export_receipt import RECEIPT, check_export

    output, args = export_files
    check_export(output, action="write", **args)
    if change == "source":
        (args["checkpoint"] / ".metadata").write_text("different source")
    elif change == "tokenizer":
        (output / "tokenizer.json").write_text('{"changed": true}')
    elif change == "shard":
        (output / "model.safetensors").unlink()
    elif change == "profile":
        args["profile"] = "qwen36_35b_a3b"
    elif change == "step":
        args["step"] = 8000
    elif change == "legacy":
        (output / RECEIPT).write_text('{"iteration": 4000}')
    else:
        (output / "model.safetensors.index.json").write_text('{"weight_map": {"w": "../model.safetensors"}}')
    with pytest.raises(ValueError):
        check_export(output, action="verify", **args)


@pytest.mark.parametrize("experts,top_k", [(256, 8), (128, 8), (256, 4)])
def test_export_moe_routing_contract(export_files, experts, top_k):
    from speech_eval.export_receipt import check_export

    output, args = export_files
    args["profile"] = "qwen36_35b_a3b"
    (output / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5AudioForCausalLM"],
                "model_type": "qwen3_5_moe_text",
                "num_experts": experts,
                "num_experts_per_tok": top_k,
            }
        )
    )
    if (experts, top_k) == (256, 8):
        assert check_export(output, action="write", **args)["model_type"] == "qwen3_5_moe_text"
    else:
        with pytest.raises(ValueError, match="routing"):
            check_export(output, action="write", **args)


@pytest.mark.parametrize("profile", ["qwen35_9b", "qwen36_35b_a3b"])
def test_compatibility_export_wrapper_reuses_shared_validation(export_files, profile):
    from speech_eval.export_receipt import check_export

    output, args = export_files
    if profile == "qwen36_35b_a3b":
        args["profile"] = profile
        (output / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3_5AudioForCausalLM"],
                    "model_type": "qwen3_5_moe_text",
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                }
            )
        )
    final = output.parent / "published/hf/iter_0004000-vllm"
    final.parent.mkdir(parents=True)
    output.rename(final)
    check_export(final, action="write", **args)
    result = subprocess.run(
        ["bash", str(ROOT / f"projects/cluster/run_{profile}_s1_2605_vllm_export.sh")],
        env={
            **os.environ,
            "SOURCE_HF": str(args["source_hf"]),
            "SOURCE_CHECKPOINT": str(args["checkpoint"]),
            "EXPORT_ROOT": str(final.parents[1]),
            "EXPORT_PYTHON": sys.executable,
            "CKPT_STEP": "4000",
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "Verified existing export" in result.stdout
    assert not final.with_suffix(".partial").exists()


def _suite(root, *, suite="openasr", compact=False):
    from speech_eval.mpi import StridedDataset, atomic_json, file_digest, write_shard

    names = (
        ["generate_librispeech-clean.txt"]
        if suite == "openasr"
        else [
            "generate_earnings21.txt",
            "generate_earnings22.txt",
            "generate_coraal.txt",
            "generate_aa_earnings22_clean.txt",
        ]
    )
    entries = []
    for dataset_index, name in enumerate(names):
        ids = ["a", "b"]
        rows = [{"audio_ids": [audio_id], "generated_texts": ["text"], "label": ["text"]} for audio_id in ids]
        metadata = []
        audits = []
        for rank in range(8):
            shard = StridedDataset([{"audio_id": audio_id} for audio_id in ids], ids, rank=rank, world_size=8)
            selected = [rows[ids.index(row["audio_id"])] for row in shard]
            output = root / "ranks" / str(rank) / "results" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            audit = "".join(
                json.dumps(
                    {
                        "audio_id": row["audio_ids"][0],
                        "status": "ok",
                        "finish_reason": "length" if rank == 1 else "stop",
                    }
                )
                + "\n"
                for row in selected
            )
            output.with_suffix(".requests.jsonl").write_text(audit)
            write_shard(output, selected, shard, {"run_key": "fixture", "task": suite})
            meta = json.loads(output.with_suffix(output.suffix + ".complete.json").read_text())
            metadata.append(meta)
            audits.append(audit)
            if compact:
                output.unlink()
                output.with_suffix(".requests.jsonl").unlink()
        output = root / "results" / name
        atomic_json(output, [*rows, {"final_score": f"WER: {dataset_index + 1}.00, #REF: 20"}])
        output.with_suffix(".requests.jsonl").write_text("".join(audits))
        merged = {
            "run_key": "fixture",
            "world_size": 8,
            "source_count": 2,
            "source_sha256": metadata[0]["source_sha256"],
            "sha256": file_digest(output),
            "audit_sha256": file_digest(output.with_suffix(".requests.jsonl")),
            "rank_sha256": [item["sha256"] for item in metadata],
        }
        atomic_json(output.with_suffix(output.suffix + ".complete.json"), merged)
        entries.append({"output": name, **merged})
    atomic_json(
        root / "summary.json",
        {
            "suite": suite,
            "world_size": 8,
            "run_key": "fixture",
            "merged": True,
            "datasets": entries,
            "partial_dataset": suite == "openasr",
            "smoke_samples": 2 if suite == "openasr" else 0,
        },
    )
    return root


@pytest.mark.parametrize("compact", [False, True])
def test_collector_retains_smoke_denominators_caps_and_precision(tmp_path, compact):
    from speech_eval.collect_results import collect_suite

    root = _suite(tmp_path, compact=compact)
    report = collect_suite(root, model="fixture", stage="S1", step=4000, protocol="test-greedy")
    assert report["partial_dataset"] and report["samples"] == report["requests"] == 2
    assert report["datasets"][0]["output_cap_hits"] == 1
    assert report["aggregate"]["reference_words"] == 20
    assert report["aggregate"]["wer_macro_percent"] == 1
    assert "not a fresh scorer run" in report["aggregate"]["precision"]


def test_collector_separates_three_and_four_set_long_metrics(tmp_path):
    from speech_eval.collect_results import collect_suite

    report = collect_suite(
        _suite(tmp_path, suite="openasr_long"), model="fixture", stage="S1", step=4000, protocol="test-greedy"
    )
    assert report["aggregate"]["wer_macro_percent"] == 2.5
    assert report["standard_three_set"]["wer_macro_percent"] == 2
    assert report["aggregate"]["samples"] == 8 and report["standard_three_set"]["samples"] == 6


@pytest.mark.parametrize(
    "score",
    [
        "BLEU: 35.44, chrF: 63.13",
        "BLEU: BLEU = 35.44 66.6/42.0/28.6/19.7 (BP = 1.000 ratio = 1.010), chrF: 63.13",
    ],
)
def test_collector_supports_native_sacrebleu_rendering(score):
    from speech_eval.collect_results import _metric

    assert _metric(score) == {"bleu": 35.44}


def test_collector_rejects_zero_reference_denominator():
    from speech_eval.collect_results import _metric

    with pytest.raises(ValueError, match="denominator"):
        _metric("WER: 0.00, #REF: 0")


def test_collection_cli_refuses_to_overwrite_a_report(tmp_path):
    root = _suite(tmp_path / "suite", compact=True)
    output = tmp_path / "report.json"
    command = [
        sys.executable,
        "-m",
        "speech_eval.collect_results",
        str(root),
        "--model",
        "fixture",
        "--stage",
        "S1",
        "--step",
        "4000",
        "--protocol",
        "fixture",
        "--output",
        str(output),
    ]
    environment = {**os.environ, "PYTHONPATH": str(ENTRY)}
    subprocess.run(command, env=environment, capture_output=True, check=True)
    original = output.read_bytes()
    result = subprocess.run(command, env=environment, capture_output=True)
    assert result.returncode != 0 and output.read_bytes() == original


@pytest.mark.parametrize("change", ["audit", "prediction", "rank", "summary", "count"])
def test_collector_rejects_tampered_evidence(tmp_path, change):
    from speech_eval.collect_results import collect_suite

    root = _suite(tmp_path, compact=True)
    if change == "audit":
        (root / "results/generate_librispeech-clean.requests.jsonl").write_text("")
    elif change == "prediction":
        (root / "results/generate_librispeech-clean.txt").write_text("[]")
    elif change == "rank":
        (root / "ranks/7/results/generate_librispeech-clean.txt.complete.json").unlink()
    else:
        path = root / "summary.json"
        data = json.loads(path.read_text())
        if change == "summary":
            data["run_key"] = "stale"
        else:
            data["datasets"] = []
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        collect_suite(root, model="fixture", stage="S1", step=4000, protocol="test-greedy")


def test_serving_wrapper_uses_repository_runtime(tmp_path):
    shim = tmp_path / "python"
    shim.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    shim.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ENTRY / "serve_vllm.sh"), "--dry-run"],
        env={**os.environ, "MODEL": "/models/export", "SPEECH_EVAL_PYTHON": str(shim)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines()[:4] == ["-m", "speech_eval.runtime", "launch", "/models/export"]
    assert "--dry-run" in result.stdout


def test_copied_image_entrypoint_preserves_specialized_image_launcher(tmp_path):
    wrapper = tmp_path / "serve-speech-eval"
    shutil.copyfile(ENTRY / "serve_vllm.sh", wrapper)
    launcher = tmp_path / "speech-runtime"
    launcher.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    launcher.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "MODEL": "/models/export",
        "SPEECH_EVAL_SOURCE_DIR": "",
    }
    result = subprocess.run(
        ["bash", str(wrapper), "--dry-run"], env=environment, text=True, capture_output=True, check=True
    )
    assert result.stdout.splitlines()[:2] == ["launch", "/models/export"]
    assert "--dry-run" in result.stdout
    result = subprocess.run(
        ["bash", str(wrapper)],
        env={**environment, "SPEECH_EVAL_SOURCE_DIR": str(tmp_path / "missing")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "must contain" in result.stderr and not result.stdout


def test_eval_wrapper_preserves_existing_server_logs(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("historical evidence\n")
    result = subprocess.run(
        ["bash", str(ENTRY / "run_vllm_eval.sh"), "true"],
        env={
            **os.environ,
            "MODEL": "/unused",
            "SPEECH_EVAL_LOG_DIR": str(tmp_path),
            "SPEECH_EVAL_PYTHON": sys.executable,
            "SPEECH_BASE_URL": "",
            "SPEECH_BASE_URLS": "",
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2 and "Refusing to overwrite" in result.stderr
    assert log.read_text() == "historical evidence\n"
    assert not list(tmp_path.glob(".runtime-*"))
