import ast
import importlib.abc
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
ENTRY = ROOT / "examples/models/speech_cascades"
PACKAGE = ENTRY / "speech_eval"


@pytest.fixture(autouse=True)
def isolated_imports(monkeypatch):
    class BlockExternal(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "phyagi" or fullname.startswith(("phyagi.", "projects.inference_vllm")):
                raise ImportError(f"External project must not be imported: {fullname}")

    monkeypatch.syspath_prepend(str(ROOT / "src/megatron/bridge/models/speech_cascades"))
    monkeypatch.syspath_prepend(str(ENTRY))
    monkeypatch.setattr(sys, "meta_path", [BlockExternal(), *sys.meta_path])


@pytest.mark.unit
def test_imported_eval_entry_emits_paths_progress_and_bounded_samples(tmp_path):
    code = """
import os, runpy, sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import eval_vllm
from speech_eval import speech_task
from speech_eval.speech_api import SpeechAPI
output = sys.argv[1]
sample = {"source": "Transcribe<audio>", "audio": np.zeros(160), "fs": 16000,
          "audio_id": "sample", "label": "hello"}
client = SpeechAPI("http://localhost:8000/v1", "mock-model", output + ".audit", transport=lambda payload: {
    "choices": [{"finish_reason": "stop", "message": {"content": "hello"}}]})
speech_task.DATA_SET_DICT["openasr"] = lambda *args, **kwargs: [sample, dict(sample, audio_id="sample2")]
sys.argv = ["eval.py", "/model", "-t", "/tokenizer", "-tn", "openasr", "-st", "hfwer_en", "-dp", "/data clean",
            "-op", output, "-bs", "1", "-ct", "qwen3_5_audio_nothink_instruct", "-if", "true", "-ap", "lfg3"]
with patch.object(eval_vllm, "load_official_tokenizer", return_value=SimpleNamespace(eos_token="end")), patch.object(eval_vllm, "_client", return_value=client):
    runpy.run_path(os.environ["EVAL_SCRIPT"], run_name="__main__")
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(ENTRY) + os.pathsep + str(ROOT / "src/megatron/bridge/models/speech_cascades"),
        "EVAL_SCRIPT": str(PACKAGE / "scripts/audio_eval/eval.py"),
        "SPEECH_LOG_SAMPLES": "1",
    }
    output = tmp_path / "predictions.json"
    result = subprocess.run(
        [sys.executable, "-c", code, str(output)], env=environment, text=True, capture_output=True, check=True
    )
    log = result.stdout + result.stderr
    for marker in ("EVAL_LOAD", "EVAL_BEGIN", "EVAL_SAMPLE", "EVAL_PROGRESS", "EVAL_DONE", "WER: 0.00", str(output)):
        assert marker in log
    assert log.count("EVAL_SAMPLE") == 1
    assert 'reference="hello" hypothesis="hello"' in log
    assert len(json.loads(output.read_text())) == 3


@pytest.mark.unit
def test_logging_smoke_limits_audio_reads_and_merges_only_selected_records(tmp_path, monkeypatch, caplog):
    from speech_eval import hf_dataset, speech_task
    from speech_eval.mpi import merge_suite
    from speech_eval.speech_api import SpeechAPI

    accessed = []

    class Source:
        def __len__(self):
            return 1000

        def __getitem__(self, index):
            accessed.append(index)
            assert index < 32
            return {"id": str(index), "text": "hello", "audio": {"array": np.zeros(16000), "sampling_rate": 16000}}

    monkeypatch.setattr(hf_dataset, "load_dataset", lambda *args, **kwargs: Source())
    caplog.set_level("INFO")
    root = tmp_path / "openasr"
    for rank in range(8):
        output = root / "ranks" / str(rank) / "results/generate_librispeech-clean.txt"
        client = SpeechAPI(
            "http://localhost:8000/v1",
            "test",
            output.with_suffix(".requests.jsonl"),
            transport=lambda payload: {"choices": [{"finish_reason": "stop", "message": {"content": "hello"}}]},
        )
        speech_task.SpeechTask.run(
            tokenizer=SimpleNamespace(eos_token="end"),
            client=client,
            task="openasr",
            sub_task="hfwer_en",
            dataset_path="/data librispeech-clean",
            output_file_path=output,
            batch_size=2,
            template="qwen3_5_audio_nothink_instruct",
            asr_prompt="lfg3",
            instruction_first="true",
            max_new_tokens=128,
            shard_rank=rank,
            shard_world_size=8,
            run_key="smoke",
            smoke_samples=32,
        )
        assert len(json.loads(output.read_text())) == 4
    merge_suite(root, suite="openasr", run_key="smoke", environment={"SPEECH_SMOKE_SAMPLES": "32"})
    summary = json.loads((root / "summary.json").read_text())
    assert summary["partial_dataset"] is True and summary["smoke_samples"] == 32
    assert len(summary["datasets"]) == 1 and max(accessed) == 31
    assert "EVAL_SMOKE" in caplog.text and "MERGE_COMPLETE" in caplog.text
    assert "WER: 0.00, #REF: 32" in caplog.text
    with pytest.raises(ValueError, match="Missing or inconsistent"):
        merge_suite(root, suite="openasr", run_key="smoke", environment={})


@pytest.mark.unit
def test_mpi_rank_context_and_disjoint_whole_recording_shards():
    from speech_eval.mpi import StridedDataset, merge_records, rank_context

    dataset = [[{"audio_id": str(index), "segment": segment} for segment in range(2)] for index in range(19)]
    ids = [str(index) for index in range(19)]
    shards = []
    for rank in range(8):
        environment = {
            "OMPI_COMM_WORLD_RANK": str(rank),
            "OMPI_COMM_WORLD_LOCAL_RANK": str(rank),
            "OMPI_COMM_WORLD_SIZE": "8",
            "OMPI_COMM_WORLD_LOCAL_SIZE": "8",
            "CUDA_VISIBLE_DEVICES": "7,5,3,1,6,4,2,0",
        }
        assert rank_context(environment) == (rank, 8, environment["CUDA_VISIBLE_DEVICES"].split(",")[rank])
        assert rank_context({**environment, "CUDA_VISIBLE_DEVICES": f"GPU-{rank}"}) == (rank, 8, f"GPU-{rank}")
        shard = StridedDataset(dataset, ids, rank=rank, world_size=8)
        records = [{"audio_ids": [chunks[0]["audio_id"]], "generated_texts": ["merged segments"]} for chunks in shard]
        shards.append({**shard.receipt(records, {"task": "long"}), "records": records})
    assert [row["audio_ids"][0] for row in merge_records(shards, 8)] == ids
    with pytest.raises(ValueError, match="Missing or duplicate"):
        merge_records(shards[:-1], 8)
    damaged = json.loads(json.dumps(shards))
    damaged[0]["records"][0]["audio_ids"] = ["1"]
    with pytest.raises(ValueError, match="Duplicate"):
        merge_records(damaged, 8)
    with pytest.raises(ValueError, match="mpi: true"):
        rank_context({})
    with pytest.raises(ValueError, match="eight MPI ranks"):
        rank_context({**environment, "OMPI_COMM_WORLD_SIZE": "1"})


@pytest.mark.unit
def test_mpi_empty_shards_are_valid_but_empty_source_is_not():
    from speech_eval.mpi import StridedDataset, merge_records

    dataset = [{"audio_id": "only"}]
    shards = []
    for rank in range(8):
        shard = StridedDataset(dataset, ["only"], rank=rank, world_size=8)
        records = [{"audio_ids": [row["audio_id"]], "generated_texts": ["answer"]} for row in shard]
        shards.append({**shard.receipt(records, {"task": "short"}), "records": records})
    assert len(merge_records(shards, 8)) == 1
    with pytest.raises(ValueError, match="Empty source"):
        StridedDataset([], [], rank=0, world_size=8)


@pytest.mark.unit
@pytest.mark.parametrize(
    "task,sub_task,id_key",
    [("openasr_long", "hfwer_en", "audio_id"), ("voice_bench_s2t", "s2t", "UUID"), ("fleurs", "bleu_EN_DE", "utt_id")],
)
def test_mpi_speech_shards_merge_scores_and_verify_receipts(tmp_path, monkeypatch, task, sub_task, id_key):
    from speech_eval import speech_task
    from speech_eval.audio_utils import apply_chat_template
    from speech_eval.mpi import merge_records, read_shard
    from speech_eval.speech_api import SpeechAPI

    class Dataset:
        data = [{id_key: str(index)} for index in range(3)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            sample = {
                "source": apply_chat_template("Transcribe", "qwen3_5_audio_nothink_instruct"),
                "audio": np.zeros(160),
                "fs": 16000,
                "audio_id": str(index),
                "label": "hello hello" if task == "openasr_long" else "hello",
            }
            return [dict(sample), dict(sample)] if task == "openasr_long" else sample

    monkeypatch.setitem(speech_task.DATA_SET_DICT, task, lambda *args, **kwargs: Dataset())
    shards = []
    for rank in range(8):
        output = tmp_path / str(rank) / "predictions.json"
        client = SpeechAPI(
            "http://localhost:8000/v1",
            "test",
            output.with_suffix(".requests.jsonl"),
            transport=lambda payload: {"choices": [{"finish_reason": "stop", "message": {"content": "hello"}}]},
        )
        speech_task.SpeechTask.run(
            tokenizer=SimpleNamespace(eos_token="<|im_end|>"),
            client=client,
            task=task,
            sub_task=sub_task,
            dataset_path="unused",
            output_file_path=output,
            template="qwen3_5_audio_nothink_instruct",
            asr_prompt="lfg3",
            instruction_first="true",
            max_new_tokens=256,
            batch_size=2,
            shard_rank=rank,
            shard_world_size=8,
            run_key="test",
        )
        shards.append(read_shard(output))
    merged = merge_records(shards, 8)
    assert [row["audio_ids"][0] for row in merged] == ["0", "1", "2"]
    if task == "openasr_long":
        assert speech_task.score_records(merged, task=task, sub_task=sub_task) == "WER: 0.00, #REF: 6"
    elif task == "voice_bench_s2t":
        assert speech_task.score_records(merged, task=task, sub_task=sub_task) is None
    assert all("final_score" not in row for shard in shards for row in shard["records"])
    output.write_text("[] ")
    with pytest.raises(ValueError, match="Modified shard"):
        read_shard(output)


@pytest.mark.unit
def test_mpi_merge_recomputes_global_metric_and_requires_every_dataset(tmp_path):
    from speech_eval.mpi import StridedDataset, merge_suite, write_shard

    root = tmp_path / "openasr"
    for rank in range(8):
        for index in range(7):
            shard = StridedDataset(
                [{"audio_id": "short"}, {"audio_id": "long"}], ["short", "long"], rank=rank, world_size=8
            )
            records = [
                {
                    "audio_ids": [row["audio_id"]],
                    "label": ["hello" if row["audio_id"] == "short" else "hello dear friend"],
                    "generated_texts": ["wrong" if row["audio_id"] == "short" else "hello dear friend"],
                }
                for row in shard
            ]
            output = root / "ranks" / str(rank) / "results" / f"dataset{index}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.with_suffix(".requests.jsonl").write_text(
                "".join(json.dumps({"audio_id": row["audio_ids"][0], "status": "ok"}) + "\n" for row in records)
            )
            write_shard(output, records, shard, {"task": "openasr_long", "sub_task": "hfwer_en", "run_key": "run"})
    merge_suite(root, suite="openasr", run_key="run", environment={})
    merged = json.loads((root / "results/dataset0.json").read_text())
    assert merged[-1]["final_score"] == "WER: 25.00, #REF: 4"
    assert json.loads((root / "summary.json").read_text())["merged"] is True
    (root / "ranks/7/results/dataset0.json.complete.json").unlink()
    with pytest.raises(ValueError, match="Missing or inconsistent dataset"):
        merge_suite(root, suite="openasr", run_key="run", environment={})


@pytest.mark.unit
def test_mpi_bfcl_shards_cover_each_record_once(tmp_path, monkeypatch):
    import eval_vllm
    from speech_eval.mpi import merge_records, read_shard
    from speech_eval.scripts.audio_eval import bfcl_runtime
    from speech_eval.speech_api import SpeechAPI

    rows = [{"id": f"simple_{index}", "function": []} for index in range(3)]
    (tmp_path / "BFCL_v3_simple.json").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(eval_vllm, "load_official_tokenizer", lambda path: SimpleNamespace(eos_token="end"))
    monkeypatch.setattr(bfcl_runtime, "read_audio", lambda row: (np.zeros(160), 16000))
    monkeypatch.setattr(bfcl_runtime, "make_qwen_prompt", lambda tokenizer, row, tools, args: row["id"] + "<audio>")
    monkeypatch.setattr(bfcl_runtime, "qwen_response_to_bfcl", lambda response, tools, names: response)
    shards = []
    for rank in range(8):
        output = tmp_path / str(rank) / "responses.worker0.json"
        client = SpeechAPI(
            "http://localhost:8000/v1",
            "test",
            output.with_suffix(".requests.jsonl"),
            transport=lambda payload: {"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]},
        )
        monkeypatch.setattr(eval_vllm, "_client", lambda destination: client)
        monkeypatch.setenv("SPEECH_EVAL_RANK", str(rank))
        monkeypatch.setenv("SPEECH_EVAL_WORLD_SIZE", "8")
        monkeypatch.setenv("SPEECH_EVAL_RUN_KEY", "run")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval_bfcl.py",
                "--model-family",
                "qwen35-audio",
                "--model",
                "/model",
                "--tokenizer",
                "/model",
                "--data-dir",
                str(tmp_path),
                "--output",
                str(output),
                "--categories",
                "simple",
                "--batch-size",
                "2",
            ],
        )
        eval_vllm.bfcl_main()
        shards.append(read_shard(output))
    assert [row["audio_ids"][0] for row in merge_records(shards, 8)] == [row["id"] for row in rows]
    assert sum(len(shard["records"]) for shard in shards) == 3


@pytest.mark.unit
@pytest.mark.parametrize("scenario", ["check", "merge", "worker_failure", "merge_failure"])
def test_mpi_eight_process_coordination(tmp_path, scenario):
    launcher = shutil.which("mpirun")
    if launcher is None:
        pytest.skip("MPI integration check requires the inference image's mpirun")
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = {
        **os.environ,
        "PYTHONPATH": str(ENTRY) + os.pathsep + str(ROOT / "src/megatron/bridge/models/speech_cascades"),
        "AMLT_OUTPUT_DIR": str(tmp_path),
        "MODEL": str(tmp_path),
        "TOKENIZER": str(tmp_path),
        "CUDA_VISIBLE_DEVICES": "7,5,3,1,6,4,2,0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "GLOO_SOCKET_IFNAME": "lo",
        "OMPI_MCA_btl": "self,tcp",
        "OMPI_MCA_pml": "ob1",
        "SCENARIO": scenario,
    }
    command = [
        launcher,
        "--oversubscribe",
        "-n",
        "8",
        sys.executable,
        "-m",
        "speech_eval.mpi",
        "--suite",
        "voicebench",
        "--coordination-port",
        str(port),
        "--coordination-timeout",
        "40",
    ]
    if scenario == "check":
        command.append("--check-ranks")
    else:
        worker = (
            "import json,os,sys,threading; from pathlib import Path; from speech_eval.mpi import StridedDataset,write_shard\n"
            "rank=int(os.environ['SPEECH_EVAL_RANK']); scenario=os.environ['SCENARIO']\n"
            "assert not any(key.startswith(('OMPI_','PMI_','PMIX_')) for key in os.environ)\n"
            "assert os.environ['SPEECH_NUM_SERVERS']=='1'\n"
            "sys.stdout.write('TEST_WORKER_OUTPUT rank='+str(rank)+'\\n'); sys.stdout.flush()\n"
            "if scenario=='worker_failure':\n"
            " if rank==2: sys.exit(9)\n"
            " threading.Event().wait(60)\n"
            "dataset=[{'audio_id':str(index)} for index in range(19)]\n"
            "shard=StridedDataset(dataset,[row['audio_id'] for row in dataset],rank=rank,world_size=8)\n"
            "records=[{'audio_ids':[row['audio_id']],'generated_texts':['hello'],'label':['hello']} for row in shard]\n"
            "output=Path(os.environ['OUTPUT_DIR'])/'generate_voice_bench_s2t.json'\n"
            "output.parent.mkdir(parents=True)\n"
            "output.with_suffix('.requests.jsonl').write_text(''.join(json.dumps({'audio_id':row['audio_ids'][0],'status':'ok'})+'\\n' for row in records))\n"
            "write_shard(output,records,shard,{'task':'voice_bench_s2t','sub_task':'s2t','run_key':os.environ['SPEECH_EVAL_RUN_KEY']})\n"
            "if scenario=='merge_failure' and rank==2: output.write_text('[]')\n"
        )
        command.extend(["--", sys.executable, "-c", worker])
    result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=90)
    root = tmp_path / "voicebench"
    if scenario in {"worker_failure", "merge_failure"}:
        assert result.returncode != 0
        assert not (root / "summary.json").exists()
        assert "MPI_FAILED" in result.stdout + result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    if scenario == "check":
        records = json.loads((root / "rank-check.json").read_text())
        assert len(records) == 8 and len({record["device"] for record in records}) == 8
    else:
        log = result.stdout + result.stderr
        for marker in (
            "MPI_RANK_START",
            "WORKER_BEGIN",
            "TEST_WORKER_OUTPUT",
            "WORKER_DONE",
            "MERGE_DATASET",
            "MERGE_COMPLETE",
            "MPI_COMPLETE",
        ):
            assert marker in log
        assert str(root / "results") in log
        for rank in range(8):
            assert f"TEST_WORKER_OUTPUT rank={rank}" in (root / "ranks" / str(rank) / "client.log").read_text()
            assert "WORKER_DONE" in (root / "ranks" / str(rank) / "rank.log").read_text()
        rows = json.loads((root / "results/generate_voice_bench_s2t.json").read_text())
        assert [row["audio_ids"][0] for row in rows] == list(map(str, range(19)))
        assert json.loads((root / "summary.json").read_text())["world_size"] == 8
        assert not (root / "results/evaluation").exists()


@pytest.mark.unit
def test_no_external_source_dependencies():
    for path in [ENTRY / "eval_vllm.py", *PACKAGE.rglob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            assert not any(
                name == "phyagi" or name.startswith(("phyagi.", "projects.inference_vllm")) for name in names
            )
    launcher = (ENTRY / "run_vllm_eval.sh").read_text()
    for forbidden in ("/opt/speech-src", "speech-pytorch-python", "speech-runtime launch"):
        assert forbidden not in launcher
    manifest = json.loads((PACKAGE / "migration_sources.json").read_text())
    assert all((PACKAGE / item["destination"]).is_file() for item in manifest)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["standard", "NativeMTP1", "ExpandedMTP4"])
def test_audio_adapter_overlay_preserves_readonly_installation(tmp_path, monkeypatch, mode):
    from speech_eval.runtime import source_overlay

    root = tmp_path / "installed"
    installed = root / "vllm/model_executor/models/qwen3_5_audio.py"
    installed.parent.mkdir(parents=True)
    original = 'raise RuntimeError("Image adapter must not execute")\n'
    installed.write_text(original)
    installed.chmod(0o444)
    installed.parent.chmod(0o555)
    adapter = tmp_path / "adapter.py"
    adapter.write_text('class Qwen3_5AudioForCausalLM:\n    source = "repository"\n')
    adapter.chmod(0o444)
    monkeypatch.setattr(source_overlay, "ADAPTER_PATH", adapter)
    monkeypatch.setenv("SPEECH_RUNTIME_MODE", mode)
    monkeypatch.setenv("SPEECH_RUNTIME_PROPOSER", "stable")
    monkeypatch.setenv("SPEECH_RUNTIME_VLLM_ROOT", str(root))
    spec = source_overlay.OverlayFinder().find_spec(source_overlay.ADAPTER_MODULE, [str(installed.parent)])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.Qwen3_5AudioForCausalLM.source == "repository"
    assert module.__file__ == str(adapter)
    assert installed.read_text() == original
    assert not list(tmp_path.rglob("*.pyc"))
    assert '"$adapter"' not in (ENTRY / "run_vllm_eval.sh").read_text()


@pytest.mark.unit
@pytest.mark.parametrize(
    "server_count,failed_index,native_tools", [(1, None, False), (8, None, False), (8, 3, False), (1, None, True)]
)
def test_wrapper_isolates_scheduler_environment_before_server_start(
    tmp_path, server_count, failed_index, native_tools
):
    server_env = tmp_path / "server"
    client_env = tmp_path / "client.json"
    log_dir = tmp_path / ("long-result-prefix-" * 8) / "ranks/4"
    log_dir.mkdir(parents=True)
    retained = log_dir / "existing-result.json"
    retained.write_text("keep\n")
    inherited_tmp = tmp_path / ("long-inherited-tmp-" * 8)
    inherited_tmp.mkdir()
    shim = tmp_path / "python-shim"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, os, socket, sys, tempfile, uuid\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == '-':\n"
        "    if os.environ.get('SPEECH_SERVER_PID'):\n"
        "        exec(sys.stdin.read())\n"
        "    sys.exit(0)\n"
        "state = dict(os.environ, launch_args=sys.argv[1:])\n"
        "Path(os.environ['SERVER_ENV'] + '-' + os.environ['PORT']).write_text(json.dumps(state))\n"
        "ipc_path = str(Path(tempfile.gettempdir()) / str(uuid.uuid4()))\n"
        "ipc_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "ipc_socket.bind(ipc_path)\n"
        "assert len(os.fsencode(ipc_path)) <= 107\n"
        "if os.environ['PORT'] == os.environ.get('FAILED_PORT'):\n"
        "    sys.exit(5)\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200)\n"
        "        self.end_headers()\n"
        "        self.wfile.write(json.dumps({'data': [{'id': os.environ['SERVED_MODEL_NAME']}]}).encode())\n"
        "HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()\n"
    )
    shim.chmod(0o755)
    distributed = dict.fromkeys(
        [
            "OMPI_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "OMPI_COMM_WORLD_SIZE",
            "OMPI_MCA_orte_hnp_uri",
            "PMI_RANK",
            "PMI_SIZE",
            "PMIX_RANK",
            "TORCHELASTIC_RUN_ID",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ],
        "7",
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {
        **os.environ,
        **distributed,
        "MODEL": "/unused",
        "SPEECH_EVAL_LOG_DIR": str(log_dir),
        "TMPDIR": str(inherited_tmp),
        "SPEECH_EVAL_NATIVE_TOOLS": "1" if native_tools else "0",
        "SPEECH_EVAL_PYTHON": str(shim),
        "SERVER_ENV": str(server_env),
        "PORT": str(port),
        "CUDA_VISIBLE_DEVICES": ",".join(f"GPU-selected-{index}" for index in range(server_count)),
        "SPEECH_BASE_URL": "",
        "SPEECH_BASE_URLS": "",
        "SPEECH_SERVER_PID": "",
        "SPEECH_NUM_SERVERS": str(server_count),
        "VLLM_ROOT": str(tmp_path / "absent-readonly-installation"),
        "FAILED_PORT": str(port + failed_index) if failed_index is not None else "",
    }
    client_code = (
        "import json,os,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(dict(os.environ)))"
    )
    process = subprocess.run(
        ["bash", str(ENTRY / "run_vllm_eval.sh"), sys.executable, "-c", client_code, str(client_env)],
        env=env,
        timeout=20,
        capture_output=True,
        text=True,
    )
    assert retained.read_text() == "keep\n"
    assert inherited_tmp.is_dir()
    for record in tmp_path.glob("server-*"):
        work = Path(json.loads(record.read_text())["TMPDIR"])
        assert work.parent == Path("/tmp")
        assert not work.exists()
    if failed_index is not None:
        assert process.returncode != 0
        assert not client_env.exists()
        for index in range(server_count):
            with socket.socket() as connection:
                assert connection.connect_ex(("127.0.0.1", port + index)) != 0
        return
    assert process.returncode == 0, process.stderr
    assert "SERVER_START" in process.stdout
    assert "SERVER_READY" in process.stdout + process.stderr
    assert "CLIENT_START" in process.stdout
    assert str(log_dir) in process.stdout
    for index in range(server_count):
        captured = json.loads(Path(str(server_env) + f"-{port + index}").read_text())
        assert ("--native-tools" in captured["launch_args"]) is native_tools
        assert distributed.keys().isdisjoint(captured)
        assert captured["CUDA_VISIBLE_DEVICES"] == f"GPU-selected-{index}"
        with socket.socket() as connection:
            assert connection.connect_ex(("127.0.0.1", port + index)) != 0
    captured = json.loads(client_env.read_text())
    assert distributed.keys().isdisjoint(captured)
    assert captured["CUDA_VISIBLE_DEVICES"] == ""
    assert json.loads(captured["SPEECH_BASE_URLS"]) == [
        f"http://127.0.0.1:{port + index}/v1" for index in range(server_count)
    ]


@pytest.mark.unit
@pytest.mark.parametrize("fail_request", [False, True])
def test_eight_endpoints_generate_once_and_preserve_order(tmp_path, fail_request):
    from speech_eval.speech_api import SpeechAPI, SpeechAPIPipeline

    barrier = threading.Barrier(8, timeout=10)
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            source = payload["chat_template_kwargs"]["delivery_prompt"]
            seen.append((self.server.server_port, source))
            assert payload["temperature"] == 0
            assert payload["bad_words"] == ["<think>"]
            if int(source.removesuffix("<audio>")) < 8:
                barrier.wait()
            if fail_request and source == "3<audio>":
                self.send_error(503)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": source}}]}).encode()
            )

    servers = [ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(8)]
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    try:
        client = SpeechAPI(
            [f"http://127.0.0.1:{server.server_port}/v1" for server in servers], "test", tmp_path / "audit.jsonl"
        )
        pipeline = SpeechAPIPipeline(None, SimpleNamespace(eos_token="<|im_end|>"), client=client)
        sources = [f"{index}<audio>" for index in range(19)]
        inputs = {
            "source": sources,
            "audio": [np.zeros(160)] * 19,
            "fs": [16000] * 19,
            "audio_id": list(range(19)),
            "label": sources,
        }
        generation = SimpleNamespace(
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            max_new_tokens=256,
            stop_tokens=[],
            bad_words=["<think>"],
        )
        if fail_request:
            with pytest.raises(RuntimeError, match="HTTP 503"):
                pipeline(inputs, generation_config=generation)
        else:
            result = pipeline(inputs, generation_config=generation)
            assert result["generated_texts"] == sources
            assert result["audio_ids"] == list(range(19))
            assert result["label"] == sources
        observed = Counter(source for _, source in seen)
        if fail_request:
            assert set(observed) <= set(sources)
            assert all(count == 1 for count in observed.values())
        else:
            assert observed == Counter(sources)
        assert len({port for port, _ in seen}) == 8
        audits = [json.loads(line) for line in client.audit_path.read_text().splitlines()]
        assert Counter(f"{row['audio_id']}<audio>" for row in audits) == observed
        assert [row["audio_id"] for row in audits if row["status"] != "ok"] == ([3] if fail_request else [])
        assert len({row["endpoint"] for row in audits}) == 8
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()


@pytest.mark.unit
def test_bfcl_uses_endpoint_pool_and_preserves_row_order(tmp_path, monkeypatch):
    import eval_vllm
    from speech_eval.speech_api import SpeechAPI

    endpoints = [f"http://localhost:{18000 + index}/v1" for index in range(8)]
    monkeypatch.setenv("SPEECH_BASE_URLS", json.dumps(endpoints))
    monkeypatch.setenv("SERVED_MODEL_NAME", "test")
    client = eval_vllm._client(tmp_path / "responses.json")
    assert isinstance(client, SpeechAPI)
    assert client.concurrency == 8
    barrier = threading.Barrier(8, timeout=10)

    def transport(payload):
        assert payload["bad_words"] == ["<think>"]
        barrier.wait()
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": payload["chat_template_kwargs"]["delivery_prompt"]}}
            ]
        }

    client.transport = transport
    monkeypatch.setattr(eval_vllm, "_client", lambda output: client)
    monkeypatch.setattr(eval_vllm, "load_official_tokenizer", lambda path: SimpleNamespace(eos_token="<|im_end|>"))
    rows = [{"id": str(index), "function": []} for index in range(8)]
    reference = SimpleNamespace(
        load_runtime=None,
        read_audio=lambda row: (np.zeros(160), 16000),
        make_tools=lambda functions: ([], []),
        make_qwen_prompt=lambda tokenizer, row, tools, args: row["id"] + "<audio>",
        qwen_response_to_bfcl=lambda response, tools, names: response,
        GenerationResult=lambda text, parsed, latency, metadata: SimpleNamespace(
            raw_response=text, generated_text=parsed, metadata=metadata
        ),
    )

    def main():
        runtime = reference.load_runtime(
            SimpleNamespace(
                model_family="qwen35-audio",
                enable_thinking=False,
                tokenizer="unused",
                output=tmp_path / "responses.json",
                max_new_tokens=512,
                model="/model",
                data_dir=tmp_path,
            )
        )
        results = runtime.generate(rows)
        assert [result.raw_response for result in results] == [row["id"] + "<audio>" for row in rows]
        assert all(result.metadata["backend"] == "vllm" for result in results)

    reference.main = main
    monkeypatch.setattr(eval_vllm, "_reference", lambda name: reference)
    eval_vllm.bfcl_main()


@pytest.mark.unit
def test_official_tokenizer_rules_survive_client_and_server_load(tmp_path):
    from speech_eval.tokenizer import load_official_tokenizer, prepare_tokenizer_view
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
    from transformers import AutoTokenizer

    source = tmp_path / "source"
    source.mkdir()
    backend = Tokenizer(models.BPE(vocab={"[UNK]": 0}, merges=[], unk_token="[UNK]"))
    backend.normalizer = normalizers.NFC()
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.ByteLevel(trim_offsets=False)
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"[\p{L}\p{M}]+|\p{N}|[^\p{L}\p{M}\p{N}]+"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
        ]
    )
    backend.add_special_tokens([AddedToken("<|im_end|>", special=True), AddedToken("<|audio_pad|>", special=True)])
    backend.save(str(source / "tokenizer.json"))
    config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "eos_token": "<|im_end|>",
        "unk_token": "[UNK]",
        "chat_template": "{{ messages[0]['content'] }}",
        "added_tokens_decoder": {
            "1": {"content": "<|im_end|>", "special": True},
            "2": {"content": "<|audio_pad|>", "special": True},
        },
    }
    (source / "tokenizer_config.json").write_text(json.dumps(config))
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    view = tmp_path / "view"
    view.mkdir()
    (view / "tokenizer.json").symlink_to(source / "tokenizer.json")
    prepare_tokenizer_view(source, view)
    client = load_official_tokenizer(source)
    server = AutoTokenizer.from_pretrained(view, local_files_only=True)
    for tokenizer in (client, server):
        loaded = json.loads(tokenizer.backend_tokenizer.to_str())
        assert loaded["pre_tokenizer"] == json.loads(backend.to_str())["pre_tokenizer"]
        for text in ("iPhone JavaScript", "\u0928\u092e\u0938\u094d\u0924\u0947", "e\u0301cole", "<|audio_pad|>"):
            assert (
                tokenizer.encode(text, add_special_tokens=False) == backend.encode(text, add_special_tokens=False).ids
            )
        assert tokenizer.eos_token_id == 1
        assert tokenizer.encode("<|audio_pad|>", add_special_tokens=False) == [2]
        assert tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False) == "hello"
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original
    with pytest.raises(ValueError, match="new and separate"):
        prepare_tokenizer_view(source, source)
    with pytest.raises(ValueError, match="new and separate"):
        prepare_tokenizer_view(source, view)


@pytest.mark.unit
def test_http_contract_and_audit(tmp_path):
    from speech_eval.speech_api import SpeechAPI

    payloads = []

    def transport(payload):
        payloads.append(payload)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "hello"}}]}

    client = SpeechAPI("http://localhost:8000/v1", "test", tmp_path / "audit.jsonl", transport=transport)
    generation = SimpleNamespace(
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        max_new_tokens=256,
        stop_tokens=[],
    )
    assert client.generate("strong<audio>", np.zeros(16000), 16000, "id", generation, "<|im_end|>") == "hello"
    assert payloads[0]["chat_template_kwargs"] == {"delivery_prompt": "strong<audio>"}
    assert payloads[0]["temperature"] == 0
    assert payloads[0]["max_tokens"] == 256
    assert json.loads((tmp_path / "audit.jsonl").read_text())["status"] == "ok"
    with pytest.raises(ValueError, match="placeholders"):
        client.generate("missing", np.zeros(16000), 16000, "id", generation, "<|im_end|>")


@pytest.mark.unit
@pytest.mark.parametrize("task", ["openasr", "openasr_long"])
def test_local_scoring_and_longform_merge(tmp_path, monkeypatch, task):
    from speech_eval import speech_task
    from speech_eval.audio_utils import apply_chat_template
    from speech_eval.speech_api import SpeechAPI

    responses = iter(["hello", "world"] if task.endswith("long") else ["hello world"])
    sample = {
        "source": apply_chat_template("Transcribe", "qwen3_5_audio_nothink_instruct"),
        "audio": np.zeros(16000),
        "fs": 16000,
        "audio_id": "sample",
        "label": "hello world",
    }
    samples = [[dict(sample), dict(sample)]] if task.endswith("long") else [sample]
    monkeypatch.setitem(speech_task.DATA_SET_DICT, task, lambda *args, **kwargs: samples)
    client = SpeechAPI(
        "http://localhost:8000/v1",
        "test",
        tmp_path / "audit.jsonl",
        transport=lambda payload: {"choices": [{"finish_reason": "stop", "message": {"content": next(responses)}}]},
    )
    output = tmp_path / "predictions.json"
    result = speech_task.SpeechTask.run(
        tokenizer=SimpleNamespace(eos_token="<|im_end|>"),
        client=client,
        task=task,
        sub_task="hfwer_en",
        dataset_path="unused",
        output_file_path=output,
        template="qwen3_5_audio_nothink_instruct",
        asr_prompt="lfg3",
        max_new_tokens=256,
        batch_size=1,
        instruction_first="true",
    )
    assert result == "WER: 0.00, #REF: 2"
    assert json.loads(output.read_text())[0]["generated_texts"] == ["hello world"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "suite,count,smoke_samples",
    [("openasr", 7, 0), ("openasr_ml", 14, 0), ("openasr_long", 4, 0), ("fleurs", 56, 0), ("openasr", 1, 32)],
)
def test_original_runner_subset_inventory(tmp_path, monkeypatch, suite, count, smoke_samples):
    from speech_eval.cli import parse_args

    binary = tmp_path / "bin"
    binary.mkdir()
    shim = binary / "python"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    shim.chmod(0o755)
    log = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "CALL_LOG": str(log),
        "OUTPUT_DIR": str(tmp_path / "results"),
        "OPENASR_ESB_ROOT": "/esb",
        "OPENASR_ML_ROOT": "/ml",
        "OPENASR_LONG_ROOT": "/long",
        "OPENASR_AA_ROOT": "/aa",
        "FLEURS_ROOT": "/fleurs",
    }
    env.pop("temperature", None)
    env["SPEECH_SMOKE_SAMPLES"] = str(smoke_samples)
    if suite == "fleurs":
        script = "run_st_eval.sh"
        args = ["id", "exp", "4000", "/tokenizer", suite, "false", "qwen3_5_audio_nothink_instruct", "true"]
    else:
        script = "run_opensr_eval.sh"
        args = [
            "id",
            "exp",
            "4000",
            "/tokenizer",
            suite,
            "1",
            "0",
            "0",
            "qwen3_5_audio_nothink_instruct",
            "true",
            "test",
            "",
            "1",
            "svad",
            "ter",
            "/data",
            "lfg3",
        ]
    subprocess.run(["bash", str(PACKAGE / "scripts/audio_eval" / script), *args], env=env, check=True)
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands) == count
    for command in commands:
        monkeypatch.setattr(sys, "argv", command)
        parsed, extra = parse_args()
        assert extra == []
        assert parsed.instruction_first == "true"
        assert parsed.num_beams == 1
        assert parsed.temperature == 1.0
        assert parsed.dataset_path
        assert parsed.output_file_path
        if smoke_samples:
            assert parsed.dataset_path == "/esb librispeech-clean"
        if suite != "fleurs":
            assert parsed.asr_prompt == "lfg3"
            assert parsed.max_new_tokens == (512 if suite == "openasr_long" else 128)


@pytest.mark.unit
@pytest.mark.parametrize("suite", ["openasr", "openasr_ml", "openasr_long", "bbh", "voicebench", "bfcl", "fleurs"])
@pytest.mark.parametrize("batch_size", [2, 8, 32, 64, 128])
def test_cluster_job_runs_one_suite_with_one_service_per_mpi_rank(tmp_path, suite, batch_size):
    driver = tmp_path / "projects/cluster/run_speech_cascades_b1_vllm.sh"
    driver.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "projects/cluster/run_speech_cascades_b1_vllm.sh", driver)
    wrapper = tmp_path / "examples/models/speech_cascades/run_vllm_eval.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'printf "%s\\n" "$SPEECH_NUM_SERVERS" "$CUDA_VISIBLE_DEVICES" "$SPEECH_EVAL_LOG_DIR" "$*" > "$CALL_LOG"\n'
        'printf "%s\\n" "${BATCH_SIZE:-}" "$MAX_NUM_SEQS" "$SPEECH_EVAL_CONCURRENCY" >> "$CALL_LOG"\n'
    )
    visible = "GPU-rank3"
    env = {
        **os.environ,
        "MODEL": "/model",
        "TOKENIZER": "/model",
        "EXP_ID": "test",
        "AML_EXP_NAME": "test",
        "CKPT_STEP": "4000",
        "AMLT_OUTPUT_DIR": str(tmp_path / "results"),
        "CALL_LOG": str(tmp_path / "calls"),
        "CUDA_VISIBLE_DEVICES": visible,
        "BBH_DATASET": "/bbh",
        "VB_DATASET": "/voicebench",
        "BFCL_SOURCE_DIR": "/bfcl",
        "SPEECH_EVAL_RANK": "3",
        "SPEECH_EVAL_WORLD_SIZE": "8",
        "SPEECH_EVAL_RUN_KEY": "run",
        "PORT": "18284",
        "SPEECH_EVAL_LOG_DIR": str(tmp_path / "results" / suite / "ranks/3"),
        "OUTPUT_DIR": str(tmp_path / "results" / suite / "ranks/3/results"),
        "ASR_BATCH_SIZE": str(batch_size),
        "BENCH_BATCH_SIZE": str(batch_size),
        "ST_BATCH_SIZE": str(batch_size),
        "MAX_NUM_SEQS": str(batch_size),
        "SPEECH_EVAL_CONCURRENCY": str(batch_size),
    }
    subprocess.run(["bash", str(driver), suite], env=env, check=True)
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls[0] == "1"
    assert calls[1] == visible
    assert calls[2] == str(tmp_path / "results" / suite / "ranks/3")
    assert calls[5:] == [str(batch_size), str(batch_size)]
    if suite in {"bbh", "voicebench", "bfcl"}:
        assert calls[4] == str(batch_size)
    else:
        assert f" {batch_size} /datablob1" in calls[3] or f" {batch_size} svad ter" in calls[3]
    missing_suite = subprocess.run(["bash", str(driver)], env=env, capture_output=True)
    assert missing_suite.returncode != 0
    wrong_size = subprocess.run(
        ["bash", str(driver), suite], env={**env, "SPEECH_EVAL_WORLD_SIZE": "1"}, capture_output=True
    )
    assert wrong_size.returncode != 0


@pytest.mark.unit
@pytest.mark.parametrize("inherited", ["0", "1"])
def test_cluster_voicebench_is_generation_only(tmp_path, inherited):
    driver = tmp_path / "projects/cluster/run_speech_cascades_b1_vllm.sh"
    driver.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "projects/cluster/run_speech_cascades_b1_vllm.sh", driver)
    wrapper = tmp_path / "examples/models/speech_cascades/run_vllm_eval.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'printf "%s %s %s\\n" "$RUN_EVAL" "$RUN_JUDGE" "$POSTPROCESS" > "$FLAG_LOG"\n'
        'cd "$RUNNER_DIR"\nexec "$@"\n'
    )
    python_shim = tmp_path / "python"
    python_shim.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
    python_shim.chmod(0o755)
    calls = tmp_path / "calls"
    flags = tmp_path / "flags"
    env = {
        **os.environ,
        "MODEL": "/model",
        "TOKENIZER": "/tokenizer",
        "EXP_ID": "test",
        "AML_EXP_NAME": "test",
        "CKPT_STEP": "4000",
        "AMLT_OUTPUT_DIR": str(tmp_path / "results"),
        "VB_DATASET": "/data/voicebench.jsonl",
        "OMPI_COMM_WORLD_LOCAL_RANK": "0",
        "OMPI_COMM_WORLD_RANK": "0",
        "OMPI_COMM_WORLD_SIZE": "1",
        "SPEECH_EVAL_RANK": "0",
        "SPEECH_EVAL_WORLD_SIZE": "8",
        "SPEECH_EVAL_RUN_KEY": "run",
        "SPEECH_EVAL_LOG_DIR": str(tmp_path / "results/ranks/0"),
        "OUTPUT_DIR": str(tmp_path / "results/ranks/0/results"),
        "PORT": "18281",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHON": str(python_shim),
        "RUNNER_DIR": str(PACKAGE / "scripts/audio_eval"),
        "CALL_LOG": str(calls),
        "FLAG_LOG": str(flags),
        "RUN_EVAL": inherited,
        "RUN_JUDGE": inherited,
        "POSTPROCESS": inherited,
        "MTP_SPECULATIVE": "false",
    }
    subprocess.run(["bash", str(driver), "voicebench"], env=env, check=True)
    assert flags.read_text().strip() == "0 0 0"
    commands = calls.read_text().splitlines()
    assert len(commands) == 1
    assert "eval.py /model" in commands[0]
    assert "-tn voice_bench_s2t" in commands[0]
    assert "-st s2t" in commands[0]
    assert "generate_voice_bench_s2t.json" in commands[0]
    assert "-mt 2048" in commands[0]
