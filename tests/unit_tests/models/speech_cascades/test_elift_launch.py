import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[4]
CONFIGS = ROOT / "projects/cluster/elift/v0.1"
CATALOG = yaml.safe_load((CONFIGS / "catalog.yaml").read_text())
LAUNCH = yaml.safe_load((CONFIGS / "launch.yaml").read_text())
CONTINUATION = {"p2", "p4", "p5", "p6", "p4-hindi-2k"}
REFERENCE_ONLY = {"e1r-2609", "e1r-2609-hindi-2k"}
LAUNCHABLE = [name for name in CATALOG["stages"] if name not in REFERENCE_ONLY]


@pytest.fixture
def tool():
    path = ROOT / "examples/models/speech_cascades/launch_elift_stage.py"
    spec = importlib.util.spec_from_file_location("elift_launch_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configs_with_checkpoints(tmp_path, checkpoint="/modelblob/native/ckpt/iter_0004000"):
    configs = tmp_path / "elift"
    shutil.copytree(CONFIGS, configs)
    launch = yaml.safe_load((configs / "launch.yaml").read_text())
    launch["init_checkpoints"] = dict.fromkeys(CONTINUATION, checkpoint)
    (configs / "launch.yaml").write_text(yaml.safe_dump(launch))
    return configs


@pytest.mark.unit
@pytest.mark.parametrize("stage", LAUNCHABLE)
def test_every_launchable_stage_builds_a_consistent_native_plan(tool, tmp_path, stage):
    plan = tool.build_plan(stage, configs=_configs_with_checkpoints(tmp_path))
    profile = CATALOG["stages"][stage]
    data = yaml.safe_load((CONFIGS / profile["data_config"]).read_text())
    train_config = f"projects/cluster/elift/v0.1/{profile['train_config']}"
    data_config = f"projects/cluster/elift/v0.1/{profile['data_config']}"
    assert (ROOT / train_config).is_file() and (ROOT / data_config).is_file()
    assert plan.finetune[2:6] == ["--train-config-yaml", train_config, "--hf-path", plan.model_hf]
    assert ["--streaming-config-yaml", data_config] == plan.finetune[6:8]
    # The data tokenizer, the prepared base LLM and the encoder block count must agree.
    base = LAUNCH["base_models"][profile["language_model"]]
    assert data["tokenizer"]["pretrained_model_name_or_path"] == base == plan.provenance["base_hf"]
    assert plan.provenance["num_blocks"] == LAUNCH["encoders"][profile["encoder"]]["num_blocks"]
    assert plan.finetune[-2:] == (
        ["--init-checkpoint", "/modelblob/native/ckpt/iter_0004000"] if stage in CONTINUATION else ["--name", stage]
    )
    assert "ds_train" not in " ".join(plan.finetune + plan.prepare)
    assert plan.output.endswith(f"/{stage}") and plan.model_hf == f"{plan.output}/initial_hf"


@pytest.mark.unit
def test_continuation_stages_require_a_complete_native_checkpoint(tool):
    for stage in CONTINUATION:
        with pytest.raises(ValueError, match="complete native Megatron checkpoint"):
            tool.build_plan(stage)
    plan = tool.build_plan("b2")
    assert "--init-checkpoint" not in plan.finetune


@pytest.mark.unit
def test_reference_only_unknown_and_relative_inputs_are_rejected(tool, tmp_path):
    for stage in REFERENCE_ONLY:
        with pytest.raises(ValueError, match="reference-only"):
            tool.build_plan(stage)
    with pytest.raises(ValueError, match="Unknown ELIFT stage"):
        tool.build_plan("nope")
    with pytest.raises(ValueError, match="absolute"):
        tool.build_plan("b2", output_root="relative/output")
    with pytest.raises(ValueError, match="absolute"):
        tool.build_plan("p4", configs=_configs_with_checkpoints(tmp_path, checkpoint="relative/ckpt"))


@pytest.mark.unit
def test_prepared_model_is_created_once_and_provenance_is_enforced(tool, tmp_path, monkeypatch):
    plan = tool.build_plan("b2", output_root=str(tmp_path))
    calls = []

    def fake_run(command, check):
        calls.append(command)
        hf = Path(command[command.index("--output-dir") + 1])
        hf.mkdir(parents=True)
        config = {"audio_processor": {"config": {"num_blocks": 24}}, "embd_layer": {"projection_cls": "mlp"}}
        (hf / "config.json").write_text(json.dumps(config))

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    tool.prepare_model(plan)
    tool.prepare_model(plan)
    assert len(calls) == 1
    changed = tool.Plan(**{**plan.__dict__, "provenance": {**plan.provenance, "num_blocks": 23}})
    with pytest.raises(ValueError, match="different provenance"):
        tool.prepare_model(changed)
    stale = tool.build_plan("b1", output_root=str(tmp_path))
    Path(stale.model_hf).mkdir(parents=True)
    (Path(stale.model_hf) / "leftover").write_text("x")
    with pytest.raises(ValueError, match="Incomplete prepared model"):
        tool.prepare_model(stale)


@pytest.mark.unit
def test_dry_run_prints_the_plan_without_running_anything():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/models/speech_cascades/launch_elift_stage.py"),
            "--stage",
            "b2",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["stage"] == "b2" and plan["finetune"][-2:] == ["--name", "b2"]


@pytest.mark.unit
def test_amulet_job_runs_the_native_launcher_on_h200_with_original_data_mounts():
    job_file = ROOT / "projects/cluster/amlt_elift_h200.yaml"
    text = job_file.read_text()
    config = yaml.safe_load(text)
    assert config["env_defaults"]["STAGE"] in CATALOG["stages"]
    assert config["environment"]["image"] == "${IMAGE}" and config["env_defaults"]["IMAGE"].startswith("speechllm:")
    assert config["storage"]["data_blob"]["mount_dir"] == "/datablob1"
    assert config["storage"]["model_blob"]["mount_dir"] == "/modelblob"
    job = config["jobs"][0]
    assert job["sku"] == "${NODES}x${SKU}" and config["env_defaults"]["SKU"].endswith("H200-IB-NvLink")
    assert job["mpi"] is True and job["process_count_per_node"] == 8
    assert "launch_elift_stage.py --stage ${STAGE}" in job["command"][-1]
    assert "ds_train" not in " ".join(job["command"]) and "SPEECH_" not in text
