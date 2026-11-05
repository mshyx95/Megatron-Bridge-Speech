import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

import pytest
import yaml
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[4]
CONFIGS = ROOT / "projects/ELIFT"
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


@pytest.fixture
def submission():
    return _submission("b2.yaml")


def _submission(filename):
    path = ROOT / "amulet/ELIFT/train" / filename
    text = path.read_text()
    sources = [
        (path.parent / include).read_text() for include in re.findall(r"^  - !include (\S+)", text, re.MULTILINE)
    ]
    body = re.sub(r"^requires:\n(?:  - !include [^\n]+\n)+\n", "", text, flags=re.MULTILINE)
    return yaml.safe_load("\n".join([*sources, body]))


@pytest.fixture
def explicit_inputs(tmp_path):
    encoder = LAUNCH["encoders"]["phi4-mm-24l"]
    return {
        "name": "custom-run-not-in-catalog",
        "train_config": Path("projects/ELIFT/train/b2.yaml"),
        "data_config": Path("projects/ELIFT/data/mixture.yaml"),
        "data_recipe": "base",
        "data_hindi": None,
        "base_hf": LAUNCH["base_models"]["Qwen3.5-9B"],
        "encoder_name": "phi4-mm-24l",
        "encoder_checkpoint": encoder["checkpoint"],
        "encoder_num_blocks": encoder["num_blocks"],
        "mean_file": encoder["mean_file"],
        "invstd_file": encoder["invstd_file"],
        "output_dir": str(tmp_path / "custom-run"),
    }


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
    train_config = f"projects/ELIFT/{profile['train_config']}"
    data_config = f"{plan.output}/data.yaml"
    assert (ROOT / train_config).is_file()
    assert plan.finetune[2:6] == ["--train-config-yaml", train_config, "--hf-path", plan.model_hf]
    assert ["--streaming-config-yaml", data_config] == plan.finetune[6:8]
    # The data tokenizer, the prepared base LLM and the encoder block count must agree.
    base = LAUNCH["base_models"][profile["language_model"]]
    assert plan.prepare_data[plan.prepare_data.index("--tokenizer") + 1] == base == plan.provenance["base_hf"]
    assert plan.prepare_data[plan.prepare_data.index("--recipe") + 1] == profile["data_recipe"]
    assert plan.prepare_data[plan.prepare_data.index("--output") + 1] == data_config
    if "data_hindi" in profile:
        assert plan.prepare_data[plan.prepare_data.index("--hindi") + 1] == profile["data_hindi"]
    else:
        assert "--hindi" not in plan.prepare_data
    assert plan.provenance["num_blocks"] == LAUNCH["encoders"][profile["encoder"]]["num_blocks"]
    assert plan.finetune[-2:] == (
        ["--init-checkpoint", "/modelblob/native/ckpt/iter_0004000"] if stage in CONTINUATION else ["--name", stage]
    )
    assert "ds_train" not in " ".join(plan.finetune + plan.prepare)
    assert plan.output.endswith(f"/{stage}") and plan.model_hf == f"{plan.output}/initial_hf"


@pytest.mark.unit
@pytest.mark.parametrize("stage", ["b1", "e1r-hindi-2k", "new-tier1-hindi-2k-e1r", "new-tier1-hindi-5k-e1r-8k"])
def test_plan_renders_a_standalone_runtime_config_without_preparing_the_model(tool, tmp_path, stage):
    plan = tool.build_plan(stage, output_root=str(tmp_path))
    subprocess.run(plan.prepare_data, cwd=ROOT, check=True, capture_output=True, text=True)
    path = Path(plan.finetune[plan.finetune.index("--streaming-config-yaml") + 1])
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert len(data["dataset"]) == 17
    assert data["tokenizer"]["pretrained_model_name_or_path"] == plan.provenance["base_hf"]
    assert all(entry["tokenizer"] == data["tokenizer"] for entry in data["dataset"])
    assert not Path(plan.model_hf).exists()


@pytest.mark.unit
@pytest.mark.parametrize("data_failure", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_stage_prepares_data_before_training_and_propagates_composition_failures(
    tool, submission, tmp_path, monkeypatch, data_failure, explicit
):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    if explicit:
        tokens = shlex.split(Template(submission["jobs"][0]["command"][-1]).substitute(submission["env_defaults"]))
        argv = tokens[tokens.index("examples/models/speech_cascades/launch_elift_stage.py") :]
        argv += ["--output-dir", str(tmp_path)]
    else:
        argv = ["launch_elift_stage.py", "--stage", "b2", "--output-root", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []
    monkeypatch.setattr(tool, "prepare_model", lambda plan: calls.append("model"))

    def run(command, check):
        if "prepare_elift_data.py" in command[1]:
            calls.append("data")
            assert check is True
            if data_failure:
                raise subprocess.CalledProcessError(1, command)
        else:
            calls.append("train")
            assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tool.subprocess, "run", run)
    if data_failure:
        with pytest.raises(RuntimeError, match="non-zero exit status 1"):
            tool.main()
        assert calls == ["data"]
    else:
        with pytest.raises(SystemExit) as result:
            tool.main()
        assert result.value.code == 0
        assert calls == ["data", "model", "train"]


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
def test_amulet_job_runs_the_native_launcher_on_h200_with_original_data_mounts(submission):
    config = submission
    inputs = config["env_defaults"]
    assert inputs["TRAIN_CONFIG"] == "projects/ELIFT/train/b2.yaml"
    assert inputs["DATA_CONFIG"] == "projects/ELIFT/data/mixture.yaml"
    assert inputs["DATA_RECIPE"] == "base" and inputs["DATA_HINDI"] == "none"
    assert "STAGE" not in inputs
    assert config["environment"]["image"] == "${IMAGE}" and config["env_defaults"]["IMAGE"].startswith("speechllm:")
    assert config["storage"]["data_blob"]["mount_dir"] == "/datablob1"
    assert config["storage"]["model_blob"]["mount_dir"] == "/modelblob"
    job = config["jobs"][0]
    assert job["sku"] == "${NODES}x${SKU}" and config["env_defaults"]["SKU"].endswith("H200-IB-NvLink")
    assert job["mpi"] is True and job["process_count_per_node"] == 8
    assert '--train-config-yaml "${TRAIN_CONFIG}"' in job["command"][-1]
    assert '--data-config-yaml "${DATA_CONFIG}"' in job["command"][-1]
    assert "ds_train" not in " ".join(job["command"])


@pytest.mark.unit
def test_public_training_command_bypasses_catalog_and_preserves_the_b2_plan(tool, submission, monkeypatch, capsys):
    expected = tool.build_plan("b2")
    command = Template(submission["jobs"][0]["command"][-1]).substitute(submission["env_defaults"])
    tokens = shlex.split(command)
    argv = tokens[tokens.index("examples/models/speech_cascades/launch_elift_stage.py") :] + ["--dry-run"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(tool, "build_plan", lambda *args, **kwargs: pytest.fail("Public command looked up a stage"))
    monkeypatch.setattr(tool.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry run executed a command"))
    load = tool._load

    def load_explicit(path):
        assert path.name not in {"catalog.yaml", "launch.yaml"}
        return load(path)

    monkeypatch.setattr(tool, "_load", load_explicit)
    tool.main()
    assert json.loads(capsys.readouterr().out) == expected.__dict__


@pytest.mark.unit
@pytest.mark.parametrize("stage", LAUNCHABLE)
def test_public_entry_can_express_every_historical_native_plan(tool, submission, tmp_path, monkeypatch, capsys, stage):
    expected = tool.build_plan(stage, configs=_configs_with_checkpoints(tmp_path))
    profile = CATALOG["stages"][stage]
    encoder = LAUNCH["encoders"][profile["encoder"]]
    compression = profile.get("data_streaming_compression")
    values = {
        **submission["env_defaults"],
        "RUN_NAME": stage,
        "OUTPUT_ROOT": str(Path(expected.output).parent),
        "TRAIN_CONFIG": f"projects/ELIFT/{profile['train_config']}",
        "DATA_RECIPE": profile["data_recipe"],
        "DATA_HINDI": profile.get("data_hindi", "none"),
        "DATA_STREAMING_COMPRESSION": "recipe" if compression is None else "on" if compression else "off",
        "BASE_MODEL_ROOT": str(Path(LAUNCH["base_models"][profile["language_model"]]).parent),
        "BASE_MODEL_NAME": Path(LAUNCH["base_models"][profile["language_model"]]).name,
        "ENCODER_NAME": profile["encoder"],
        "ENCODER_DIR": str(Path(encoder["checkpoint"]).parent),
        "ENCODER_NUM_BLOCKS": encoder["num_blocks"],
        "NORMALIZATION_DIR": str(Path(encoder["mean_file"]).parent),
        "INIT_CHECKPOINT": expected.provenance.get("init_checkpoint", "none"),
    }
    tokens = shlex.split(Template(submission["jobs"][0]["command"][-1]).substitute(values))
    argv = tokens[tokens.index("examples/models/speech_cascades/launch_elift_stage.py") :] + ["--dry-run"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(tool, "build_plan", lambda *args, **kwargs: pytest.fail("Explicit entry looked up a stage"))
    tool.main()
    assert json.loads(capsys.readouterr().out) == expected.__dict__


@pytest.mark.unit
@pytest.mark.parametrize(
    ("filename", "stage"),
    [
        ("b2.yaml", "b2"),
        ("p4.yaml", "p4"),
        ("p4-hindi-2k.yaml", "p4-hindi-2k"),
        ("p5.yaml", "p5"),
        ("new-hindi-2k.yaml", "new-tier1-hindi-2k-e1r"),
    ],
)
def test_five_public_training_commands_match_their_catalog_contracts(tool, tmp_path, filename, stage):
    config = _submission(filename)
    values = config["env_defaults"]
    if stage in {"p4", "p5"}:
        assert values["INIT_CHECKPOINT"] == "none"
        values = {**values, "INIT_CHECKPOINT": "/modelblob/native/iter_0004000"}
    checkpoint = Template(str(values["INIT_CHECKPOINT"])).substitute(values)
    expected = tool.build_plan(stage, configs=_configs_with_checkpoints(tmp_path, checkpoint=checkpoint))
    command = Template(config["jobs"][0]["command"][-1]).substitute(values)
    invocation = command[command.index("examples/models/speech_cascades/launch_elift_stage.py") :]
    script = f"export OUTPUT_ROOT={shlex.quote(values['OUTPUT_ROOT'])}\n{shlex.quote(sys.executable)} {invocation} --dry-run"
    result = subprocess.run(["bash", "-e", "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == expected.__dict__


@pytest.mark.unit
@pytest.mark.parametrize("filename", ["p4.yaml", "p5.yaml"])
def test_public_mtp_entries_refuse_an_unconfigured_native_initializer(tool, filename):
    config = _submission(filename)
    command = Template(config["jobs"][0]["command"][-1]).substitute(config["env_defaults"])
    invocation = command[command.index("examples/models/speech_cascades/launch_elift_stage.py") :]
    result = subprocess.run(
        ["bash", "-e", "-c", f"{shlex.quote(sys.executable)} {invocation} --dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "complete native Megatron checkpoint" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    ("recipe", "hindi", "compression"),
    [("base", None, None), ("base", "2k", False), ("new-2k", "2k", None), ("new-5k", "5k", True)],
)
def test_explicit_data_choices_render_before_model_preparation(tool, explicit_inputs, recipe, hindi, compression):
    inputs = {**explicit_inputs, "data_recipe": recipe, "data_hindi": hindi, "streaming_compression": compression}
    plan = tool.build_explicit_plan(**inputs)
    assert plan.stage not in CATALOG["stages"]
    subprocess.run(plan.prepare_data, cwd=ROOT, check=True, capture_output=True, text=True)
    data = OmegaConf.to_container(OmegaConf.load(f"{plan.output}/data.yaml"), resolve=True)
    assert len(data["dataset"]) == 17
    assert data["tokenizer"]["pretrained_model_name_or_path"] == inputs["base_hf"]
    if compression is not None:
        assert data["streaming_compression"] is compression
    assert not Path(plan.model_hf).exists()


@pytest.mark.unit
def test_explicit_mtp_initialization_is_required_and_passed_through(tool, explicit_inputs):
    inputs = {**explicit_inputs, "train_config": Path("projects/ELIFT/train/p4.yaml")}
    with pytest.raises(ValueError, match="complete native Megatron checkpoint"):
        tool.build_explicit_plan(**inputs)
    plan = tool.build_explicit_plan(**inputs, init_checkpoint="/modelblob/native/iter_0004000")
    assert plan.finetune[-2:] == ["--init-checkpoint", "/modelblob/native/iter_0004000"]
    assert plan.provenance["init_checkpoint"] == "/modelblob/native/iter_0004000"


@pytest.mark.unit
@pytest.mark.parametrize(
    "key",
    [
        "base_hf",
        "encoder_checkpoint",
        "mean_file",
        "invstd_file",
        "output_dir",
        "init_checkpoint",
        "pretrained_checkpoint",
    ],
)
def test_explicit_runtime_paths_must_be_absolute(tool, explicit_inputs, key):
    with pytest.raises(ValueError, match="absolute"):
        tool.build_explicit_plan(**{**explicit_inputs, key: "relative/path"})


@pytest.mark.unit
def test_standalone_data_is_passed_through_and_cannot_silently_drop_overlays(tool, explicit_inputs):
    inputs = {
        **explicit_inputs,
        "data_recipe": None,
        "data_config": Path("projects/ELIFT/data/data_en_asr_s1.yaml"),
    }
    plan = tool.build_explicit_plan(**inputs)
    assert plan.prepare_data == []
    assert plan.finetune[6:8] == ["--streaming-config-yaml", str(inputs["data_config"])]
    with pytest.raises(ValueError, match="overlays require a composed data recipe"):
        tool.build_explicit_plan(**{**inputs, "data_hindi": "2k"})
    with pytest.raises(ValueError, match="overlays require a composed data recipe"):
        tool.build_explicit_plan(**inputs, streaming_compression=False)
    with pytest.raises(ValueError, match="requires data/mixture.yaml"):
        tool.build_explicit_plan(**{**inputs, "data_recipe": "base"})


@pytest.mark.unit
def test_pretrained_checkpoint_and_initialization_cannot_be_confused(tool, explicit_inputs, tmp_path):
    checkpoint = "/modelblob/native/iter_0004000"
    with pytest.raises(ValueError, match="not both"):
        tool.build_explicit_plan(**explicit_inputs, init_checkpoint=checkpoint, pretrained_checkpoint=checkpoint)
    config = tmp_path / "lora.yaml"
    config.write_text(yaml.safe_dump({"s2_kd": False, "lora_rank": 32}))
    inputs = {**explicit_inputs, "train_config": config}
    with pytest.raises(ValueError, match="use PRETRAINED_CHECKPOINT"):
        tool.build_explicit_plan(**inputs, init_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="LoRA requires"):
        tool.build_explicit_plan(**inputs)
    plan = tool.build_explicit_plan(**inputs, pretrained_checkpoint=checkpoint)
    assert plan.finetune[-2:] == ["--pretrained-checkpoint", checkpoint]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--train-config-yaml", "projects/ELIFT/train/b2.yaml"], "Explicit training is missing inputs"),
        (["--stage", "b2", "--data-recipe", "base"], "cannot be combined with explicit training inputs"),
        (
            ["--train-config-yaml", "projects/ELIFT/train/b2.yaml", "--output-root", "/modelblob/run"],
            "Use --output-dir for explicit training",
        ),
    ],
)
def test_explicit_cli_rejects_missing_or_ambiguous_inputs(tool, monkeypatch, capsys, arguments, message):
    monkeypatch.setattr(sys, "argv", ["launch_elift_stage.py", *arguments])
    with pytest.raises(SystemExit) as result:
        tool.main()
    assert result.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.unit
def test_standalone_data_cli_skips_composition_but_still_prepares_the_model(tool, submission, tmp_path, monkeypatch):
    config = tmp_path / "s1.yaml"
    config.write_text(yaml.safe_dump({"s2_kd": False, "load_hf_weights": True}))
    values = {
        **submission["env_defaults"],
        "TRAIN_CONFIG": str(config),
        "DATA_CONFIG": "projects/ELIFT/data/data_en_asr_s1.yaml",
        "DATA_RECIPE": "none",
        "OUTPUT_ROOT": str(tmp_path),
        "RUN_NAME": "run",
    }
    tokens = shlex.split(Template(submission["jobs"][0]["command"][-1]).substitute(values))
    monkeypatch.setattr(sys, "argv", tokens[tokens.index("examples/models/speech_cascades/launch_elift_stage.py") :])
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    calls = []
    monkeypatch.setattr(tool, "prepare_model", lambda plan: calls.append("model"))

    def run(command, check):
        assert command[1].endswith("/finetune.py")
        assert command[command.index("--streaming-config-yaml") + 1] == values["DATA_CONFIG"]
        assert check is False
        calls.append("train")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tool.subprocess, "run", run)
    with pytest.raises(SystemExit) as result:
        tool.main()
    assert result.value.code == 0
    assert calls == ["model", "train"]
