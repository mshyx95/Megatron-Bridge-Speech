import argparse
import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[4]
CONFIGS = ROOT / "projects/cluster/elift/v0.1"
CLI_PATH = ROOT / "examples/models/speech_cascades/finetune.py"
CATALOG = yaml.safe_load((CONFIGS / "catalog.yaml").read_text())


def _load_cli():
    spec = importlib.util.spec_from_file_location("elift_finetune_test", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_complete_elift_inventory_and_sampling_grid():
    assert len(CATALOG["reference_sha256"]) == 18
    assert set(CATALOG["reference_sha256"]) == {p.name for p in (CONFIGS / "reference").glob("*.yaml")}
    assert set(CATALOG["stages"]) == {
        "b1",
        "b2",
        "b3",
        "b4",
        "c",
        "e1r",
        "e1r-hindi-2k",
        "e1r-2609",
        "e1r-2609-hindi-2k",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p4-hindi-2k",
        "new-tier1-hindi-2k-e1r",
        "new-tier1-hindi-5k-e1r-8k",
    }
    assert CATALOG["sampling_hours"] == [2000, 5000, 10000, 20000]
    for stage in CATALOG["stages"].values():
        assert (CONFIGS / stage["reference"]).is_file()
        assert (CONFIGS / stage["base_reference"]).is_file()
        assert stage["training_gpu_qualification"] == "not-run"


@pytest.mark.unit
@pytest.mark.parametrize("name", list(CATALOG["stages"]))
def test_stage_training_profiles_preserve_source_contract(name, monkeypatch):
    stage = CATALOG["stages"][name]
    base = OmegaConf.load(CONFIGS / stage["base_reference"])
    reference = OmegaConf.merge(base, OmegaConf.load(CONFIGS / stage["reference"]))
    if name.endswith("5k-e1r-8k"):
        reference = OmegaConf.merge(reference, OmegaConf.load(CONFIGS / "reference/new-tier1-e1r-8k-steps.yaml"))
    if stage["train_config"] is None:
        assert reference.audio_tower.name == "flash"
        assert stage["runtime_status"] == "reference-only-flash-encoder"
        return
    training = yaml.safe_load((CONFIGS / stage["train_config"]).read_text())
    assert training["train_iters"] == reference.training_args.max_steps
    assert training["full_iters"] == reference.training_args.max_steps
    assert training["lr"] == reference.training_args.ds_config.optimizer.params.lr
    assert training["global_batch_size"] == reference.training_args.ds_config.train_batch_size
    assert training["micro_batch_size"] == reference.training_args.ds_config.train_micro_batch_size_per_gpu
    assert training["save_interval"] == reference.training_args.save_steps == 400
    assert training["lr_warmup_iters"] == reference.training_args.ds_config.scheduler.params.warmup_num_steps == 100
    assert training["seq_length"] == reference.seq_len == 8192
    assert training["s2_kd"] and training["distillation_weight"] == 1.0
    assert training["freeze_language_model"] and not training["freeze_audio_model"]
    assert training["freeze_audio_encoder"] is (name != "p3")
    assert training["lora_rank"] is None
    assert training["mtp_init_extra_layers_from_first"] is False
    depth = reference.model.get("mtp_num_hidden_layers", 0)
    assert training["mtp_num_hidden_layers"] == depth
    assert training["mtp_distillation_weight"] == 0
    assert training["mtp_loss_weight"] == (0.1 if depth else 0)
    assert training["load_hf_weights"] is (depth == 0)
    assert training["mtp_loss_depth_weights"] == list(reference.model.get("mtp_loss_depth_weights", []))
    assert stage["optimizer"] == {"adam_beta1": 0.9, "adam_beta2": 0.95, "adam_eps": 1e-7, "weight_decay": 0.01}
    fields = set()
    for node in ast.walk(ast.parse(CLI_PATH.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for value in node.args:
                if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.startswith("--"):
                    fields.add(value.value[2:].replace("-", "_"))
                    break
    assert set(training) <= fields
    args = argparse.Namespace(**dict.fromkeys(fields))
    args.train_config_yaml = str(CONFIGS / stage["train_config"])
    monkeypatch.setattr(sys, "argv", ["finetune.py", "--train-config-yaml", args.train_config_yaml])
    _load_cli()._apply_train_config_yaml(args, argparse.ArgumentParser())
    for key, value in training.items():
        assert getattr(args, key) == value


@pytest.mark.unit
@pytest.mark.parametrize("name", list(CATALOG["stages"]))
def test_standalone_data_keeps_all_source_slots_and_teacher_transforms(name, monkeypatch):
    stage = CATALOG["stages"][name]
    source = OmegaConf.merge(
        OmegaConf.load(CONFIGS / stage["base_reference"]), OmegaConf.load(CONFIGS / stage["reference"])
    )
    data = OmegaConf.load(CONFIGS / stage["data_config"])
    assert len(data.dataset) == len(source.dataset) == 17
    for original, migrated in zip(source.dataset, data.dataset):
        for field in ("dataset_spec", "filter_options", "weight", "seq_len", "batcher_batch_size", "dataset_provider"):
            assert (
                OmegaConf.to_container(original, resolve=False)[field]
                == OmegaConf.to_container(migrated, resolve=False)[field]
            )
        transform = OmegaConf.to_container(migrated.transform_cls, resolve=False)
        assert transform.pop("knowledge_distillation") is True
        assert transform == OmegaConf.to_container(original.transform_cls, resolve=False)
    assert data.dataset_collator.knowledge_distillation is True
    for variable in (
        "SPEECH_DATA_SPEC_ROOT",
        "SPEECH_REWRITE_ROOT",
        "SPEECH_TIER1_ROOT",
        "SPEECH_HINDI_ROOT",
        "SPEECH_NEW_DATA_2K_SPEC_ROOT",
        "SPEECH_NEW_DATA_5K_SPEC_ROOT",
        "SPEECH_DATA_MOUNT_ROOT",
        "SPEECH_TOKENIZER_PATH",
    ):
        monkeypatch.setenv(variable, "/fixture/" + variable.lower())
    resolved = OmegaConf.to_container(data, resolve=True)
    assert resolved["seq_len"] == 8192
    assert all(entry["transform_cls"]["knowledge_distillation"] for entry in resolved["dataset"])


@pytest.mark.unit
def test_no_private_runtime_bindings_in_migrated_configs():
    forbidden = (
        "/home/",
        "/datablob",
        "/modelblob",
        "users/",
        "blob.core.windows.net",
        "WANDB_RUN_ID",
        "workspace_name",
    )
    for path in CONFIGS.rglob("*.yaml"):
        content = path.read_text()
        assert not any(token in content for token in forbidden), path


@pytest.mark.unit
@pytest.mark.parametrize("stage", ["p3", "p4-hindi-2k"])
def test_new_encoder_and_mtp_flags_reach_the_actual_finetune_dispatch(stage, monkeypatch):
    cli = _load_cli()
    captured = []
    calls = []
    cfg = SimpleNamespace(model=SimpleNamespace(freeze_audio_encoder=True))
    recipe = SimpleNamespace(
        speech_cascades_finetune_config=lambda **kwargs: pytest.fail("Unexpected CE recipe"),
        speech_cascades_s2_kd_config=lambda **kwargs: calls.append(kwargs) or cfg,
    )
    monkeypatch.setattr(cli, "_load_recipe", lambda: recipe)
    modules = {
        "megatron.bridge.models.speech_cascades": {"dcp_compat": None},
        "megatron.bridge.models.speech_cascades.obs_logging": {"install_speech_obs_logging": lambda: None},
        "megatron.bridge.models.speech_cascades.peft": {"SpeechCascadesLoRA": None},
        "megatron.bridge.models.speech_cascades.step": {"forward_step": None},
        "megatron.bridge.training.finetune": {"finetune": lambda config, **kwargs: captured.append(config)},
        "megatron.bridge.training.pretrain": {"pretrain": lambda config, **kwargs: captured.append(config)},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finetune.py",
            "--hf-path",
            "/fixture/hf",
            "--streaming-config-yaml",
            "/fixture/data.yaml",
            "--train-config-yaml",
            str(CONFIGS / CATALOG["stages"][stage]["train_config"]),
        ],
    )
    cli.main()
    assert captured == [cfg]
    assert cfg.model.freeze_audio_encoder is (stage != "p3")
    assert calls[0]["mtp_init_extra_layers_from_first"] is False
    assert calls[0]["distillation_weight"] == 1
