import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.unit
@pytest.mark.parametrize("missing_audio", [False, True])
def test_load_lora_weights_includes_audio_and_gdn(monkeypatch, missing_audio):
    path = Path(__file__).resolve().parents[4] / "examples/models/speech_cascades/lora_eval.py"
    spec = importlib.util.spec_from_file_location("lora_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights = {
        "language_model.layer.adapter.linear_in.weight": object(),
        "audio_model.encoder.weight": object(),
        "language_model.layer.A_log": object(),
    }
    loaded = dict(weights)
    if missing_audio:
        loaded.pop("audio_model.encoder.weight")
    checkpointing = ModuleType("megatron.bridge.training.checkpointing")
    checkpointing._generate_model_state_dict = Mock(return_value={"model": weights})
    checkpointing.apply_peft_adapter_filter_to_state_dict = Mock(side_effect=lambda state, peft: state)
    core = ModuleType("megatron.core")
    core.dist_checkpointing = SimpleNamespace(
        load_common_state_dict=Mock(return_value={}),
        load_content_metadata=Mock(return_value={}),
        load=Mock(return_value={"model": loaded}),
    )
    utils = ModuleType("megatron.core.utils")
    utils.unwrap_model = lambda model: model
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.checkpointing", checkpointing)
    chunk = Mock()
    chunk.load_state_dict.return_value = SimpleNamespace(missing_keys=["frozen.weight"], unexpected_keys=[])
    peft = Mock()
    if missing_audio:
        with pytest.raises(ValueError, match="audio_model.encoder.weight"):
            module.load_lora_weights([chunk], peft, "/checkpoint")
        chunk.load_state_dict.assert_not_called()
    else:
        module.load_lora_weights([chunk], peft, "/checkpoint")
        peft.set_params_to_save.assert_called_once_with([chunk])
        chunk.load_state_dict.assert_called_once_with(weights, strict=False)
        chunk.eval.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("mode, expected_alpha", [("legacy", 640), ("normalized", 80)])
def test_expert_scaling_preserves_training_semantics(monkeypatch, mode, expected_alpha):
    path = Path(__file__).resolve().parents[4] / "examples/models/speech_cascades/lora_eval.py"
    spec = importlib.util.spec_from_file_location("lora_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    utils = ModuleType("megatron.bridge.peft.utils")
    utils.is_expert_linear = lambda name: name == "routed"
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft.utils", utils)
    routed = SimpleNamespace(adapter=SimpleNamespace(dim=40, alpha=80))
    shared = SimpleNamespace(adapter=SimpleNamespace(dim=320, alpha=640))
    chunk = Mock()
    chunk.named_modules.return_value = [("routed", routed), ("shared", shared)]
    module.configure_expert_scaling([chunk], SimpleNamespace(normalize_moe_lora=True, alpha=640), mode)
    assert routed.adapter.alpha == expected_alpha
    assert shared.adapter.alpha == 640


@pytest.mark.unit
@pytest.mark.parametrize("base_checkpoint", [None, "/base/iter_0012000"])
def test_build_loads_base_before_adapters(monkeypatch, base_checkpoint):
    path = Path(__file__).resolve().parents[4] / "examples/models/speech_cascades/lora_eval.py"
    spec = importlib.util.spec_from_file_location("lora_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []

    @dataclass
    class FakePeft:
        dim: int

        def __call__(self, chunks, training):
            assert training is True
            events.append("inject")

    config = {
        "peft": {
            "_target_": "megatron.bridge.models.speech_cascades.peft.SpeechCascadesLoRA",
            "dim": 320,
        },
        "checkpoint": {"pretrained_checkpoint": "/base" if base_checkpoint else "/hf"},
        "model": {"moe_grouped_gemm": True, "moe_token_dispatcher_type": "alltoall"},
    }
    provider = Mock()
    chunk = Mock()
    provider.register_pre_wrap_hook.side_effect = lambda hook: hooks.append(hook)
    hooks = []

    def provide(**kwargs):
        assert kwargs == {"wrap_with_ddp": False, "bf16": True}
        if base_checkpoint is None:
            events.append("base")
        for hook in hooks:
            hook([chunk])
        return [chunk]

    provider.provide_distributed_model.side_effect = provide
    bridge = Mock()
    bridge.to_megatron_provider.return_value = provider
    modules = {
        "torch": {"bfloat16": "bf16", "cuda": SimpleNamespace(empty_cache=lambda: None)},
        "megatron.core.transformer.enums": {"AttnBackend": SimpleNamespace(flash="flash")},
        "megatron.core.utils": {"unwrap_model": lambda chunks: chunks},
        "megatron.bridge": {"AutoBridge": SimpleNamespace(from_hf_pretrained=lambda path: bridge)},
        "megatron.bridge.models.speech_cascades": {"dcp_compat": None},
        "megatron.bridge.models.speech_cascades.peft": {"SpeechCascadesLoRA": FakePeft},
        "megatron.bridge.training.checkpointing": {
            "_load_model_weights_from_checkpoint": lambda *args, **kwargs: events.append("base")
        },
        "megatron.bridge.training.utils.checkpoint_utils": {"read_run_config": lambda path: config},
    }
    for name, attributes in modules.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(module, "configure_expert_scaling", lambda *args: events.append("scaling"))
    monkeypatch.setattr(module, "load_lora_weights", lambda *args: events.append("weights"))
    result = module.build_lora_model(
        "/hf", "/adapter", base_checkpoint=base_checkpoint, expert_scaling="normalized", tensor_parallel_size=2
    )
    assert events == ["base", "inject", "scaling", "weights"]
    assert result is chunk
    assert provider.parallel_output is False
    assert provider.tensor_model_parallel_size == 2
    assert provider.expert_tensor_parallel_size == 2
    assert provider.mtp_num_hidden_layers == 0
    bridge.to_megatron_provider.assert_called_once_with(load_weights=base_checkpoint is None)
    chunk.requires_grad_.assert_called_once_with(False)