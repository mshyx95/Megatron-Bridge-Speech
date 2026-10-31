# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.speech_cascades.bridge import SpeechCascadesBridge


_CONVERTER = (
    Path(__file__).resolve().parents[4] / "examples" / "conversion" / "convert_speech_cascades_stage1_to_hf.py"
)


def _load_converter():
    spec = importlib.util.spec_from_file_location("_speech_cascades_converter", _CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _moe_config() -> SimpleNamespace:
    return SimpleNamespace(
        audio_processor={"config": {}},
        embd_layer={"projection_cls": "mlp", "compression_rate": 8},
        num_hidden_layers=40,
        hidden_size=2048,
        moe_intermediate_size=512,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        vocab_size=248320,
        max_position_embeddings=262144,
        tie_word_embeddings=False,
        rope_parameters={"rope_theta": 10000000, "partial_rotary_factor": 0.25},
        dtype="bfloat16",
        attention_bias=False,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        num_experts=256,
        num_experts_per_tok=8,
        router_aux_loss_coef=0.001,
        shared_expert_intermediate_size=512,
    )


@pytest.mark.unit
@pytest.mark.parametrize("aux_loss_coeff", [0.0, 0.001, 0.25])
def test_qwen36_moe_provider_settings(aux_loss_coeff: float) -> None:
    bridge = SpeechCascadesBridge()
    config = _moe_config()
    config.router_aux_loss_coef = aux_loss_coeff
    provider = bridge.provider_bridge(SimpleNamespace(config=config))

    assert provider.params_dtype == torch.bfloat16
    assert provider.ffn_hidden_size == 512
    assert provider.num_moe_experts == 256
    assert provider.moe_router_topk == 8
    assert provider.moe_aux_loss_coeff == aux_loss_coeff
    assert provider.moe_ffn_hidden_size == 512
    assert provider.moe_shared_expert_intermediate_size == 512
    assert provider.moe_grouped_gemm is True
    assert provider.moe_token_dispatcher_type == "alltoall"


@pytest.mark.unit
@pytest.mark.parametrize("num_experts", [0, 256])
def test_qwen35_common_provider_settings(num_experts: int) -> None:
    config = _moe_config()
    config.num_experts = num_experts
    config.attention_bias = True
    config.linear_conv_kernel_dim = 8
    config.linear_num_key_heads = 8
    config.linear_num_value_heads = 16
    config.rope_parameters["partial_rotary_factor"] = 0.5

    provider = SpeechCascadesBridge().provider_bridge(SimpleNamespace(config=config))

    assert provider.normalization == "RMSNorm"
    assert provider.qk_layernorm is True
    assert provider.add_qkv_bias is True
    assert provider.experimental_attention_variant == "gdn"
    assert provider.linear_attention_freq == [1, 1, 1, 0] * 10
    assert provider.linear_conv_kernel_dim == 8
    assert provider.linear_num_key_heads == 8
    assert provider.linear_num_value_heads == 16
    assert provider.rotary_percent == 0.5
    assert provider.audio_compression_rate == 8
    assert provider.audio_encoder_config["activation_checkpointing"] == ""
    assert provider.params_dtype == torch.bfloat16
    assert provider.position_embedding_type == "rope"


def test_qwen36_packed_expert_mapping() -> None:
    source = SimpleNamespace(
        get_all_keys=lambda: [
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
        ]
    )
    bridge = SpeechCascadesBridge()
    bridge.hf_pretrained = SimpleNamespace(config=_moe_config(), state=SimpleNamespace(source=source))

    registry = bridge.mapping_registry()

    mapping = registry.hf_to_megatron_lookup("model.layers.0.mlp.experts.gate_up_proj")
    assert mapping.megatron_param == "language_model.decoder.layers.0.mlp.experts.linear_fc1.weight*"


def test_converter_preserves_moe_model_type(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    output_dir = tmp_path / "output"
    base_dir.mkdir()
    output_dir.mkdir()
    (base_dir / "config.json").write_text(
        json.dumps({"text_config": {"hidden_size": 2048, "model_type": "qwen3_5_moe_text", "dtype": "bfloat16"}})
    )

    hidden_size = _load_converter()._write_config(base_dir, output_dir)
    converted = json.loads((output_dir / "config.json").read_text())

    assert hidden_size == 2048
    assert converted["model_type"] == "qwen3_5_moe_text"
    assert converted["architectures"] == ["Qwen3_5SpeechCascadesForConditionalGeneration"]


def _load_exporter():
    path = _CONVERTER.parents[1] / "models" / "speech_cascades" / "export_hf.py"
    spec = importlib.util.spec_from_file_location("_speech_cascades_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_exporter_loads_trained_checkpoint(tmp_path, monkeypatch):
    from megatron.bridge import AutoBridge

    checkpoint = tmp_path / "iter_0004000"
    checkpoint.mkdir()
    for filename in (".metadata", "run_config.yaml"):
        (checkpoint / filename).touch()
    output = tmp_path / "export"
    bridge = Mock()
    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", Mock(return_value=bridge))
    monkeypatch.setattr(
        "sys.argv", ["export_hf", "--hf-dir", "initial", "--checkpoint", str(checkpoint), "--output", str(output)]
    )

    _load_exporter().main()

    bridge.export_ckpt.assert_called_once_with(str(checkpoint), str(output), strict=True, source_path="initial")
    bridge.to_megatron_model.assert_not_called()


@pytest.mark.parametrize("case", ["missing_metadata", "missing_config", "nonempty_output", "initial_vllm"])
def test_exporter_rejects_unsafe_inputs(tmp_path, monkeypatch, case):
    checkpoint = tmp_path / "iter_0004000"
    checkpoint.mkdir()
    for filename in (".metadata", "run_config.yaml"):
        (checkpoint / filename).touch()
    output = tmp_path / "export"
    argv = ["export_hf", "--hf-dir", "initial", "--output", str(output)]
    if case == "initial_vllm":
        argv.append("--vllm")
    else:
        argv.extend(["--checkpoint", str(checkpoint)])
        if case == "missing_metadata":
            (checkpoint / ".metadata").unlink()
        elif case == "missing_config":
            (checkpoint / "run_config.yaml").unlink()
        else:
            output.mkdir()
            (output / "existing.safetensors").touch()
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as error:
        _load_exporter().main()
    assert error.value.code == 2


@pytest.mark.unit
@pytest.mark.parametrize("retain_mtp", [False, True])
@pytest.mark.parametrize("model_type", ["qwen3_5_text", "qwen3_5_moe_text"])
def test_vllm_metadata_preserves_audio_and_weights(tmp_path, retain_mtp, model_type):
    base = tmp_path / "base"
    output = tmp_path / "export"
    base.mkdir()
    output.mkdir()
    (base / "config.json").write_text(
        json.dumps({"text_config": {"hidden_size": 2560, "model_type": model_type, "mtp_num_hidden_layers": 1}})
    )
    _load_converter()._write_config(base, output, num_blocks=24)
    original = json.loads((output / "config.json").read_text())
    weight_map = {"model.embed_tokens_extend.audio_projection.0.weight": "model.safetensors"}
    if retain_mtp:
        weight_map["mtp.layers.0.mlp.down_proj.weight"] = "model.safetensors"
    index = json.dumps({"weight_map": weight_map})
    (output / "model.safetensors.index.json").write_text(index)

    _load_exporter().prepare_vllm_config(output)

    converted = json.loads((output / "config.json").read_text())
    assert converted["architectures"] == ["Qwen3_5AudioForCausalLM"]
    assert converted["embd_layer"] == {**original["embd_layer"], "embedding_cls": "audio"}
    assert converted["audio_processor"] == original["audio_processor"]
    assert converted["n_embd"] == 2560
    assert converted["mtp_num_hidden_layers"] == int(retain_mtp)
    assert (output / "model.safetensors.index.json").read_text() == index


@pytest.mark.unit
def test_vllm_metadata_rejects_unsupported_backbone(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "unsupported"}))
    with pytest.raises(ValueError, match="qwen3_5_text or qwen3_5_moe_text"):
        _load_exporter().prepare_vllm_config(tmp_path)
