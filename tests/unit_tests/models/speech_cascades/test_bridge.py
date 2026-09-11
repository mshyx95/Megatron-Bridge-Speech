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
from types import SimpleNamespace
from pathlib import Path

import torch

from megatron.bridge.models.speech_cascades.bridge import SpeechCascadesBridge


_CONVERTER = Path(__file__).resolve().parents[4] / "examples" / "conversion" / "convert_speech_cascades_stage1_to_hf.py"


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


def test_qwen36_moe_provider_settings() -> None:
    bridge = SpeechCascadesBridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=_moe_config()))

    assert provider.params_dtype == torch.bfloat16
    assert provider.ffn_hidden_size == 512
    assert provider.num_moe_experts == 256
    assert provider.moe_router_topk == 8
    assert provider.moe_aux_loss_coeff == 0.001
    assert provider.moe_ffn_hidden_size == 512
    assert provider.moe_shared_expert_intermediate_size == 512
    assert provider.moe_grouped_gemm is True
    assert provider.moe_token_dispatcher_type == "alltoall"


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