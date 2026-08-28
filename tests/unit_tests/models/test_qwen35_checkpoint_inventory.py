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

from megatron.bridge.models.qwen.checkpoint_inventory import inventory_qwen35_checkpoint


def _config() -> dict[str, object]:
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "layer_types": ["linear_attention", "full_attention"],
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 2,
        "tie_word_embeddings": False,
    }


def _complete_keys() -> set[str]:
    common_layer_suffixes = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    keys.update(f"model.layers.0.{suffix}" for suffix in common_layer_suffixes)
    keys.update(f"model.layers.1.{suffix}" for suffix in common_layer_suffixes)
    keys.update(
        {
            "model.layers.0.linear_attn.A_log",
            "model.layers.0.linear_attn.conv1d.weight",
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.0.linear_attn.in_proj_a.weight",
            "model.layers.0.linear_attn.in_proj_b.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.layers.0.linear_attn.in_proj_z.weight",
            "model.layers.0.linear_attn.norm.weight",
            "model.layers.0.linear_attn.out_proj.weight",
            "model.layers.1.self_attn.k_norm.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.self_attn.q_norm.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
        }
    )
    return keys


def test_complete_dense_qwen35_checkpoint_inventory() -> None:
    inventory = inventory_qwen35_checkpoint(_config(), _complete_keys())

    assert inventory.valid
    assert inventory.model_prefix == "model."
    assert inventory.tensor_count == 28
    assert inventory.missing_keys == ()
    assert inventory.unmapped_keys == ()


def test_inventory_reports_missing_projection_and_attention_bias() -> None:
    keys = _complete_keys()
    keys.remove("model.layers.0.linear_attn.in_proj_a.weight")
    keys.add("model.layers.1.self_attn.q_proj.bias")

    inventory = inventory_qwen35_checkpoint(_config(), keys)

    assert not inventory.valid
    assert inventory.missing_keys == ("model.layers.0.linear_attn.in_proj_a.weight",)
    assert inventory.unmapped_keys == ("model.layers.1.self_attn.q_proj.bias",)
    assert inventory.attention_bias_keys == ("model.layers.1.self_attn.q_proj.bias",)