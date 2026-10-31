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

import fnmatch
from dataclasses import dataclass
from typing import Iterable, Mapping


_COMMON_LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
_FULL_ATTENTION_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)
_LINEAR_ATTENTION_SUFFIXES = (
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.out_proj.weight",
    "linear_attn.norm.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
)
_MTP_KEY_PATTERNS = (
    "mtp.fc.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    "mtp.layers.*.input_layernorm.weight",
    "mtp.layers.*.post_attention_layernorm.weight",
    "mtp.layers.*.self_attn.q_proj.weight",
    "mtp.layers.*.self_attn.k_proj.weight",
    "mtp.layers.*.self_attn.v_proj.weight",
    "mtp.layers.*.self_attn.o_proj.weight",
    "mtp.layers.*.self_attn.q_norm.weight",
    "mtp.layers.*.self_attn.k_norm.weight",
    "mtp.layers.*.mlp.gate_proj.weight",
    "mtp.layers.*.mlp.up_proj.weight",
    "mtp.layers.*.mlp.down_proj.weight",
)


@dataclass(frozen=True)
class CheckpointInventory:
    """Summary of a Qwen3.5 HF checkpoint's config and tensor mapping coverage."""

    model_type: str
    architecture: str | None
    num_hidden_layers: int
    tensor_count: int
    model_prefix: str
    missing_keys: tuple[str, ...]
    unmapped_keys: tuple[str, ...]
    attention_bias_keys: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (self.missing_keys or self.unmapped_keys or self.attention_bias_keys)


def expected_qwen35_keys(config: Mapping[str, object], model_prefix: str) -> set[str]:
    """Build the exact dense Qwen3.5 key set covered by the current Bridge mappings."""
    num_hidden_layers = int(config["num_hidden_layers"])
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != num_hidden_layers:
        raise ValueError("layer_types must contain one entry per hidden layer")

    expected = {
        f"{model_prefix}embed_tokens.weight",
        f"{model_prefix}norm.weight",
    }
    if not bool(config.get("tie_word_embeddings", False)):
        expected.add("lm_head.weight")

    for layer_index, layer_type in enumerate(layer_types):
        layer_prefix = f"{model_prefix}layers.{layer_index}."
        expected.update(layer_prefix + suffix for suffix in _COMMON_LAYER_SUFFIXES)
        if layer_type == "full_attention":
            expected.update(layer_prefix + suffix for suffix in _FULL_ATTENTION_SUFFIXES)
        elif layer_type == "linear_attention":
            expected.update(layer_prefix + suffix for suffix in _LINEAR_ATTENTION_SUFFIXES)
        else:
            raise ValueError(f"Unsupported layer type at index {layer_index}: {layer_type!r}")

    return expected


def inventory_qwen35_checkpoint(
    config: Mapping[str, object], tensor_keys: Iterable[str]
) -> CheckpointInventory:
    """Compare Qwen3.5 config and tensor keys with the current dense Bridge mappings."""
    keys = set(tensor_keys)
    if "model.embed_tokens.weight" in keys:
        model_prefix = "model."
    elif "model.language_model.embed_tokens.weight" in keys:
        model_prefix = "model.language_model."
    else:
        raise ValueError("Could not identify the Qwen3.5 language-model key prefix")

    expected = expected_qwen35_keys(config, model_prefix)
    known = set(expected)
    if bool(config.get("tie_word_embeddings", False)) and "lm_head.weight" in keys:
        known.add("lm_head.weight")
    known.update(key for key in keys if any(fnmatch.fnmatch(key, pattern) for pattern in _MTP_KEY_PATTERNS))

    attention_bias_keys = tuple(sorted(key for key in keys if ".self_attn." in key and key.endswith(".bias")))
    architectures = config.get("architectures")
    architecture = architectures[0] if isinstance(architectures, list) and architectures else None
    return CheckpointInventory(
        model_type=str(config.get("model_type", "unknown")),
        architecture=architecture,
        num_hidden_layers=int(config["num_hidden_layers"]),
        tensor_count=len(keys),
        model_prefix=model_prefix,
        missing_keys=tuple(sorted(expected - keys)),
        unmapped_keys=tuple(sorted(keys - known)),
        attention_bias_keys=attention_bias_keys,
    )