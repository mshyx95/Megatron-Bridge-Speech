# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Tests for extracting text weights from a Qwen3.5 multimodal checkpoint."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "examples" / "conversion" / "extract_qwen35_text_checkpoint.py"


@pytest.fixture(scope="module")
def extraction_module():
	"""Load the extraction script as a module."""
	spec = importlib.util.spec_from_file_location("extract_qwen35_text_checkpoint_under_test", _SCRIPT_PATH)
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	try:
		spec.loader.exec_module(module)
		yield module
	finally:
		sys.modules.pop(spec.name, None)


@pytest.mark.unit
def test_extracts_language_lm_head_and_mtp_weights_only(extraction_module, tmp_path: Path) -> None:
	source_dir = tmp_path / "source"
	output_dir = tmp_path / "output"
	source_dir.mkdir()

	config = {
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"model_type": "qwen3_5",
		"tie_word_embeddings": False,
		"text_config": {"model_type": "qwen3_5_text", "tie_word_embeddings": True},
	}
	(source_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
	(source_dir / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
	(source_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
	(source_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")

	shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
	language_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
	lm_head_weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
	mtp_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
	save_file(
		{
			"model.language_model.layers.0.mlp.down_proj.weight": language_weight,
			"model.visual.patch_embed.weight": torch.ones(1),
		},
		source_dir / shard_names[0],
	)
	save_file(
		{"lm_head.weight": lm_head_weight, "mtp.fc.weight": mtp_weight},
		source_dir / shard_names[1],
	)
	source_index = {
		"metadata": {"total_size": 76},
		"weight_map": {
			"model.language_model.layers.0.mlp.down_proj.weight": shard_names[0],
			"model.visual.patch_embed.weight": shard_names[0],
			"lm_head.weight": shard_names[1],
			"mtp.fc.weight": shard_names[1],
		},
	}
	(source_dir / "model.safetensors.index.json").write_text(json.dumps(source_index), encoding="utf-8")

	extraction_module.extract_qwen35_text_checkpoint(source_dir, output_dir)

	output_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
	assert output_config["architectures"] == ["Qwen3_5ForCausalLM"]
	assert output_config["model_type"] == "qwen3_5_text"
	assert output_config["tie_word_embeddings"] is False
	assert (output_dir / "tokenizer.json").read_text(encoding="utf-8") == '{"version": "1.0"}'
	output_tokenizer_config = json.loads((output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
	assert output_tokenizer_config["fix_mistral_regex"] is True
	assert not (output_dir / "preprocessor_config.json").exists()

	output_index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
	assert set(output_index["weight_map"]) == {
		"lm_head.weight",
		"model.layers.0.mlp.down_proj.weight",
		"mtp.fc.weight",
	}
	assert output_index["metadata"]["total_size"] == (
		language_weight.nbytes + lm_head_weight.nbytes + mtp_weight.nbytes
	)

	with safe_open(output_dir / shard_names[0], framework="pt", device="cpu") as shard:
		assert shard.metadata() == {"format": "pt"}
		assert shard.keys() == ["model.layers.0.mlp.down_proj.weight"]
		assert torch.equal(shard.get_tensor("model.layers.0.mlp.down_proj.weight"), language_weight)
	with safe_open(output_dir / shard_names[1], framework="pt", device="cpu") as shard:
		assert shard.metadata() == {"format": "pt"}
		assert shard.keys() == ["lm_head.weight", "mtp.fc.weight"]
		assert torch.equal(shard.get_tensor("lm_head.weight"), lm_head_weight)
		assert torch.equal(shard.get_tensor("mtp.fc.weight"), mtp_weight)
