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

"""Extract the Qwen3.5 language model from a conditional-generation checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


LANGUAGE_MODEL_PREFIX = "model.language_model."
LM_HEAD_KEY = "lm_head.weight"
MTP_PREFIX = "mtp."
TEXT_ARTIFACT_NAMES = (
	"added_tokens.json",
	"chat_template.jinja",
	"generation_config.json",
	"merges.txt",
	"special_tokens_map.json",
	"tokenizer.json",
	"tokenizer_config.json",
	"vocab.json",
)


def _text_key(source_key: str) -> str | None:
	if source_key.startswith(LANGUAGE_MODEL_PREFIX):
		return f"model.{source_key.removeprefix(LANGUAGE_MODEL_PREFIX)}"
	if source_key == LM_HEAD_KEY or source_key.startswith(MTP_PREFIX):
		return source_key
	return None


def extract_qwen35_text_checkpoint(source_dir: Path, output_dir: Path) -> None:
	"""Extract a text-only Qwen3.5 checkpoint without modifying the source files.

	Args:
		source_dir: Directory containing the source config, index, and safetensor shards.
		output_dir: New or empty directory for the extracted checkpoint.

	Raises:
		ValueError: If the source has no text config or language-model tensors.
		FileExistsError: If the output directory is not empty.
	"""
	config_path = source_dir / "config.json"
	index_path = source_dir / "model.safetensors.index.json"
	with config_path.open(encoding="utf-8") as config_file:
		source_config = json.load(config_file)
	with index_path.open(encoding="utf-8") as index_file:
		source_index = json.load(index_file)

	if "text_config" not in source_config:
		raise ValueError("Source config does not contain text_config")
	if output_dir.exists() and any(output_dir.iterdir()):
		raise FileExistsError(f"Output directory is not empty: {output_dir}")
	output_dir.mkdir(parents=True, exist_ok=True)

	text_config = dict(source_config["text_config"])
	text_config["architectures"] = ["Qwen3_5ForCausalLM"]
	text_config["tie_word_embeddings"] = source_config.get(
		"tie_word_embeddings", text_config.get("tie_word_embeddings", False)
	)
	with (output_dir / "config.json").open("w", encoding="utf-8") as config_file:
		json.dump(text_config, config_file, indent=2)
		config_file.write("\n")

	for artifact_name in TEXT_ARTIFACT_NAMES:
		artifact_path = source_dir / artifact_name
		if artifact_path.is_file():
			shutil.copy2(artifact_path, output_dir / artifact_name)

	tokenizer_config_path = output_dir / "tokenizer_config.json"
	if tokenizer_config_path.is_file():
		with tokenizer_config_path.open(encoding="utf-8") as tokenizer_config_file:
			tokenizer_config = json.load(tokenizer_config_file)
		tokenizer_config["fix_mistral_regex"] = True
		with tokenizer_config_path.open("w", encoding="utf-8") as tokenizer_config_file:
			json.dump(tokenizer_config, tokenizer_config_file, indent=2)
			tokenizer_config_file.write("\n")

	source_weight_map = source_index["weight_map"]
	source_shards = sorted(set(source_weight_map.values()))
	output_weight_map: dict[str, str] = {}
	total_size = 0
	language_tensor_count = 0

	for shard_name in source_shards:
		output_tensors = {}
		with safe_open(source_dir / shard_name, framework="pt", device="cpu") as shard:
			for source_key in shard.keys():
				output_key = _text_key(source_key)
				if output_key is None:
					continue
				tensor = shard.get_tensor(source_key)
				output_tensors[output_key] = tensor
				output_weight_map[output_key] = shard_name
				total_size += tensor.numel() * tensor.element_size()
				if source_key.startswith(LANGUAGE_MODEL_PREFIX):
					language_tensor_count += 1
		if output_tensors:
			save_file(output_tensors, output_dir / shard_name, metadata={"format": "pt"})

	if language_tensor_count == 0:
		raise ValueError("No Qwen3.5 language-model tensors found")

	output_index = {"metadata": {"total_size": total_size}, "weight_map": output_weight_map}
	with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as index_file:
		json.dump(output_index, index_file, indent=2, sort_keys=True)
		index_file.write("\n")


def main() -> None:
	"""Parse command-line arguments and extract the checkpoint."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source-dir", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	args = parser.parse_args()
	extract_qwen35_text_checkpoint(args.source_dir, args.output_dir)


if __name__ == "__main__":
	main()
