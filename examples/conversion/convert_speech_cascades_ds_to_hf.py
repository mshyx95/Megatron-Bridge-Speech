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

"""One-time converter: legacy DeepSpeed speech checkpoint -> HuggingFace-format directory.

The consolidated DeepSpeed ``module`` payload already uses HuggingFace-style keys (``model.layers.*``
for the Qwen3.5 GDN language model and ``model.embed_tokens_extend.*`` for the Cascades audio encoder
and projection). This script re-saves those tensors as safetensors and writes a ``config.json`` whose
``architectures`` entry dispatches to ``SpeechCascadesBridge``, so the native
``AutoBridge.from_hf_pretrained`` path can import the weights. It is a one-time import step; training
and inference then use the native Megatron representation.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_ARCHITECTURE = "Qwen3_5SpeechCascadesForConditionalGeneration"
_SHARD_SIZE_BYTES = 4 * 1024**3


def _iter_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(payload, dict) or "module" not in payload:
        raise ValueError(f"Expected a DeepSpeed payload with a 'module' entry in {checkpoint_path}")
    return payload["module"]


def _sharded_save(state_dict: dict[str, torch.Tensor], output_dir: Path) -> None:
    """Save tensors into ~4GB safetensors shards with a HuggingFace weight-map index."""
    shards: list[dict[str, torch.Tensor]] = [{}]
    sizes = [0]
    weight_map: dict[str, str] = {}
    for key, tensor in state_dict.items():
        tensor = tensor.contiguous().clone()  # detach from mmap and any shared (tied) storage
        nbytes = tensor.numel() * tensor.element_size()
        if sizes[-1] > 0 and sizes[-1] + nbytes > _SHARD_SIZE_BYTES:
            shards.append({})
            sizes.append(0)
        shards[-1][key] = tensor
        sizes[-1] += nbytes

    total = len(shards)
    for index, shard in enumerate(shards, start=1):
        shard_name = f"model-{index:05d}-of-{total:05d}.safetensors"
        save_file(shard, str(output_dir / shard_name), metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = shard_name
        logger.info("wrote %s (%d tensors)", shard_name, len(shard))

    total_size = sum(sizes)
    index_path = output_dir / "model.safetensors.index.json"
    with index_path.open("w", encoding="utf-8") as index_file:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, index_file, indent=2)


def _write_config(source_dir: Path, output_dir: Path) -> None:
    with (source_dir / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["architectures"] = [_ARCHITECTURE]
    config.setdefault("torch_dtype", config.get("dtype", "bfloat16"))
    with (output_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)


def _copy_tokenizer_artifacts(source_dir: Path, output_dir: Path) -> None:
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
        artifact = source_dir / name
        if artifact.is_file():
            shutil.copy2(artifact, output_dir / name)


def main() -> None:
    """Parse arguments and convert the DeepSpeed checkpoint to a HuggingFace-format directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to mp_rank_00_model_states.pt")
    parser.add_argument("--source-dir", required=True, help="Directory holding the source config.json")
    parser.add_argument("--output-dir", required=True, help="Destination HuggingFace-format directory")
    parser.add_argument("--tokenizer-dir", default=None, help="Optional directory to copy tokenizer artifacts from")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = _iter_state_dict(Path(args.checkpoint))
    logger.info("loaded %d tensors from %s", len(state_dict), args.checkpoint)
    _sharded_save(state_dict, output_dir)
    _write_config(Path(args.source_dir), output_dir)
    _copy_tokenizer_artifacts(Path(args.tokenizer_dir or args.source_dir), output_dir)
    logger.info("wrote HuggingFace-format checkpoint to %s (architectures=%s)", output_dir, _ARCHITECTURE)


if __name__ == "__main__":
    main()
