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

"""One-time converter: base Qwen3.5-4B + Cascades 22L encoder -> stage-1 HuggingFace directory.

Builds the combined ``Qwen3_5SpeechCascadesForConditionalGeneration`` HF checkpoint used to cold-start
EN-ASR stage-1 training (LLM frozen, encoder + MLP projector trained). The language model comes from a
base Qwen3.5-4B checkpoint (its ``model.language_model.*`` weights, flattened to ``model.*``); the
audio encoder + CMVN come from the standalone ``checkpoint_22L`` (dropping its extra conformer block,
decoder, and CTC head); the MLP audio projector is randomly initialized. The written ``config.json``
flattens ``text_config`` and adds the stage-1 ``audio_processor`` (nemo_conv 23L, non-flash t5 bias)
and ``embd_layer`` so ``AutoBridge.from_hf_pretrained`` dispatches to ``SpeechCascadesBridge``.
"""

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_ARCHITECTURE = "Qwen3_5SpeechCascadesForConditionalGeneration"
_AUDIO_PREFIX = "model.embed_tokens_extend."
_SHARD_SIZE_BYTES = 4 * 1024**3
_NUM_BLOCKS = 23

# Stage-1 EN-ASR encoder (nemo_conv 23L). Non-flash t5 bias so the pipeline runs without the
# custom flash-attention-t5 build; activation checkpointing off so parameter names match the mapping.
_STAGE1_AUDIO_CONFIG = {
    "input_size": 80,
    "attention_dim": 1024,
    "attention_heads": 16,
    "linear_units": 1536,
    "dropout_rate": 0.0,
    "num_blocks": _NUM_BLOCKS,
    "kernel_size": 3,
    "causal": True,
    "batch_norm": False,
    "activation": "swish",
    "conv_activation": "swish",
    "conv_glu_type": "swish",
    "cnn_layer_norm": True,
    "depthwise_seperable_out_channel": 1024,
    "depthwise_multiplier": 1,
    "ext_pw_out_channel": 1024,
    "ext_pw_kernel_size": 1,
    "chunk_size": -1,
    "left_chunk": 18,
    "bias_in_glu": True,
    "input_layer": "nemo_conv",
    "time_reduction": 8,
    "nemo_conv_settings": {"conv_channels": 1024},
    "relative_attention_bias_args": {"type": "t5", "t5_bias_max_distance": 500},
    "encoder_embedding_config": {"input_size": 80},
    "use_flashattention_t5bias": False,
    "activation_checkpointing": "",
}
_EMBD_LAYER = {"projection_cls": "mlp", "compression_rate": 8, "use_qformer": False}


def _load_base_llm(base_dir: Path) -> dict[str, torch.Tensor]:
    """Load the base checkpoint and keep only the language model, flattened to ``model.*``."""
    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    shard_files = sorted(set(index["weight_map"].values()))
    llm: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        for key, tensor in load_file(str(base_dir / shard)).items():
            if key.startswith("model.language_model."):
                llm["model." + key[len("model.language_model.") :]] = tensor.contiguous().clone()
            elif key == "lm_head.weight":
                llm[key] = tensor.contiguous().clone()
            # drop model.visual.*, mtp.*, and any other non-LLM tensors
    if not any(k.startswith("model.layers.") for k in llm):
        raise ValueError(f"No language-model layers found under {base_dir}")
    logger.info("base LLM tensors kept: %d", len(llm))
    return llm


def _load_encoder(encoder_ckpt: Path) -> dict[str, torch.Tensor]:
    """Map the standalone 22L encoder checkpoint into the audio-encoder prefix."""
    payload = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    out: dict[str, torch.Tensor] = OrderedDict()
    for key, tensor in payload.items():
        if key.startswith("decoder.") or key.startswith("ctc_layer."):
            continue
        if key.startswith("encoder."):
            inner = key[len("encoder.") :]
            if inner.startswith(f"encoders.{_NUM_BLOCKS}."):  # extra conformer block beyond num_blocks
                continue
            out[_AUDIO_PREFIX + "encoder." + inner] = tensor.contiguous().clone()
        elif key.startswith("encoder_embedding."):  # CMVN -> nested under the encoder
            out[_AUDIO_PREFIX + "encoder.encoder_embedding." + key[len("encoder_embedding.") :]] = (
                tensor.contiguous().clone()
            )
    logger.info("encoder tensors mapped: %d", len(out))
    return out


def _random_projector(hidden_size: int, seed: int) -> dict[str, torch.Tensor]:
    """Random-init the MLP audio projector matching CascadesAudioEmbedding's structure."""
    from megatron.bridge.models.speech_cascades.audio_embedding import CascadesAudioEmbedding

    torch.manual_seed(seed)
    audio = CascadesAudioEmbedding(
        _STAGE1_AUDIO_CONFIG, hidden_size=hidden_size, projection_cls="mlp", compression_rate=8, downsample_rate=1
    )
    return {
        _AUDIO_PREFIX + "audio_projection." + key: value.contiguous().clone()
        for key, value in audio.audio_projection.state_dict().items()
    }


def _sharded_save(state_dict: dict[str, torch.Tensor], output_dir: Path) -> None:
    shards: list[dict[str, torch.Tensor]] = [{}]
    sizes = [0]
    weight_map: dict[str, str] = {}
    for key, tensor in state_dict.items():
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
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(sizes)}, "weight_map": weight_map}, indent=2)
    )


def _write_config(base_dir: Path, output_dir: Path) -> int:
    base = json.loads((base_dir / "config.json").read_text())
    text = dict(base["text_config"])
    text["architectures"] = [_ARCHITECTURE]
    text["model_type"] = "qwen3_5_text"
    text["audio_processor"] = {"name": "cascades", "config": dict(_STAGE1_AUDIO_CONFIG)}
    text["embd_layer"] = dict(_EMBD_LAYER)
    text.setdefault("torch_dtype", text.get("dtype", "bfloat16"))
    (output_dir / "config.json").write_text(json.dumps(text, indent=2))
    return int(text["hidden_size"])


def main() -> None:
    """Assemble the stage-1 combined HF checkpoint from the base LLM and the 22L encoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-hf", required=True, help="Base Qwen3.5-4B HF directory.")
    parser.add_argument("--encoder-checkpoint", required=True, help="Standalone checkpoint_22L/model.checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Destination combined HF directory.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for the random projector init.")
    args = parser.parse_args()

    base_dir = Path(args.base_hf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_size = _write_config(base_dir, output_dir)
    combined: dict[str, torch.Tensor] = {}
    combined.update(_load_base_llm(base_dir))
    combined.update(_load_encoder(Path(args.encoder_checkpoint)))
    combined.update(_random_projector(hidden_size, args.seed))
    logger.info("total combined tensors: %d", len(combined))
    _sharded_save(combined, output_dir)

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
        artifact = base_dir / name
        if artifact.is_file():
            (output_dir / name).write_bytes(artifact.read_bytes())
    logger.info("wrote stage-1 HF checkpoint to %s (architectures=%s)", output_dir, _ARCHITECTURE)


if __name__ == "__main__":
    main()
