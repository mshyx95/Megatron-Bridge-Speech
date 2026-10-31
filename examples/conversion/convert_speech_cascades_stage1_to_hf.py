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

"""One-time converter: base Qwen3.5/Qwen3.6 + Cascades 22L encoder -> stage-1 HuggingFace directory.

Builds the combined ``Qwen3_5SpeechCascadesForConditionalGeneration`` HF checkpoint used to cold-start
EN-ASR stage-1 training (LLM frozen, encoder + MLP projector trained). The language model comes from a
base Qwen3.5/Qwen3.6 checkpoint (its ``model.language_model.*`` weights, flattened to ``model.*``); the
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

import numpy as np
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


def _load_encoder(
    encoder_ckpt: Path, *, num_blocks: int = _NUM_BLOCKS,
    mean_file: Path | None = None, invstd_file: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Map standalone encoder weights and optional reference CMVN into the audio prefix."""
    payload = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    out: dict[str, torch.Tensor] = OrderedDict()
    for key, tensor in payload.items():
        if key.startswith("decoder.") or key.startswith("ctc_layer."):
            continue
        if key.startswith("encoder."):
            inner = key[len("encoder.") :]
            if inner.startswith("encoders.") and int(inner.split(".")[1]) >= num_blocks:
                continue
            out[_AUDIO_PREFIX + "encoder." + inner] = tensor.contiguous().clone()
        elif key.startswith("encoder_embedding."):  # CMVN -> nested under the encoder
            out[_AUDIO_PREFIX + "encoder.encoder_embedding." + key[len("encoder_embedding.") :]] = (
                tensor.contiguous().clone()
            )
    if (mean_file is None) != (invstd_file is None):
        raise ValueError("Both mean_file and invstd_file are required together")
    if mean_file is not None:
        for name, path in (("global_mean", mean_file), ("global_invstd", invstd_file)):
            values = torch.from_numpy(np.loadtxt(path))
            if values.shape != (80,) or not torch.isfinite(values).all():
                raise ValueError(f"Invalid CMVN vector {name}: expected 80 finite values")
            if name == "global_invstd" and not (values > 0).all():
                raise ValueError("CMVN inverse standard deviations must be positive")
            out[_AUDIO_PREFIX + "encoder.encoder_embedding." + name] = values
    logger.info("encoder tensors mapped: %d; requested blocks: %d", len(out), num_blocks)
    return out


def _random_projector(
    hidden_size: int, seed: int, *, num_blocks: int = _NUM_BLOCKS,
    encoder_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Random-init the MLP audio projector matching CascadesAudioEmbedding's structure."""
    from megatron.bridge.models.speech_cascades.audio_embedding import CascadesAudioEmbedding

    torch.manual_seed(seed)
    audio = CascadesAudioEmbedding(
        {**_STAGE1_AUDIO_CONFIG, "num_blocks": num_blocks},
        hidden_size=hidden_size, projection_cls="mlp", compression_rate=8, downsample_rate=1
    )
    if encoder_state is not None:
        prefix = _AUDIO_PREFIX + "encoder."
        audio.encoder.load_state_dict({key.removeprefix(prefix): value for key, value in encoder_state.items()}, strict=True)
        logger.info("Encoder validation passed: blocks=%d, all weights and CMVN loaded", num_blocks)
    logger.info("Random MLP parameters: %d", sum(param.numel() for param in audio.audio_projection.parameters()))
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


def _write_config(base_dir: Path, output_dir: Path, *, num_blocks: int = _NUM_BLOCKS) -> int:
    base = json.loads((base_dir / "config.json").read_text())
    text = dict(base["text_config"])
    text["architectures"] = [_ARCHITECTURE]
    text["audio_processor"] = {"name": "cascades", "config": {**_STAGE1_AUDIO_CONFIG, "num_blocks": num_blocks}}
    text["tie_word_embeddings"] = base.get("tie_word_embeddings", text.get("tie_word_embeddings", True))
    text["embd_layer"] = dict(_EMBD_LAYER)
    text.setdefault("torch_dtype", text.get("dtype", "bfloat16"))
    (output_dir / "config.json").write_text(json.dumps(text, indent=2))
    return int(text["hidden_size"])


def main() -> None:
    """Assemble a combined HF checkpoint from the base LLM and a standalone speech encoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-hf", required=True, help="Base Qwen3.5/Qwen3.6 HF directory.")
    parser.add_argument("--encoder-checkpoint", required=True, help="Standalone checkpoint_22L/model.checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Destination combined HF directory.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for the random projector init.")
    parser.add_argument("--encoder-num-blocks", type=int, choices=(23, 24), default=23)
    parser.add_argument("--mean-file", type=Path)
    parser.add_argument("--invstd-file", type=Path)
    args = parser.parse_args()
    if (args.mean_file is None) != (args.invstd_file is None):
        parser.error("--mean-file and --invstd-file must be supplied together")

    base_dir = Path(args.base_hf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_size = _write_config(base_dir, output_dir, num_blocks=args.encoder_num_blocks)
    encoder_state = _load_encoder(
        Path(args.encoder_checkpoint), num_blocks=args.encoder_num_blocks,
        mean_file=args.mean_file, invstd_file=args.invstd_file,
    )
    projector = _random_projector(
        hidden_size, args.seed, num_blocks=args.encoder_num_blocks, encoder_state=encoder_state,
    )
    combined: dict[str, torch.Tensor] = {}
    combined.update(_load_base_llm(base_dir))
    combined.update(encoder_state)
    combined.update(projector)
    logger.info("total combined tensors: %d", len(combined))
    _sharded_save(combined, output_dir)

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
        artifact = base_dir / name
        if artifact.is_file():
            (output_dir / name).write_bytes(artifact.read_bytes())
    logger.info("wrote stage-1 HF checkpoint to %s (architectures=%s)", output_dir, _ARCHITECTURE)


if __name__ == "__main__":
    main()
