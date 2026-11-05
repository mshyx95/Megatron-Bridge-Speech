# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export projector-only Qwen4Exp speech training without materializing the frozen LM."""

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import yaml
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


logger = logging.getLogger(__name__)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_audio_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Read only encoder/projector tensors from a trusted native DCP checkpoint on CPU."""
    reader = dcp.FileSystemReader(checkpoint)
    metadata = reader.read_metadata()
    state = {
        name: torch.empty(tuple(tensor.size), dtype=tensor.properties.dtype)
        for name, tensor in metadata.state_dict_metadata.items()
        if name.startswith("audio_model.") and isinstance(tensor, TensorStorageMetadata)
    }
    if not any(name.startswith("audio_model.encoder.") for name in state) or not any(
        name.startswith("audio_model.audio_projection.") for name in state
    ):
        raise ValueError("Checkpoint must contain both the trained projector and the encoder")
    dcp.load(state, storage_reader=reader, no_dist=True)
    return {name.replace("audio_model.", "model.embed_tokens_extend.", 1): tensor for name, tensor in state.items()}


def export_checkpoint(
    *,
    base: Path,
    checkpoint: Path,
    data_config: Path,
    output: Path,
    base_config_sha256: str,
    base_index_sha256: str,
    audio_file: Path | None = None,
    audio_receipt: Path | None = None,
) -> None:
    """Combine checkpoint audio tensors with the verified immutable, frozen language shards.

    This path is valid only for projector-only training with a frozen encoder and
    language model. It does not convert an updated language backbone.
    """
    if output.exists():
        raise FileExistsError(f"Export requires a fresh output directory: {output}")
    for name, expected in (("config.json", base_config_sha256), ("model.safetensors.index.json", base_index_sha256)):
        if _digest(base / name) != expected:
            raise ValueError(f"Frozen backbone source hash mismatch: {name}")
    run = yaml.safe_load((checkpoint / "run_config.yaml").read_text())
    model = run["model"]
    data = yaml.safe_load(data_config.read_text())
    if type(data["streaming_compression"]) is not bool:
        raise ValueError("streaming_compression must be explicitly specified as a boolean")
    if data["compression_rate"] != model["audio_compression_rate"]:
        raise ValueError("Data and model audio compression rates differ")
    if model["_target_"] != "megatron.bridge.models.qwen4_exp.speech.Qwen4ExpSpeechProvider":
        raise ValueError("Expected a Qwen4Exp speech training checkpoint")
    if not model["freeze_language_model"] or not model["freeze_audio_encoder"] or model["freeze_audio_model"]:
        raise ValueError("Frozen-backbone export requires projector-only training")
    if model["mtp_num_layers"] or model["mtp_num_hidden_layers"] or model["mtp_enabled"]:
        raise ValueError("This export supports non-MTP training only")
    config = json.loads((base / "config.json").read_text())
    source_text = dict(config["text_config"])
    saved_text = model["hf_text_config"]
    # HF normalizes full_attention to QSA and adds serialization-only defaults.
    if "layer_types" in source_text:
        source_text["layer_types"] = [
            "qwen_sparse_attention" if layer == "full_attention" else layer for layer in source_text["layer_types"]
        ]
    if config["model_type"] != "qwen4_exp" or any(saved_text.get(key) != value for key, value in source_text.items()):
        raise ValueError("Training text configuration differs from the frozen source")
    source = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {
        name: filename
        for name, filename in source.items()
        if (name.startswith("model.language_model.") or name == "lm_head.weight") and ".mtp." not in name
    }
    if "lm_head.weight" not in weights or not any(name.startswith("model.language_model.layers.") for name in weights):
        raise ValueError("Missing frozen language backbone weights")
    for filename in set(weights.values()):
        if Path(filename).name != filename or not (base / filename).is_file():
            raise ValueError(f"Missing or invalid backbone shard: {filename}")
    encoder_config = dict(model["audio_encoder_config"])
    if encoder_config.pop("use_flashattention_t5bias", False):
        raise ValueError("The serving encoder does not support FlashAttention T5 bias")
    if audio_file is None and audio_receipt is None:
        audio = load_audio_state(checkpoint)
    elif audio_file is not None and audio_receipt is not None:
        receipt = json.loads(audio_receipt.read_text())
        if receipt["metadata_sha256"] != _digest(checkpoint / ".metadata") or receipt["audio_sha256"] != _digest(
            audio_file
        ):
            raise ValueError("Staged audio does not match the checkpoint extraction receipt")
        audio = load_file(audio_file)
        metadata = dcp.FileSystemReader(checkpoint).read_metadata()
        expected_tensors = {
            name.replace("audio_model.", "model.embed_tokens_extend.", 1): tensor
            for name, tensor in metadata.state_dict_metadata.items()
            if name.startswith("audio_model.") and isinstance(tensor, TensorStorageMetadata)
        }
        if (
            not any(name.startswith("model.embed_tokens_extend.encoder.") for name in expected_tensors)
            or not any(name.startswith("model.embed_tokens_extend.audio_projection.") for name in expected_tensors)
            or set(audio) != set(expected_tensors)
            or any(
                tensor.shape != expected_tensors[name].size or tensor.dtype != expected_tensors[name].properties.dtype
                for name, tensor in audio.items()
            )
        ):
            raise ValueError("Staged audio keys, shapes or dtypes differ from the checkpoint")
    else:
        raise ValueError("Staged audio requires both its safetensors file and extraction receipt")
    output.mkdir(parents=True)
    for filename in sorted(set(weights.values())):
        (output / filename).symlink_to((base / filename).resolve(strict=True))
    save_file(audio, str(output / "speech.safetensors"))
    weights.update({name: "speech.safetensors" for name in audio})
    config["architectures"] = ["Qwen4ExpAudioForCausalLM"]
    config["audio_processor"] = {"name": "cascades", "config": encoder_config}
    config["embd_layer"] = {
        "embedding_cls": "audio",
        "projection_cls": model["audio_projection_cls"],
        "compression_rate": model["audio_compression_rate"],
        "downsample_rate": model["audio_downsample_rate"],
    }
    config["hidden_size"] = config["n_embd"] = config["text_config"]["hidden_size"]
    config["vocab_size"] = config["text_config"]["vocab_size"]
    config["speech_streaming_compression"] = data["streaming_compression"]
    config["language_model_only"] = False
    for target in (config, config["text_config"]):
        for field in ("mtp_num_hidden_layers", "num_nextn_predict_layers", "mtp_num_layers"):
            target[field] = 0
    config["text_config"]["mtp"] = {**config["text_config"].get("mtp", {}), "num_hidden_layers": 0}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}, indent=2) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
        if (base / name).is_file():
            shutil.copyfile(base / name, output / name)
    receipt = {
        "checkpoint": str(checkpoint),
        "checkpoint_metadata_sha256": _digest(checkpoint / ".metadata"),
        "run_config_sha256": _digest(checkpoint / "run_config.yaml"),
        "data_config_sha256": _digest(data_config),
        "base_config_sha256": base_config_sha256,
        "base_index_sha256": base_index_sha256,
        "audio_sha256": _digest(output / "speech.safetensors"),
        "audio_tensors": len(audio),
        "backbone_tensors": len(weights) - len(audio),
        "mtp": False,
    }
    (output / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    logger.info(
        "Exported %d trained-checkpoint audio tensors; referenced %d frozen LM tensors",
        len(audio),
        receipt["backbone_tensors"],
    )


def main() -> None:
    """Export a non-MTP projector-only checkpoint using explicit source identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-hf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config-sha256", required=True)
    parser.add_argument("--base-index-sha256", required=True)
    parser.add_argument("--audio-file", type=Path, help="Previously extracted audio, validated against its receipt.")
    parser.add_argument("--audio-receipt", type=Path)
    args = parser.parse_args()
    export_checkpoint(
        base=args.base_hf,
        checkpoint=args.checkpoint,
        data_config=args.data_config,
        output=args.output,
        base_config_sha256=args.base_config_sha256,
        base_index_sha256=args.base_index_sha256,
        audio_file=args.audio_file,
        audio_receipt=args.audio_receipt,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
