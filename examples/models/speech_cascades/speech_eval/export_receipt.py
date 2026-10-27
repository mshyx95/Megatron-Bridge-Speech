# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind a reusable export to its source metadata; never trust a marker alone."""

import argparse
import hashlib
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)
PROFILES = {"qwen35_9b": "qwen3_5_text", "qwen36_35b_a3b": "qwen3_5_moe_text"}
RECEIPT = "vllm-export-ready.json"


def _hash(path: Path) -> str:
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing or empty required artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(*, source_hf: Path, checkpoint: Path, profile: str, step: int) -> dict[str, object]:
    """Fingerprint source metadata, without claiming checkpoint tensor parity."""
    if profile not in PROFILES or step < 1 or checkpoint.name != f"iter_{step:07d}":
        raise ValueError("Unsupported export profile or checkpoint iteration mismatch")
    return {
        "profile": profile,
        "iteration": step,
        "source_hf": str(source_hf.resolve(strict=True)),
        "source_checkpoint": str(checkpoint.resolve(strict=True)),
        "source_metadata": {
            name: _hash(root / name)
            for root, names in (
                (source_hf, ("config.json", "model.safetensors.index.json")),
                (checkpoint, (".metadata", "run_config.yaml")),
            )
            for name in names
        },
    }


def export_identity(output: Path, *, profile: str) -> dict[str, object]:
    """Validate serving files and record metadata hashes and shard sizes.

    Full tensor/header validation remains the existing runtime inspector's job.
    Sizes are a completeness check, not cryptographic weight provenance.
    """
    config = json.loads((output / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5AudioForCausalLM"] or config.get("model_type") != PROFILES[profile]:
        raise ValueError("Export architecture does not match the selected speech profile")
    if profile == "qwen36_35b_a3b" and (config.get("num_experts") != 256 or config.get("num_experts_per_tok") != 8):
        raise ValueError("Export MoE routing metadata does not match 256 experts / top-8")
    weight_map = json.loads((output / "model.safetensors.index.json").read_text()).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Export has an empty or invalid safetensors index")
    shards = {}
    for filename in weight_map.values():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Safetensors shards must be local filenames")
        path = output / filename
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty export shard: {filename}")
        shards[filename] = path.stat().st_size
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "tensor_count": len(weight_map),
        "shard_sizes": shards,
        "metadata_sha256": {
            name: _hash(output / name)
            for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json", "tokenizer.json")
        },
    }


def check_export(
    output: Path, *, action: str, source_hf: Path, checkpoint: Path, profile: str, step: int
) -> dict[str, object]:
    """Preflight, publish, or revalidate a non-overwriting export receipt."""
    identity = source_identity(source_hf=source_hf, checkpoint=checkpoint, profile=profile, step=step)
    if action == "preflight":
        return identity
    report = {"schema_version": 1, **identity, **export_identity(output, profile=profile)}
    receipt = output / RECEIPT
    if action == "verify":
        if json.loads(receipt.read_text()) != report:
            raise ValueError("Export receipt is stale, legacy, or belongs to different source metadata")
    elif action == "write":
        with receipt.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    else:
        raise ValueError(f"Unknown action: {action}")
    return report


def main() -> None:
    """Validate an export without importing GPU libraries or loading weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "write", "verify"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-hf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    check_export(**vars(args))
    logging.basicConfig(level=logging.INFO)
    logger.info("Export %s passed: %s", args.action, args.output)


if __name__ == "__main__":
    main()
