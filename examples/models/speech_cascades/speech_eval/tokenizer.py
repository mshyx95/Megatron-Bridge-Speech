# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the checkpoint's serialized tokenizer without model-class regex overrides."""

import json
from pathlib import Path


def load_official_tokenizer(path: str | Path):
    """Preserve serialized rules, vocabulary, special IDs and chat template."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    root = Path(path)
    saved = json.loads(Tokenizer.from_file(str(root / "tokenizer.json")).to_str())
    config = json.loads((root / "tokenizer_config.json").read_text())
    tokenizer = PreTrainedTokenizerFast.from_pretrained(root, local_files_only=True, fix_mistral_regex=False)
    loaded = json.loads(tokenizer.backend_tokenizer.to_str())
    for field in ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder"):
        if saved.get(field) != loaded.get(field):
            raise ValueError(f"Tokenizer loader changed serialized {field}")
    for token_id, token in config.get("added_tokens_decoder", {}).items():
        if tokenizer.convert_tokens_to_ids(token["content"]) != int(token_id):
            raise ValueError(f"Tokenizer loader changed special token ID: {token['content']}")
    return tokenizer


def prepare_tokenizer_view(source: Path, view: Path) -> None:
    """Write only local loader metadata; never modify original tokenizer assets."""
    destination = view / "tokenizer_config.json"
    if source.resolve() == view.resolve() or destination.exists() or destination.is_symlink():
        raise ValueError("Tokenizer metadata view must be new and separate from its source")
    config = json.loads((source / "tokenizer_config.json").read_text())
    if config.get("tokenizer_class") not in {
        "Qwen2Tokenizer",
        "Qwen2TokenizerFast",
        "PreTrainedTokenizerFast",
        "TokenizersBackend",
    }:
        raise ValueError("Expected a serialized Qwen speech tokenizer")
    if config.get("auto_map"):
        raise ValueError("Custom tokenizer code is not supported by this metadata view")
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config["fix_mistral_regex"] = False
    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
