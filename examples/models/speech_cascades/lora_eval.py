# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load Speech Cascades trainable weights on top of the original frozen base."""

import logging
from dataclasses import fields
from pathlib import Path


logger = logging.getLogger(__name__)


def configure_expert_scaling(model: list, peft, mode: str) -> None:
    """Reproduce the expert scaling used when the adapter checkpoint was trained."""
    from megatron.bridge.peft.utils import is_expert_linear

    if mode not in {"normalized", "legacy"}:
        raise ValueError(f"Unknown expert scaling mode: {mode}")
    for chunk in model:
        for name, module in chunk.named_modules():
            if is_expert_linear(name) and hasattr(module, "adapter"):
                adapter = module.adapter
                if mode == "legacy":
                    if not peft.normalize_moe_lora:
                        raise ValueError("Legacy expert scaling requires normalized MoE LoRA")
                    adapter.alpha = peft.alpha
                logger.info(
                    "Eval expert adapter %s: rank=%s alpha=%s scaling=%s",
                    name, adapter.dim, adapter.alpha, adapter.alpha / adapter.dim,
                )


def build_lora_model(
    hf_path: str,
    checkpoint: str,
    *,
    base_checkpoint: str | None,
    expert_scaling: str,
    tensor_parallel_size: int,
):
    """Build the frozen base before injecting and restoring Speech Cascades LoRA."""
    import torch
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.utils import unwrap_model

    from megatron.bridge import AutoBridge
    from megatron.bridge.models.speech_cascades import dcp_compat  # noqa: F401
    from megatron.bridge.models.speech_cascades.peft import SpeechCascadesLoRA
    from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
    from megatron.bridge.training.utils.checkpoint_utils import read_run_config

    run_config = read_run_config(str(Path(checkpoint) / "run_config.yaml"))
    peft_config = run_config.get("peft") or {}
    if peft_config.get("_target_") != "megatron.bridge.models.speech_cascades.peft.SpeechCascadesLoRA":
        raise ValueError("Evaluation requires a SpeechCascadesLoRA checkpoint")
    source = Path(run_config["checkpoint"]["pretrained_checkpoint"])
    if base_checkpoint is None:
        if source.resolve() != Path(hf_path).resolve():
            raise ValueError("Specify --base-checkpoint for the original pretrained Megatron base")
    elif source.resolve() not in {Path(base_checkpoint).resolve(), Path(base_checkpoint).resolve().parent}:
        raise ValueError("The supplied base checkpoint does not match the saved training configuration")
    allowed = {field.name for field in fields(SpeechCascadesLoRA) if field.init}
    peft = SpeechCascadesLoRA(**{key: value for key, value in peft_config.items() if key in allowed})
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    provider = bridge.to_megatron_provider(load_weights=base_checkpoint is None)
    provider.tensor_model_parallel_size = tensor_parallel_size
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = tensor_parallel_size
    provider.sequence_parallel = False
    provider.parallel_output = False
    provider.params_dtype = torch.bfloat16
    provider.pipeline_dtype = torch.bfloat16
    provider.attention_backend = AttnBackend.flash
    provider.mtp_num_hidden_layers = 0
    provider.moe_grouped_gemm = run_config["model"]["moe_grouped_gemm"]
    provider.moe_token_dispatcher_type = run_config["model"]["moe_token_dispatcher_type"]
    provider.finalize()

    def prepare(chunks):
        if base_checkpoint is not None:
            _load_model_weights_from_checkpoint(base_checkpoint, chunks, strict=True)
        peft(chunks, training=True)
        configure_expert_scaling(chunks, peft, expert_scaling)
        return chunks

    provider.register_pre_wrap_hook(prepare)
    model = provider.provide_distributed_model(wrap_with_ddp=False, bf16=True)
    load_lora_weights(model, peft, checkpoint)
    for chunk in model:
        chunk.eval()
        chunk.requires_grad_(False)
    torch.cuda.empty_cache()
    logger.info("Loaded LoRA checkpoint %s on base %s", checkpoint, base_checkpoint or hf_path)
    return unwrap_model(model)[0]


def load_lora_weights(model: list, peft, checkpoint: str) -> None:
    """Restore adapters and all other parameters selected by the training policy."""
    from megatron.core import dist_checkpointing
    from megatron.core.utils import unwrap_model

    from megatron.bridge.training.checkpointing import (
        _generate_model_state_dict,
        apply_peft_adapter_filter_to_state_dict,
    )

    chunks = unwrap_model(model)
    peft.set_params_to_save(chunks)
    common = dist_checkpointing.load_common_state_dict(checkpoint)
    metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=common)
    sharded = _generate_model_state_dict(chunks, {"metadata": metadata})
    selected = apply_peft_adapter_filter_to_state_dict(sharded, peft)
    if not any(".adapter." in name for section in selected.values() for name in section):
        raise ValueError("No LoRA adapter parameters selected for evaluation")
    loaded = dist_checkpointing.load(selected, checkpoint)
    for index, chunk in enumerate(chunks):
        key = "model" if len(chunks) == 1 else f"model{index}"
        expected = set(selected[key])
        missing = expected - set(loaded[key])
        if missing:
            raise ValueError(f"Checkpoint is missing trainable parameters: {sorted(missing)}")
        incompatible = chunk.load_state_dict(loaded[key], strict=False)
        missing = expected.intersection(incompatible.missing_keys)
        if missing or incompatible.unexpected_keys:
            raise ValueError(f"LoRA checkpoint mismatch: missing={sorted(missing)}, {incompatible.unexpected_keys=}")
        chunk.eval()