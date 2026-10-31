# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PEFT policies for Speech Cascades models."""

from dataclasses import dataclass

from megatron.core.utils import unwrap_model

from megatron.bridge.peft.lora import LoRA


SPEECH_CASCADES_LM_LORA_TARGETS = [
    "language_model.decoder.layers.*.self_attention.linear_qkv",
    "language_model.decoder.layers.*.self_attention.linear_proj",
    "language_model.decoder.layers.*.self_attention.in_proj",
    "language_model.decoder.layers.*.self_attention.out_proj",
    "language_model.decoder.layers.*.mlp*.linear_fc1",
    "language_model.decoder.layers.*.mlp*.linear_fc2",
]


@dataclass
class SpeechCascadesLoRA(LoRA):
    """Train LM LoRA adapters together with the Speech Cascades audio module.

    This mirrors the production Speech Cascades policy: the audio encoder and projector remain
    trainable, the language-model base is frozen, and the standalone GDN state parameters remain
    trainable. Qwen3.5's HF ``in_proj_a`` and ``in_proj_b`` weights are fused into MCore's single
    ``in_proj`` weight, so their adaptation is represented by the ``in_proj`` LoRA adapter.
    """

    target_modules: list[str] | None = None
    freeze_audio_model: bool = False
    train_gdn_state: bool = True

    def __post_init__(self) -> None:
        if self.target_modules is None:
            self.target_modules = list(SPEECH_CASCADES_LM_LORA_TARGETS)

    def freeze_model(self, model, training: bool = True) -> None:
        unwrapped = unwrap_model(model)
        model_chunks = unwrapped if isinstance(unwrapped, list) else [unwrapped]

        for model_chunk in model_chunks:
            language_model = getattr(model_chunk, "language_model", None)
            if language_model is not None:
                for param in language_model.parameters():
                    param.requires_grad = False
                if self.train_gdn_state:
                    for name, param in language_model.named_parameters():
                        if name.endswith(("conv1d.weight", "conv1d.bias", "A_log", "dt_bias")):
                            param.requires_grad = True

            audio_model = getattr(model_chunk, "audio_model", None)
            if self.freeze_audio_model and audio_model is not None:
                for param in audio_model.parameters():
                    param.requires_grad = False

            if training:
                model_chunk.train(mode=True)