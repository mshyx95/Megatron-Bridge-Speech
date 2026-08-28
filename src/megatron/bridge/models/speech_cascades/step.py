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

"""Training forward step for the speech-cascades ALM.

The ALM data collator emits **unshifted** labels (``labels[i]`` is the target token *at* position
``i``, matching the captured reference batch). Megatron-Core's GPT loss computes
``CrossEntropy(logits[i], labels[i])`` with no internal shift, so this step applies the standard
causal next-token shift here: ``shifted[i] = labels[i + 1]``. Ignored positions are masked via
``loss_mask`` and their labels are clamped to ``0`` so the cross-entropy gather stays in range.
"""

import logging
from functools import partial
from typing import Any, Iterable

import torch
from megatron.core.transformer import MegatronModule

from megatron.bridge.models.speech_cascades.obs_logging import add_audio_seconds
from megatron.bridge.training.losses import create_masked_next_token_loss_function
from megatron.bridge.training.state import GlobalState


logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _to_cuda(value: torch.Tensor | None) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor) and torch.cuda.is_available():
        return value.cuda(non_blocking=True)
    return value


def get_alm_batch(data_iterator: Iterable) -> dict[str, Any]:
    """Pull one ALM batch off the iterator and move tensors to the current device.

    Applies the causal next-token shift to the (unshifted) collator labels and derives ``loss_mask``.
    """
    batch = next(data_iterator)

    input_ids = _to_cuda(batch["input_ids"])
    labels = _to_cuda(batch["labels"])

    # Causal shift: score logits[i] against labels[i + 1]; last position is ignored.
    shifted = labels.new_full(labels.shape, IGNORE_INDEX)
    shifted[:, :-1] = labels[:, 1:]
    loss_mask = (shifted != IGNORE_INDEX).float()
    model_labels = shifted.clamp_min(0)

    # Track audio duration consumed (audio_frames * 10ms), matching phyagi's BatchTracker.
    add_audio_seconds(batch.get("audio_frames"))

    audio_features = batch.get("inputs_embeds")
    audio_features = _to_cuda(audio_features) if isinstance(audio_features, torch.Tensor) else None
    if audio_features is not None and audio_features.numel() == 0:
        audio_features = None
    elif audio_features is not None:
        # Log-mel features arrive as float32; match the bf16 audio encoder weights.
        audio_features = audio_features.to(torch.bfloat16)

    return {
        "input_ids": input_ids,
        "labels": model_labels,
        "loss_mask": loss_mask,
        "audio_input_features": audio_features,
        "audio_attention_mask": _to_cuda(batch.get("audio_attention_mask")),
        "audio_embed_sizes": _to_cuda(batch.get("audio_embed_sizes")),
    }


def forward_step(
    state: GlobalState, data_iterator: Iterable, model: MegatronModule, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step for :class:`SpeechCascadesModel`.

    Args:
        state: Global run state (timers, rerun config).
        data_iterator: Iterator yielding ALM collator batches.
        model: The :class:`SpeechCascadesModel` (possibly wrapped).
        return_schedule_plan: Unused; kept for forward-step signature compatibility.

    Returns:
        Tuple of the per-token loss tensor and the masked-mean loss function.
    """
    timers = state.timers

    timers("batch-generator", log_level=2).start()
    batch = get_alm_batch(data_iterator)
    timers("batch-generator").stop()

    loss_mask = batch.pop("loss_mask")

    forward_args = {
        "input_ids": batch["input_ids"],
        "position_ids": None,
        "attention_mask": None,
        "labels": batch["labels"],
    }
    if batch["audio_input_features"] is not None:
        forward_args["audio_input_features"] = batch["audio_input_features"]
        forward_args["audio_attention_mask"] = batch["audio_attention_mask"]
        forward_args["audio_embed_sizes"] = batch["audio_embed_sizes"]

    output_tensor = model(**forward_args)

    check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
    loss_function = create_masked_next_token_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

    return output_tensor, loss_function
