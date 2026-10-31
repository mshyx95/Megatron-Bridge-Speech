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

from megatron.bridge.models.speech_cascades.distillation import (
    chunked_kl_divergence,
    distillation_loss,
    mtp_depth_loss,
)
from megatron.bridge.models.speech_cascades.obs_logging import add_audio_seconds
from megatron.bridge.training.losses import create_masked_next_token_loss_function, masked_next_token_loss
from megatron.bridge.training.state import GlobalState


logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# Defensive defaults for the KD hyperparameters until WP-C1's provider and WP-A3's recipe wire them
# onto the model config; these mirror the reference stage-2 S2T recipe.
_DEFAULT_DISTILLATION_WEIGHT = 0.75
_DEFAULT_DISTILLATION_TEMPERATURE = 2.0
_DEFAULT_DISTILLATION_KL_CHUNK_SIZE = 128


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

    out = {
        "input_ids": input_ids,
        "labels": model_labels,
        "loss_mask": loss_mask,
        "audio_input_features": audio_features,
        "audio_attention_mask": _to_cuda(batch.get("audio_attention_mask")),
        "audio_embed_sizes": _to_cuda(batch.get("audio_embed_sizes")),
    }
    _add_teacher_tensors(batch, out)
    return out


def _add_teacher_tensors(batch: dict[str, Any], out: dict[str, Any]) -> None:
    """Move KD teacher tensors (when present) to the current device, leaving the ASR path untouched.

    Keys are only added when the collator produced them (``knowledge_distillation`` enabled), so the
    non-KD batch dict is byte-identical to the ASR path. ``teacher_input_ids`` / ``teacher_labels`` are
    ``long`` and ``teacher_attention_mask`` is ``bool``; dtypes are preserved by the device move.
    """
    teacher_input_ids = batch.get("teacher_input_ids")
    if teacher_input_ids is None:
        return
    out["teacher_input_ids"] = _to_cuda(teacher_input_ids)
    out["teacher_labels"] = _to_cuda(batch.get("teacher_labels"))
    out["teacher_attention_mask"] = _to_cuda(batch.get("teacher_attention_mask"))


def forward_step(
    state: GlobalState, data_iterator: Iterable, model: MegatronModule, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step for :class:`SpeechCascadesModel`.

    When the batch carries KD teacher tensors (``teacher_input_ids``), this runs the dual-forward
    knowledge-distillation path (WP-A2): the student speech forward exposes the response-token hidden
    states, a text-only teacher forward of the frozen LM produces the aligned teacher hidden states,
    and the loss mixes a chunked KL with the existing masked next-token CE. Without teacher tensors
    the ASR path is byte-identical to before.

    Args:
        state: Global run state (timers, rerun config).
        data_iterator: Iterator yielding ALM collator batches.
        model: The :class:`SpeechCascadesModel` (possibly wrapped).
        return_schedule_plan: Unused; kept for forward-step signature compatibility.

    Returns:
        Tuple of the per-token loss tensor and the loss function (masked CE, or the KD mix).
    """
    timers = state.timers

    timers("batch-generator", log_level=2).start()
    batch = get_alm_batch(data_iterator)
    timers("batch-generator").stop()

    loss_mask = batch.pop("loss_mask")
    kd_enabled = batch.get("teacher_input_ids") is not None

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
    if kd_enabled:
        # Ask the model to expose the student response hidden states for the KD forward.
        forward_args["return_hidden_states"] = True

    output_tensor = model(**forward_args)

    check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss

    if not kd_enabled:
        loss_function = create_masked_next_token_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
        return output_tensor, loss_function

    inner = _unwrap_model(model)
    kl_loss = _compute_kd_kl(inner, batch, loss_mask)
    mtp_loss = _compute_mtp_loss(inner, batch, loss_mask)
    distillation_weight = float(getattr(inner.config, "distillation_weight", _DEFAULT_DISTILLATION_WEIGHT))
    loss_function = partial(
        _kd_loss_function,
        loss_mask,
        kl_loss,
        distillation_weight,
        check_for_nan_in_loss,
        check_for_spiky_loss,
        mtp_loss=mtp_loss,
    )
    return output_tensor, loss_function


def _unwrap_model(model: MegatronModule) -> MegatronModule:
    """Unwrap DDP/Float16 wrappers to reach the inner :class:`SpeechCascadesModel`."""
    inner = model
    while not hasattr(inner, "text_forward_hidden_states") and hasattr(inner, "module"):
        inner = inner.module
    return inner


def _make_logits_fn(inner: MegatronModule):
    """Build a ``[N, H] -> [N, V]`` logits function from the frozen tied output head."""
    gpt = inner.language_model
    weight = gpt.shared_embedding_or_output_weight()

    def logits_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        # The output layer expects sbhd ``[S, B, H]``; treat the N response tokens as a length-N,
        # batch-1 sequence, then drop the batch axis.
        logits, _ = gpt.output_layer(hidden_states.unsqueeze(1), weight=weight)
        return logits.squeeze(1)

    return logits_fn


def _compute_kd_kl(inner: MegatronModule, batch: dict[str, Any], loss_mask: torch.Tensor) -> torch.Tensor:
    """Run the teacher text forward and compute the chunked KL over aligned response tokens.

    The student response hidden states come from the just-completed speech forward
    (``inner.captured_hidden_states``); the teacher hidden states come from a text-only forward of
    the frozen LM under ``torch.no_grad()``. A hard assert requires the student and teacher response
    label ids to match (token alignment).
    """
    student_hidden = inner.captured_hidden_states  # [B, S, H], carries the student graph
    if student_hidden is None:
        raise RuntimeError(
            "Knowledge distillation is enabled but the student hidden states were not captured; "
            "the model forward must be called with return_hidden_states=True"
        )

    # Student response selection mirrors the reference: shifted labels, response tokens only.
    student_label_mask = loss_mask[:, :-1].bool()  # [B, S-1]
    student_response_hidden = student_hidden[:, :-1, :][student_label_mask]  # [N, H]
    student_response_labels = batch["labels"][:, :-1][student_label_mask]  # [N]

    with torch.no_grad():
        teacher_hidden = inner.text_forward_hidden_states(batch["teacher_input_ids"], attention_mask=None)
    teacher_shift_labels = batch["teacher_labels"][:, 1:]  # [B, T-1]
    teacher_label_mask = teacher_shift_labels != IGNORE_INDEX
    teacher_response_hidden = teacher_hidden[:, :-1, :][teacher_label_mask]  # [N, H]
    teacher_response_labels = teacher_shift_labels[teacher_label_mask]  # [N]

    if not torch.equal(student_response_labels, teacher_response_labels):
        raise ValueError("KD student and teacher response labels are not token-aligned")

    config = inner.config
    temperature = float(getattr(config, "distillation_temperature", _DEFAULT_DISTILLATION_TEMPERATURE))
    chunk_size = int(getattr(config, "distillation_kl_chunk_size", _DEFAULT_DISTILLATION_KL_CHUNK_SIZE))
    return chunked_kl_divergence(
        student_response_hidden,
        teacher_response_hidden.to(student_response_hidden.dtype),
        _make_logits_fn(inner),
        temperature,
        chunk_size,
    )


def _compute_mtp_loss(inner: MegatronModule, batch: dict[str, Any], loss_mask: torch.Tensor) -> torch.Tensor | None:
    """Compute the aggregated MTP per-depth KL+CE loss, or ``None`` when MTP is disabled.

    Gated to a no-op (returns ``None``) when ``mtp_num_hidden_layers == 0`` or the model did not
    expose per-depth hidden states, so the WP-A2 KD path and the ASR path stay byte-identical. The
    depth-KL teacher target is the student's own main hidden states (``captured_hidden_states``,
    ``h^0``), detached inside :func:`mtp_depth_loss` (self-distillation, no teacher-text forward).
    """
    config = inner.config
    if int(getattr(config, "mtp_num_hidden_layers", 0)) == 0:
        return None
    mtp_hidden_states = getattr(inner, "mtp_hidden_states", None)
    if not mtp_hidden_states:
        return None
    main_hidden = inner.captured_hidden_states
    if main_hidden is None:
        return None

    temperature = float(getattr(config, "distillation_temperature", _DEFAULT_DISTILLATION_TEMPERATURE))
    chunk_size = int(getattr(config, "distillation_kl_chunk_size", _DEFAULT_DISTILLATION_KL_CHUNK_SIZE))
    mtp_loss, _ = mtp_depth_loss(
        mtp_hidden_states,
        main_hidden,
        batch["labels"],
        loss_mask,
        _make_logits_fn(inner),
        temperature=temperature,
        chunk_size=chunk_size,
        mtp_distillation_weight=float(getattr(config, "mtp_distillation_weight", 0.0)),
        mtp_loss_weight=float(getattr(config, "mtp_loss_weight", 0.0)),
        mtp_loss_depth_weights=getattr(config, "mtp_loss_depth_weights", None),
    )
    return mtp_loss


def _kd_loss_function(
    loss_mask: torch.Tensor,
    kl_loss: torch.Tensor,
    distillation_weight: float,
    check_for_nan_in_loss: bool,
    check_for_spiky_loss: bool,
    output_tensor: torch.Tensor,
    *,
    mtp_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Mix the precomputed KD KL with the existing masked next-token CE (+ optional MTP loss).

    ``masked_next_token_loss`` returns a token-summed CE and the pipeline schedule later divides the
    returned scalar by ``num_tokens``. The KL from ``chunked_kl_divergence`` is a per-token mean, so
    it is scaled by ``num_tokens`` before mixing to keep the effective loss
    ``w * KL_mean + (1 - w) * CE_mean`` after normalization (matching the reference).

    WP-C2: when ``mtp_loss`` is provided it is the aggregated per-depth MTP KL+CE (already scaled by
    ``mtp_loss_weight`` and reduced across depths by :func:`mtp_depth_loss`), a per-token-mean scale,
    so it is likewise multiplied by ``num_tokens`` before being added onto the total. ``None`` (MTP
    disabled) leaves the WP-A2 KD loss byte-identical.
    """
    ce_loss, num_tokens, reporting = masked_next_token_loss(
        loss_mask, output_tensor, check_for_nan_in_loss, check_for_spiky_loss
    )
    scaled_kl = kl_loss * num_tokens.clamp(min=1)
    total_loss = distillation_loss(scaled_kl, ce_loss, distillation_weight)
    if mtp_loss is not None:
        total_loss = total_loss + mtp_loss * num_tokens.clamp(min=1)
    reporting["ce loss"] = reporting["lm loss"]
    reporting["kl loss"] = torch.cat([scaled_kl.detach().view(1), num_tokens.view(1)])
    reporting["lm loss"] = torch.cat([total_loss.detach().view(1), num_tokens.view(1)])
    return total_loss, num_tokens, reporting
