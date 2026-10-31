# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Forward-side simul sequence builder for the raw-audio S2S path.

Given per-sample ``seq_plan`` entries (resolved conversation messages, question
audio embed sizes, alignment/use_eoc, codec prefix info) and flat codec-token
lists produced by an :class:`AudioTokenizer`, this module rebuilds the exact same
``input_ids`` / ``labels`` / ``input_codecs`` / ``label_codecs`` tensors that the
dataloader-side transform (``SpeechOutAudioTextConversationTransform``) + collator
produce for the ``.answer_cv_token`` path.

It REUSES ``S2SConversation.get_prompt`` (and hence ``get_codec_input``) verbatim
— the only reliable route to bit-exact equivalence — and merges the transform's
``_pad_codec_batch`` + collator cross-example padding into a single pass to one
global max (so the compared region never mixes pad values).
"""

import re
from typing import Dict, List, Optional, Tuple

import torch

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.s2s_alm_sample_processor import (
    S2SBlobChunkALMSampleProcessor,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.conversation_utils import (
    _audio_tok_in_message,
    _audio_tok_match_message,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.s2s_conversation_utils import conv_templates
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger

logger = get_logger(__name__)

IGNORE_IDX = -100


def _codec_layer_count(codec_cfg: Optional[dict]) -> int:
    codec_cfg = codec_cfg or {}
    return codec_cfg.get("codec_layer", 1)


def _audio_pad_id(codec_cfg: Optional[dict]) -> int:
    codec_cfg = codec_cfg or {}
    return codec_cfg.get("audio_vocab_size", 6624) + codec_cfg.get("text_vocab_size", 200064) - 2


def _rebuild_conversation(seq_plan: Dict, tokenizer, codec_cfg: Optional[dict]):
    """Recreate the resolved ``S2SConversation`` for one sample (no re-randomization)."""
    template_name = seq_plan["conv_template"]
    conv = conv_templates[template_name].copy()
    conv.tokenizer = tokenizer
    conv.codec_cfg = codec_cfg
    conv.messages = [list(message) for message in seq_plan["messages"]]
    conv.system = seq_plan.get("system", conv.system)
    return conv


def build_sample_sequence(
    seq_plan: Dict,
    codec_tokens: Optional[List[int]],
    tokenizer,
    codec_cfg: Optional[dict],
):
    """Build the unpadded (input_ids, labels, input_codecs, label_codecs) for one sample.

    ``codec_tokens`` is the flat codec-token list from the audio tokenizer (or
    ``None`` for text-only / failed audio). Returns a 4-tuple; ``input_codecs`` /
    ``label_codecs`` are ``None`` when the sample has no codec output (text-only).
    """
    codec_layer = _codec_layer_count(codec_cfg)
    prefix_sample = seq_plan.get("codec_prefix_sample", {})

    codec = None
    if codec_tokens is not None:
        codec = S2SBlobChunkALMSampleProcessor.build_target_codec_layers(
            codec_tokens, codec_layer, codec_cfg or {}, prefix_sample
        )

    conv = _rebuild_conversation(seq_plan, tokenizer, codec_cfg)
    ret, input_id, label, codec_labels = conv.get_prompt(
        is_train=True,
        audio_input_id=-1,
        only_add_first_audio_occurrence=False,
        num_audio_tokens=seq_plan.get("num_audio_tokens", []),
        codecs_list=[codec],
        alignments_list=[seq_plan.get("alignment")],
        use_eoc_endings_list=[seq_plan.get("use_eoc_ending", False)],
    )

    # get_prompt returns codec tuple (input_codecs, input_ids, labels, label_codecs)
    # only when codec output exists; otherwise (ret_str, input_ids, labels, audio_positions).
    # Gate on the actual return contract (not the input `codec`) so fake-codec
    # patterns — where get_prompt emits codec output with codec is None — are kept,
    # matching transform's _has_codec_output.
    has_codec = isinstance(ret, list) and len(ret) == codec_layer and isinstance(ret[0], list)
    if has_codec:
        return input_id, label, ret, codec_labels
    return input_id, label, None, None


def build_packed_sample_sequence(
    seq_plan: Dict,
    codec_tokens_multi: List[Optional[List[int]]],
    tokenizer,
    codec_cfg: Optional[dict],
):
    """Build the unpadded packed sequence for one ``samples_to_sequence`` seq_plan.

    ``codec_tokens_multi`` is the list of ``num_samples`` flat codec-token lists (one
    per packed answer; entries may be ``None`` for text-only / failed audio). Mirrors
    the OLD ``transform_concat_to_context_length`` exactly: builds K codecs, then a
    SINGLE ``get_prompt`` with the full ``codecs_list`` / ``num_audio_tokens`` /
    ``alignments_list`` / ``use_eoc_endings_list``, then applies the
    ``pad_to_context_length`` truncation. Returns a 4-tuple; ``input_codecs`` /
    ``label_codecs`` are ``None`` when the packed sequence has no codec output.
    """
    codec_layer = _codec_layer_count(codec_cfg)
    num_samples = seq_plan.get("num_samples", len(codec_tokens_multi))
    prefix_samples = seq_plan.get("codec_prefix_samples", [{}] * num_samples)

    codecs = []
    for tokens, prefix in zip(codec_tokens_multi, prefix_samples):
        codec = None
        if tokens is not None:
            codec = S2SBlobChunkALMSampleProcessor.build_target_codec_layers(
                tokens, codec_layer, codec_cfg or {}, prefix
            )
        codecs.append(codec)

    conv = _rebuild_conversation(seq_plan, tokenizer, codec_cfg)
    ret, input_id, label, codec_labels = conv.get_prompt(
        is_train=True,
        audio_input_id=-1,
        only_add_first_audio_occurrence=False,
        num_audio_tokens=seq_plan.get("num_audio_tokens", []),
        codecs_list=codecs,
        alignments_list=seq_plan.get("alignments", [None] * num_samples),
        use_eoc_endings_list=seq_plan.get("use_eoc_endings", [False] * num_samples),
    )

    has_codec = isinstance(ret, list) and len(ret) == codec_layer and isinstance(ret[0], list)

    if seq_plan.get("pad_to_context_length"):
        context_length = seq_plan["context_length"]
        input_id = input_id[:context_length]
        label = label[:context_length]
        if has_codec:
            ret = [layer[:context_length] for layer in ret]
            codec_labels = [layer[:context_length] for layer in codec_labels]

    if has_codec:
        return input_id, label, ret, codec_labels
    return input_id, label, None, None


def build_simul_batch(
    seq_plan_list: List[Dict],
    codec_tokens_list: List[Optional[List[int]]],
    tokenizer,
    codec_cfg: Optional[dict],
    batch_format: str = "samples_to_batch",
    shift_labels: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build padded batch tensors for the simul pattern.

    Args:
        seq_plan_list: len-B list of resolved per-sample plans.
        codec_tokens_list: len-B list of flat codec-token lists (or ``None``).
        tokenizer: text tokenizer (needs ``pad_token_id`` + ``encode``).
        codec_cfg: dataset-side codec params (audio/text vocab sizes, prefix, ...).
        batch_format: ``samples_to_batch`` (ROW-drop shift) or
            ``samples_to_sequence`` (TOKEN-drop shift). Used as the fallback when a
            ``seq_plan`` entry does not carry its own ``batch_format``, and to select
            the batch-level ``shift_labels`` semantics.
        shift_labels: whether to apply the transform's label shift.
        device: target device for the built tensors.

    Returns:
        Dict with ``input_ids`` [B,T], ``labels`` [B,Tl], ``input_codecs``
        [B,cl,T], ``label_codecs`` [B,cl,Tl].

    Each ``seq_plan`` entry consumes ``entry["num_samples"]`` (default 1) codec-token
    lists from ``codec_tokens_list`` and produces ONE output row: a single sample row
    for ``samples_to_batch`` entries, or one packed context-length row for
    ``samples_to_sequence`` entries. This supports batches that mix both formats.
    """
    total_samples = sum(sp.get("num_samples", 1) for sp in seq_plan_list)
    assert total_samples == len(codec_tokens_list), (
        f"seq_plan num_samples sum ({total_samples}) and codec_tokens "
        f"({len(codec_tokens_list)}) length mismatch"
    )

    codec_layer = _codec_layer_count(codec_cfg)
    audio_pad_id = _audio_pad_id(codec_cfg)
    pad_token_id = tokenizer.pad_token_id

    input_ids: List[List[int]] = []
    labels: List[List[int]] = []
    input_codecs: List[List[List[int]]] = []
    label_codecs: List[List[List[int]]] = []

    cursor = 0
    for seq_plan in seq_plan_list:
        entry_format = seq_plan.get("batch_format", batch_format)
        num_samples = seq_plan.get("num_samples", 1)
        entry_tokens = codec_tokens_list[cursor : cursor + num_samples]
        cursor += num_samples

        if entry_format == "samples_to_sequence":
            built_rows = [build_packed_sample_sequence(seq_plan, entry_tokens, tokenizer, codec_cfg)]
        else:
            built_rows = [build_sample_sequence(seq_plan, entry_tokens[0], tokenizer, codec_cfg)]

        for sample_input_ids, sample_labels, sample_input_codecs, sample_label_codecs in built_rows:
            input_ids.append(sample_input_ids)
            labels.append(sample_labels)
            if sample_input_codecs is None:
                # Text-only sample: fill codec layers with ignore/pad so shapes stay [B, cl, .].
                sample_input_codecs = [[audio_pad_id] * len(sample_input_ids) for _ in range(codec_layer)]
                sample_label_codecs = [[IGNORE_IDX] * len(sample_labels) for _ in range(codec_layer)]
            input_codecs.append(sample_input_codecs)
            label_codecs.append(sample_label_codecs)

    max_input_id_len = max(len(ids) for ids in input_ids)
    max_label_len = max(len(lab) for lab in labels)

    input_ids = [ids + [pad_token_id] * (max_input_id_len - len(ids)) for ids in input_ids]
    labels = [lab + [IGNORE_IDX] * (max_label_len - len(lab)) for lab in labels]

    for i in range(len(input_codecs)):
        for j in range(codec_layer):
            input_codecs[i][j] = input_codecs[i][j] + [audio_pad_id] * (
                max_input_id_len - len(input_codecs[i][j])
            )
            label_codecs[i][j] = label_codecs[i][j] + [IGNORE_IDX] * (
                max_label_len - len(label_codecs[i][j])
            )

    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    input_codecs_t = torch.tensor(input_codecs, dtype=torch.long)
    label_codecs_t = torch.tensor(label_codecs, dtype=torch.long)

    # ROOT-CAUSE FIX (positional): the codec label stream must carry an audio TARGET only at
    # audio-codec positions. `append_codec_segment` builds label_codecs as `list(labels) + layer`,
    # i.e. it prepends the running TEXT labels into the audio-codec stream. At assistant-TEXT
    # positions the TEXT token id (non-IGNORE) therefore leaks into label_codecs, and the model's
    # codec mask `label_codecs[:,0,1:] != -100` selects those positions, feeding TEXT token ids
    # into the audio F.cross_entropy (n_classes=audio_vocab_size). Chinese text ids (>6624) trip a
    # CUDA device-side assert that aborts the whole 32-GPU job; smaller text ids (<6624) silently
    # poison the audio loss. There is never an audio target where a text token is predicted, so
    # mask every codec-label position with a predicted TEXT label (labels_t != IGNORE) to IGNORE
    # across all codec layers. label_codecs_t and labels_t are position-aligned and both unshifted
    # here (shift happens later / inside the model), so this alignment is exact.
    text_pred_mask = (labels_t != IGNORE_IDX).unsqueeze(1).expand_as(label_codecs_t)
    label_codecs_t = label_codecs_t.masked_fill(text_pred_mask, IGNORE_IDX)

    # Secondary guard: the audio loss is F.cross_entropy over `audio_vocab_size` classes, so every
    # audio TARGET must be in [0, audio_vocab_size) or == IGNORE_IDX. The live
    # forward-tokenize path can occasionally yield a label token outside that range
    # (e.g. a stray/garbage codec id on a pathological long sample) which the clean
    # precomputed `.answer_cv_token` path never produced; feeding it to nll_loss trips a
    # CUDA device-side assert (`t >= 0 && t < n_classes`) that aborts the whole 32-GPU job.
    # Mask any such invalid target to IGNORE_IDX (matching the `.answer_cv_token` pad
    # convention) and log the offending values so the source can be traced. No-op when all
    # labels are already valid, so it stays bit-exact with the clean path.
    audio_vocab_size = (codec_cfg or {}).get("audio_vocab_size", 6624)
    if audio_vocab_size:
        invalid_mask = (label_codecs_t != IGNORE_IDX) & (
            (label_codecs_t < 0) | (label_codecs_t >= audio_vocab_size)
        )
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            bad_vals = torch.unique(label_codecs_t[invalid_mask]).tolist()
            logger.warning(
                "[forward_tokenize] masked %d out-of-range audio label token(s) to IGNORE "
                "(audio_vocab_size=%d, valid=[0,%d)); offending values=%s. label_codecs shape=%s.",
                n_invalid,
                audio_vocab_size,
                audio_vocab_size,
                bad_vals[:32],
                tuple(label_codecs_t.shape),
            )
            label_codecs_t[invalid_mask] = IGNORE_IDX

    if shift_labels:
        if batch_format == "samples_to_batch":
            # ROW drop: mirrors transform_concat_to_batch `labels = labels[1:]`.
            labels_t = labels_t[1:]
        else:
            # TOKEN drop: mirrors transform_concat_to_context_length `labels = labels[1:]`.
            labels_t = labels_t[:, 1:]

    if device is not None:
        input_ids_t = input_ids_t.to(device)
        labels_t = labels_t.to(device)
        input_codecs_t = input_codecs_t.to(device)
        label_codecs_t = label_codecs_t.to(device)

    return {
        "input_ids": input_ids_t,
        "labels": labels_t,
        "input_codecs": input_codecs_t,
        "label_codecs": label_codecs_t,
    }
