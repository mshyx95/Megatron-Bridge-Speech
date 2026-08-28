# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Streaming EOC padding for simultaneous text-audio generation.

When streaming, the model must learn to produce audio *chunks* delimited by
EOC tokens.  At a randomly chosen word boundary the text stream is padded
(with ``text_pad_id``) from the word boundary to the corresponding audio
boundary, an EOC is inserted in every codec layer at that audio position,
and the remaining text tokens are shifted right.  Both text and codec
sequences grow by exactly 1 token (the EOC / its text-side pad), while the
text-side gap is created by *replacing* existing trailing pads — the shape
invariant is preserved.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_alignment(
    alignment: List[Dict],
) -> List[Dict[str, Union[str, float]]]:
    """Return ``[{"word": str, "start": float, "end": float}]`` in seconds."""
    normalized: List[Dict[str, Union[str, float]]] = []
    for entry in alignment:
        word = entry.get("word", entry.get("text", ""))
        if "st" in entry and "et" in entry:
            start = float(entry["st"]) / 1000.0
            end = float(entry["et"]) / 1000.0
        elif "start" in entry and "end" in entry:
            start = float(entry["start"])
            end = float(entry["end"])
        else:
            start = float(entry.get("start_time", 0))
            end = float(entry.get("end_time", 0))
        normalized.append({"word": word, "start": start, "end": end})
    return normalized


def _filter_silence(
    words: List[Dict[str, Union[str, float]]],
) -> List[Dict[str, Union[str, float]]]:
    """Remove silence / filler entries."""
    silence_tokens = {"SIL", "sil", "<sil>", "[SIL]", "<SIL>", ""}
    return [
        w for w in words
        if w["word"].strip() not in silence_tokens
        and w["word"].strip().lower() != "sil"
    ]


def _count_real_text_tokens(message_label_ids: list, ignore_id: int = -100) -> int:
    """Count real text tokens using label_ids (-100 = padding region)."""
    # In simul pattern: labels = [real_tokens..., -100, -100, ...]
    # When text >= codec: labels = all real (no -100 at all)
    num_real = len(message_label_ids)
    for i in range(len(message_label_ids)):
        if message_label_ids[i] == ignore_id:
            num_real = i
            break
    return num_real


def _find_text_split_idx(
    message_input_ids: list,
    words: List[Dict],
    split_word_idx: int,
    tokenizer,
    num_real: int,
) -> int:
    """Find the token index after the last BPE token of word ``split_word_idx``."""
    cumulative = 0
    for widx, w in enumerate(words):
        word_text = w["word"]
        word_tokens = tokenizer.encode(" " + word_text, add_special_tokens=False)
        if widx == 0:
            word_tokens_no_space = tokenizer.encode(word_text, add_special_tokens=False)
            if cumulative + len(word_tokens_no_space) <= num_real:
                word_tokens = word_tokens_no_space

        cumulative += len(word_tokens)

        if widx == split_word_idx:
            return min(cumulative, num_real)

    return min(cumulative, num_real)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def insert_streaming_eoc_pad(
    message_input_ids: list,
    message_label_ids: list,
    input_codecs: list,
    label_codecs: list,
    alignment: list,
    tokenizer=None,
    codec_frame_rate: float = 25.0,
    num_splits: int = 1,
    eoc_token_id: int = 206685,
    eoc_label_id: int = 6621,
    text_pad_id: int = 199999,
    label_ignore_id: int = -100,
    pad_audio_id: int = 206687,
    pad_token_id: int = 0,
    seed: Optional[int] = None,
) -> Tuple[list, list, list, list]:
    """Insert EOC at a random word boundary with proper text padding.

    The simul-pattern output has::

        input_ids:  [T1 .. Tm  PAD PAD .. PAD]   (length N)
        codecs:     [A1 .. A(N-1) EOA]            (length N)

    After splitting at word *k* (text ends at ``text_split``, audio at
    ``audio_split``)::

        input_ids:  [T1..Tk  199999..199999  199999  T(k+1)..Tm  PAD..PAD]
                              ← gap pads →    EOC     ← shifted →
        codecs:     [A1 .. A(audio_split)  EOC  A(audio_split+1) .. EOA]

    Both grow by exactly 1 token.  The gap pads are carved from the
    original trailing pads (no net increase except the EOC insertion).
    """
    try:
        # 0. Validation ------------------------------------------------
        if alignment is None or len(alignment) == 0:
            return message_input_ids, message_label_ids, input_codecs, label_codecs
        if len(input_codecs) == 0 or len(input_codecs[0]) == 0:
            return message_input_ids, message_label_ids, input_codecs, label_codecs

        codec_len = len(input_codecs[0])
        for li, layer in enumerate(input_codecs):
            if len(layer) != codec_len:
                logger.warning("input_codecs layer %d length mismatch; skip EOC.", li)
                return message_input_ids, message_label_ids, input_codecs, label_codecs
        if len(message_input_ids) != codec_len:
            logger.warning("input_ids/codec length mismatch; skip EOC.")
            return message_input_ids, message_label_ids, input_codecs, label_codecs

        # 1. Normalize & filter alignment ------------------------------
        words = _normalize_alignment(alignment)
        words = _filter_silence(words)
        if len(words) < 2:
            return message_input_ids, message_label_ids, input_codecs, label_codecs

        # 2. Random split word -----------------------------------------
        rng = random.Random(seed)
        k = rng.randint(0, len(words) - 2)

        # 3. Compute audio_split_idx -----------------------------------
        split_time = words[k]["end"]
        audio_split_idx = int(split_time * codec_frame_rate)
        # Clamp to next word's start to protect next word's audio
        next_start_idx = int(words[k + 1]["start"] * codec_frame_rate)
        if 0 < next_start_idx < audio_split_idx:
            audio_split_idx = next_start_idx

        # Find actual audio end: skip leading pad_audio, then find where real audio ends
        eoa_in_input = pad_audio_id + 1
        # Skip leading pad tokens (some pipelines prepend pad_audio)
        audio_start = 0
        for idx in range(codec_len):
            tok = input_codecs[0][idx]
            if tok != pad_audio_id and tok != eoa_in_input:
                audio_start = idx
                break
        # Find end of real audio (EOA or pad_audio AFTER the real audio starts)
        real_audio_end = codec_len
        for idx in range(audio_start, codec_len):
            tok = input_codecs[0][idx]
            if tok == eoa_in_input or tok == pad_audio_id:
                real_audio_end = idx
                break
        if real_audio_end - audio_start < 3:
            logger.warning(
                "Not enough real audio (%d tokens from %d to %d); skip EOC.",
                real_audio_end - audio_start, audio_start, real_audio_end,
            )
            return message_input_ids, message_label_ids, input_codecs, label_codecs

        # Clamp audio_split to within real audio region
        audio_split_idx = max(audio_start + 1, min(audio_split_idx, real_audio_end - 1))
        audio_split_idx = min(audio_split_idx, codec_len - 2)

        # 4. Compute text_split_idx ------------------------------------
        num_real = _count_real_text_tokens(message_label_ids, label_ignore_id)
        if tokenizer is not None:
            text_split_idx = _find_text_split_idx(
                message_input_ids, words, k, tokenizer, num_real,
            )
        else:
            word_frac = (k + 1) / len(words)
            text_split_idx = max(1, min(int(num_real * word_frac), num_real))

        # text_split must be <= audio_split (text is shorter than audio)
        text_split_idx = min(text_split_idx, audio_split_idx)

        # 5. Sanity: need room for the gap between text and audio split
        #    The gap is carved from existing trailing pads.
        gap_size = audio_split_idx - text_split_idx
        trailing_pads = codec_len - num_real
        if trailing_pads < 0:
            trailing_pads = 0
        if gap_size > trailing_pads:
            logger.warning(
                "Not enough trailing pads (%d) for gap (%d); skip EOC. "
                "codec_len=%d, num_real=%d, text_split=%d, audio_split=%d",
                trailing_pads, gap_size, codec_len, num_real,
                text_split_idx, audio_split_idx,
            )
            return message_input_ids, message_label_ids, input_codecs, label_codecs

        # 6. Build new text sequences ----------------------------------
        #    Rearrange: [first_chunk_text | gap_pads | second_chunk_text | remaining_pads]
        #    Then insert 1 PAD at audio_split_idx for EOC (grows by 1).
        first_text = list(message_input_ids[:text_split_idx])
        second_text = list(message_input_ids[text_split_idx:num_real])  # remaining real text
        gap_pads_input = [text_pad_id] * gap_size
        remaining_pads_input = [message_input_ids[i] for i in range(num_real, codec_len)]
        # Trim remaining pads by gap_size (gap pads replace trailing pads)
        remaining_pads_input = remaining_pads_input[gap_size:]

        rearranged_input = first_text + gap_pads_input + second_text + remaining_pads_input
        assert len(rearranged_input) == codec_len, (
            f"Rearranged length {len(rearranged_input)} != {codec_len}"
        )

        # Same for labels
        first_lbl = list(message_label_ids[:text_split_idx])
        second_lbl = list(message_label_ids[text_split_idx:num_real])
        gap_pads_label = [label_ignore_id] * gap_size
        remaining_pads_label = list(message_label_ids[num_real:codec_len])
        remaining_pads_label = remaining_pads_label[gap_size:]

        rearranged_label = first_lbl + gap_pads_label + second_lbl + remaining_pads_label
        assert len(rearranged_label) == codec_len

        # 7. Insert EOC at audio_split_idx (grows by 1) ---------------
        new_input_ids = (
            rearranged_input[:audio_split_idx]
            + [text_pad_id]
            + rearranged_input[audio_split_idx:]
        )
        new_label_ids = (
            rearranged_label[:audio_split_idx]
            + [label_ignore_id]
            + rearranged_label[audio_split_idx:]
        )
        new_input_codecs = []
        new_label_codecs = []
        for layer in input_codecs:
            new_input_codecs.append(
                list(layer[:audio_split_idx]) + [eoc_token_id] + list(layer[audio_split_idx:])
            )
        for layer in label_codecs:
            new_label_codecs.append(
                list(layer[:audio_split_idx]) + [eoc_label_id] + list(layer[audio_split_idx:])
            )

        # 8. Shape invariant -------------------------------------------
        expected = codec_len + 1
        assert len(new_input_ids) == expected, f"input_ids {len(new_input_ids)} != {expected}"
        assert len(new_label_ids) == expected
        for li, layer in enumerate(new_input_codecs):
            assert len(layer) == expected, f"input_codecs[{li}] {len(layer)} != {expected}"
        for li, layer in enumerate(new_label_codecs):
            assert len(layer) == expected

        return new_input_ids, new_label_ids, new_input_codecs, new_label_codecs

    except Exception as exc:
        logger.warning("insert_streaming_eoc_pad failed (%s); returning originals.", exc)
        return message_input_ids, message_label_ids, input_codecs, label_codecs
