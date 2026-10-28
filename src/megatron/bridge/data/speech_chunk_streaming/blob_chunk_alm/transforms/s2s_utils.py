# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import random
import re

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.s2s_prompt import (
    ASR_PROMPT,
    CODEC_PROMPT,
    NTP_CODEC_PROMPT,
    SEMANTIC_CODEC_ASR_PROMPT,
    TEXTIN_CODEC_PROMPT,
    TTS_CODEC_PROMPT,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.utils import (
    add_audio_text_in_conversation as _base_add_audio_text_in_conversation,
    combine_audio_and_text_ordered,
    combine_audio_and_text_random,
    convert_to_prompt as _base_convert_to_prompt,
    get_audio_id,
    is_audio_token,
    pad_features,
    process_asr_annotation,
)

_CODEC_PROMPT_FORMATTERS = {
    "codec_chat",
    "codec_pre",
    "tts_codec_pre",
    "semantic_codec_asr_pre",
    "ntp_code_pre",
    "textin_codec_chat",
}

_FAKE_CODEC_PATTERNS = {"interleave_fake", "interleave_short_fake", "simul_fake"}


def convert_to_prompt(
    roles,
    text,
    prompt_formatter,
    fixed_prompt=True,
    user_prompt=None,
    language=None,
    context=None,
):
    if prompt_formatter not in _CODEC_PROMPT_FORMATTERS:
        return _base_convert_to_prompt(
            roles, text, prompt_formatter, fixed_prompt, user_prompt, language, context
        )

    user_role_str, assistant_role_str = roles[0], roles[1]
    pattern_tag = "<audio>\n"

    if prompt_formatter == "codec_chat":
        prompt = CODEC_PROMPT[0] if fixed_prompt else random.choice(CODEC_PROMPT)
    elif prompt_formatter == "codec_pre":
        prompt = ASR_PROMPT[0] if fixed_prompt else random.choice(ASR_PROMPT)
    elif prompt_formatter == "tts_codec_pre":
        prompt = TTS_CODEC_PROMPT[0] + f" {text}" if fixed_prompt else random.choice(TTS_CODEC_PROMPT)
        pattern_tag = "<TEXT>"
    elif prompt_formatter == "semantic_codec_asr_pre":
        prompt = SEMANTIC_CODEC_ASR_PROMPT[0] if fixed_prompt else random.choice(SEMANTIC_CODEC_ASR_PROMPT)
        pattern_tag = "<SEMANTIC_CODEC_ASR>"
    elif prompt_formatter == "ntp_code_pre":
        prompt = NTP_CODEC_PROMPT[0] if fixed_prompt else random.choice(NTP_CODEC_PROMPT)
        pattern_tag = "<TEXT>"
    elif prompt_formatter == "textin_codec_chat":
        prompt = TEXTIN_CODEC_PROMPT[0] + f" {user_prompt}" if fixed_prompt else random.choice(TEXTIN_CODEC_PROMPT)
        pattern_tag = "<TEXT>"
    else:
        raise NotImplementedError(f"Unsupported codec prompt formatter: {prompt_formatter}")

    return [[user_role_str, f"{pattern_tag}{prompt}"], [assistant_role_str, f"{text}"]]


def add_audio_text_in_conversation(
    texts,
    user_prompts,
    languages,
    contexts=None,
    conv_template=None,
    prompt_formatter: str = None,
    fixed_prompt: bool = True,
    instruction_first: bool = False,
    bias_after_audio: bool = False,
):
    if prompt_formatter not in _CODEC_PROMPT_FORMATTERS:
        if contexts is None:
            contexts = [None] * len(texts)
        return _base_add_audio_text_in_conversation(
            texts,
            user_prompts,
            languages,
            contexts,
            conv_template,
            prompt_formatter,
            fixed_prompt,
            instruction_first,
            bias_after_audio,
        )

    for i in range(len(texts)):
        roles = conv_template.get_roles()
        system_role = conv_template.get_system_role()
        context = contexts[i] if contexts is not None else None

        if len(texts[i]) == 1 and prompt_formatter is not None:
            messages = convert_to_prompt(
                roles,
                texts[i][0],
                prompt_formatter,
                fixed_prompt,
                user_prompts[i],
                languages[i],
                context,
            )
        else:
            role_mapping = {"user": roles[0], "assistant": roles[1], "system": system_role}
            messages = [[role_mapping[m["role"]], m["content"]] for m in texts[i]]

        for role, message in messages:
            if conv_template.sep2 in ["<|end|>", "</s>"]:
                message = message.replace("<audio>\n", "<audio>")

            if instruction_first and role == roles[0] and re.search("<audio>", message) is not None:
                instruction = message.replace("<audio>", "")
                if instruction != "":
                    message = f"{instruction}<audio>"

            if role == system_role:
                conv_template.append_system_message(role, message)
            else:
                conv_template.append_message(role, message)

    return conv_template


def _with_eoa(codec, eoa):
    return list(codec) + [eoa]


def _codec_pad_id(pad_audio_id, text_vocab_size):
    return pad_audio_id + text_vocab_size


def _audio_input_ids(codec, text_vocab_size):
    return [token + text_vocab_size for token in codec]


def _interleave_codec(
    codec,
    message_input_ids,
    pad_token_id,
    ignore_id,
    pad_audio_id,
    text_vocab_size,
    pattern_window,
    pattern_ratio,
    stop_after_text=False,
    label_codec_values=None,
):
    i, j = 0, 0
    label_codec = []
    message_label_ids = []
    interleave_input_ids = []
    codec_window_size = pattern_window * pattern_ratio
    text_finished = False

    while (i < len(message_input_ids) or j < len(codec)) and not text_finished:
        text_chunk = message_input_ids[i : i + pattern_window]
        interleave_input_ids.extend(text_chunk)
        message_label_ids.extend(text_chunk)
        label_codec.extend([ignore_id] * len(text_chunk))
        i += len(text_chunk)

        if len(text_chunk) < pattern_window:
            if stop_after_text:
                interleave_input_ids.append(text_vocab_size - 1)
                message_label_ids.append(text_vocab_size - 1)
                label_codec.append(ignore_id)
                text_finished = True
                break

            padding_size = pattern_window - len(text_chunk)
            interleave_input_ids.extend([pad_token_id] * padding_size)
            message_label_ids.extend([pad_token_id] * padding_size)
            label_codec.extend([ignore_id] * padding_size)

        audio_chunk = codec[j : j + codec_window_size]
        label_audio_chunk = (
            label_codec_values[j : j + codec_window_size]
            if label_codec_values is not None
            else audio_chunk
        )
        interleave_input_ids.extend(_audio_input_ids(audio_chunk, text_vocab_size))
        message_label_ids.extend([ignore_id] * len(audio_chunk))
        label_codec.extend(label_audio_chunk)
        j += len(audio_chunk)

        if len(audio_chunk) < codec_window_size:
            padding_size = codec_window_size - len(audio_chunk)
            interleave_input_ids.extend([_codec_pad_id(pad_audio_id, text_vocab_size)] * padding_size)
            message_label_ids.extend([ignore_id] * padding_size)
            padding_label = ignore_id if label_codec_values is not None else pad_audio_id
            label_codec.extend([padding_label] * padding_size)

    if stop_after_text:
        remaining_codec = codec[j:]
        remaining_label_codec = (
            label_codec_values[j:] if label_codec_values is not None else remaining_codec
        )
        interleave_input_ids.extend(_audio_input_ids(remaining_codec, text_vocab_size))
        message_label_ids.extend([ignore_id] * len(remaining_codec))
        label_codec.extend(remaining_label_codec)

    input_codec = copy.deepcopy(label_codec)
    return interleave_input_ids, message_label_ids, [input_codec], [label_codec]


def get_codec_input(
    codecs,
    message_input_ids,
    pad_token_id,
    ignore_id,
    pad_audio_id,
    eoa,
    pattern="simul",
    pattern_window=10,
    pattern_ratio=2,
    text_vocab_size=200064,
    use_eoc_ending=False,
    eoc=None,
    **kwargs,
):
    if pattern in {"interleave", "interleave_short", "codec_only"}:
        assert len(codecs) == 1, "codec layer must be 1"

    if pattern == "interleave":
        return _interleave_codec(
            _with_eoa(codecs[0], eoa),
            message_input_ids,
            pad_token_id,
            ignore_id,
            pad_audio_id,
            text_vocab_size,
            pattern_window,
            pattern_ratio,
        )

    if pattern == "interleave_short":
        return _interleave_codec(
            _with_eoa(codecs[0], eoa),
            message_input_ids,
            pad_token_id,
            ignore_id,
            pad_audio_id,
            text_vocab_size,
            pattern_window,
            pattern_ratio,
            stop_after_text=True,
        )

    if pattern == "simul":
        input_codecs = []
        label_codecs = []
        max_codec_len = len(message_input_ids)
        end_token = eoc if use_eoc_ending and eoc is not None else eoa
        normalized_codecs = [_with_eoa(codec, end_token) for codec in codecs]
        max_codec_len = max(max_codec_len, *(len(codec) for codec in normalized_codecs))

        for codec in normalized_codecs:
            codec = codec + [pad_audio_id] * (max_codec_len - len(codec))
            input_codecs.append(_audio_input_ids(codec, text_vocab_size))
            label_codecs.append(codec)

        if max_codec_len > len(message_input_ids):
            message_label_ids = message_input_ids + [ignore_id] * (max_codec_len - len(message_input_ids))
            message_input_ids = message_input_ids + [pad_token_id] * (max_codec_len - len(message_input_ids))
        else:
            message_label_ids = message_input_ids

        return message_input_ids, message_label_ids, input_codecs, label_codecs

    if pattern == "codec_only":
        codec = _with_eoa(codecs[0], eoa)
        message_input_ids = _audio_input_ids(codec, text_vocab_size)
        message_label_ids = copy.deepcopy(message_input_ids)
        return message_input_ids, message_label_ids, [copy.deepcopy(codec)], [codec]

    raise NotImplementedError(f"Not implemented codec pattern: {pattern}")


def _fake_codec(message_input_ids, ignore_id, pad_audio_id, eoa):
    codec = [pad_audio_id] * len(message_input_ids)
    fake_codec = [ignore_id] * len(message_input_ids)
    if codec:
        codec[-1] = eoa
        fake_codec[-1] = eoa
    return codec, fake_codec


def get_fake_codec_input(
    input_ids,
    labels,
    message_input_ids,
    pad_token_id,
    ignore_id,
    pad_audio_id,
    eoa,
    pattern="interleave_fake",
    pattern_window=10,
    pattern_ratio=2,
    text_vocab_size=200064,
    **kwargs,
):
    if pattern == "interleave_fake":
        codec, fake_codec = _fake_codec(message_input_ids, ignore_id, pad_audio_id, eoa)
        return _interleave_codec(
            codec,
            message_input_ids,
            pad_token_id,
            ignore_id,
            pad_audio_id,
            text_vocab_size,
            pattern_window,
            pattern_ratio,
            label_codec_values=fake_codec,
        )

    if pattern == "interleave_short_fake":
        codec, fake_codec = _fake_codec(message_input_ids, ignore_id, pad_audio_id, eoa)
        return _interleave_codec(
            codec,
            message_input_ids,
            pad_token_id,
            ignore_id,
            pad_audio_id,
            text_vocab_size,
            pattern_window,
            pattern_ratio,
            stop_after_text=True,
            label_codec_values=fake_codec,
        )

    if pattern == "simul_fake":
        codec, fake_codec = _fake_codec(message_input_ids, ignore_id, pad_audio_id, eoa)
        left_pad = [pad_audio_id] * 3
        codec = left_pad + codec + [eoa]
        fake_codec = [ignore_id] * len(left_pad) + fake_codec + [eoa]
        max_len = max(len(message_input_ids), len(codec))
        codec = codec + [pad_audio_id] * (max_len - len(codec))
        fake_codec = fake_codec + [ignore_id] * (max_len - len(fake_codec))
        input_codecs = [copy.deepcopy(input_ids) + _audio_input_ids(codec, text_vocab_size)]
        label_codecs = [copy.deepcopy(labels) + fake_codec]

        if max_len > len(message_input_ids):
            message_label_ids = message_input_ids + [ignore_id] * (max_len - len(message_input_ids))
            message_input_ids = message_input_ids + [pad_token_id] * (max_len - len(message_input_ids))
        else:
            message_label_ids = message_input_ids

        return message_input_ids, message_label_ids, input_codecs, label_codecs

    raise NotImplementedError(f"Not implemented fake codec pattern: {pattern}")


def uses_fake_codec_pattern(codec_cfg):
    return codec_cfg is not None and codec_cfg.get("pattern") in _FAKE_CODEC_PATTERNS
