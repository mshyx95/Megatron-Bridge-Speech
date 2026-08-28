# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import re

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.conversation_utils import (
    Conversation,
    SeparatorStyle,
    _audio_tok_in_message,
    _audio_tok_match_message,
    conv_templates as base_conv_templates,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.s2s_utils import (
    get_codec_input,
    get_fake_codec_input,
    uses_fake_codec_pattern,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.streaming_pad import insert_streaming_eoc_pad

IGNORE_INDEX = -100


class S2SConversation(Conversation):
    def copy(self):
        copied = S2SConversation(
            system=self.system,
            system_role=self.system_role,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            system_sep=self.system_sep,
            version=self.version,
            skip_next=self.skip_next,
            max_num_audios=self.max_num_audios,
            audio_token_length=self.audio_token_length,
            role_suffix=self.role_suffix,
            role_prefix=self.role_prefix,
        )
        copied.codec_cfg = getattr(self, "codec_cfg", None)
        return copied

    def get_prompt(
        self,
        is_train=False,
        audio_input_id=0,
        only_add_first_audio_occurrence=True,
        audio_start_token="",
        audio_end_token="",
        num_audio_tokens=None,
        codecs_list=None,
        alignments_list=None,
        use_eoc_endings_list=None,
    ):
        codec_cfg = getattr(self, "codec_cfg", None)
        has_fake_codec = uses_fake_codec_pattern(codec_cfg)
        has_codec_request = codecs_list is not None or has_fake_codec
        if not is_train or not has_codec_request:
            return super().get_prompt(
                is_train=is_train,
                audio_input_id=audio_input_id,
                only_add_first_audio_occurrence=only_add_first_audio_occurrence,
                audio_start_token=audio_start_token,
                audio_end_token=audio_end_token,
                num_audio_tokens=num_audio_tokens or [],
            )

        assert self.sep_style == SeparatorStyle.TWO, "Currently only support LLAVA v1.5"

        codec_cfg = codec_cfg or {}
        num_audio_tokens = num_audio_tokens or []
        codecs_list = codecs_list or []
        alignments_list = alignments_list or []
        use_eoc_endings_list = use_eoc_endings_list or []

        input_ids = []
        labels = []
        input_codecs = []
        label_codecs = []
        audio_positions = []
        audio_idx = 0
        codec_idx = 0

        codec_params = self._codec_params(codec_cfg)
        codec_pad_input_id = codec_params["pad_audio_id"] + codec_params["text_vocab_size"]

        def append_plain_codec_tokens(token_ids, label_ids):
            if not input_codecs or not token_ids:
                return
            for layer in input_codecs:
                layer.extend(token_ids)
            for layer in label_codecs:
                layer.extend(label_ids)

        def append_tokens(token_ids, label_ids):
            input_ids.extend(token_ids)
            labels.extend(label_ids)
            append_plain_codec_tokens(token_ids, label_ids)

        def append_codec_segment(segment_input_ids, segment_label_ids, segment_input_codecs, segment_label_codecs):
            if not input_codecs:
                for layer in segment_input_codecs:
                    input_codecs.append(list(input_ids) + list(layer))
                for layer in segment_label_codecs:
                    label_codecs.append(list(labels) + list(layer))
            else:
                assert len(input_codecs) == len(segment_input_codecs), (
                    f"codec layer mismatch: existing {len(input_codecs)}, "
                    f"new {len(segment_input_codecs)}"
                )
                for target, layer in zip(input_codecs, segment_input_codecs):
                    target.extend(layer)
                for target, layer in zip(label_codecs, segment_label_codecs):
                    target.extend(layer)

            input_ids.extend(segment_input_ids)
            labels.extend(segment_label_ids)

        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            raise NotImplementedError

        seps = [self.sep, self.sep2]
        if len(self.system):
            system_sep = self.system_sep if self.system_sep is not None else seps[0]
            ret = self.system + system_sep
            has_system_prompt = True
        else:
            ret = ""
            has_system_prompt = False

        if len(ret):
            sys_input_ids = self.tokenizer.encode(ret, add_special_tokens=False)
            append_tokens(sys_input_ids, [IGNORE_INDEX] * len(sys_input_ids))

        first_audio_token_processed = False
        for i, (role, message) in enumerate(messages):
            if i > 0 and role == self.roles[0]:
                newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
                append_tokens(newline_ids, [IGNORE_INDEX] * len(newline_ids))
                ret += "\n"

            used_role = role
            if has_system_prompt and role.startswith(self.role_prefix):
                used_role = role.replace(self.role_prefix, "", 1)

            if not message:
                continue

            if type(message) is tuple:
                message, _, _ = message
            if type(message) is list:
                message = random.choice(message)

            ret += used_role + self.role_suffix + message + seps[i % 2]
            prefix = used_role + self.role_suffix
            prefix_input_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            append_tokens(prefix_input_ids, [IGNORE_INDEX] * len(prefix_input_ids))

            audio_token = _audio_tok_in_message(message)
            if audio_token and audio_token in message and first_audio_token_processed and only_add_first_audio_occurrence:
                message = message.replace(audio_token, "")

            if audio_token and _audio_tok_in_message(message):
                audio_idx, audio_input_id, first_audio_token_processed = self._append_audio_message(
                    message,
                    role,
                    seps[i % 2],
                    num_audio_tokens,
                    audio_idx,
                    audio_input_id,
                    audio_start_token,
                    audio_end_token,
                    audio_positions,
                    input_ids,
                    append_tokens,
                )
                continue

            message_input_ids = self.tokenizer.encode(message + seps[i % 2], add_special_tokens=False)
            if role not in [self.roles[1]]:
                if role in [self.roles[0], self.system_role]:
                    append_tokens(message_input_ids, [IGNORE_INDEX] * len(message_input_ids))
                else:
                    append_tokens(message_input_ids, message_input_ids)
                continue

            codec = codecs_list[codec_idx] if codec_idx < len(codecs_list) else None
            alignment = alignments_list[codec_idx] if codec_idx < len(alignments_list) else None
            use_eoc_ending = (
                use_eoc_endings_list[codec_idx]
                if codec_idx < len(use_eoc_endings_list)
                else False
            )
            codec_idx += 1

            segment = self._build_codec_segment(
                codec,
                message_input_ids,
                codec_cfg,
                codec_params,
                use_eoc_ending,
                has_fake_codec,
            )
            if segment is None:
                append_tokens(message_input_ids, message_input_ids)
                continue

            (
                segment_input_ids,
                segment_label_ids,
                segment_input_codecs,
                segment_label_codecs,
            ) = segment

            if alignment is not None and use_eoc_ending:
                segment_input_ids, segment_label_ids, segment_input_codecs, segment_label_codecs = (
                    insert_streaming_eoc_pad(
                        segment_input_ids,
                        segment_label_ids,
                        segment_input_codecs,
                        segment_label_codecs,
                        alignment,
                        tokenizer=self.tokenizer,
                        eoc_token_id=codec_params["eoc"] + codec_params["text_vocab_size"],
                        eoc_label_id=codec_params["eoc"],
                        text_pad_id=codec_params["text_pad_id"],
                        pad_audio_id=codec_pad_input_id,
                    )
                )

            append_codec_segment(
                segment_input_ids,
                segment_label_ids,
                segment_input_codecs,
                segment_label_codecs,
            )

        assert len(input_ids) == len(labels), (
            f"all input_ids {len(input_ids)} and labels {len(labels)} length not match"
        )

        if input_codecs:
            for layer in input_codecs:
                assert len(layer) == len(input_ids), (
                    f"input codec length {len(layer)} does not match input ids {len(input_ids)}"
                )
            for layer in label_codecs:
                assert len(layer) == len(input_ids), (
                    f"label codec length {len(layer)} does not match input ids {len(input_ids)}"
                )
            return input_codecs, input_ids, labels, label_codecs

        return ret, input_ids, labels, audio_positions

    def _codec_params(self, codec_cfg):
        codec_cfg = codec_cfg or {}
        audio_vocab_size = codec_cfg.get("audio_vocab_size", 6624)
        text_vocab_size = codec_cfg.get("text_vocab_size", 200064)
        eoa = codec_cfg.get("eoa", codec_cfg.get("eoa_token_id", audio_vocab_size - 1))
        return {
            "audio_vocab_size": audio_vocab_size,
            "text_vocab_size": text_vocab_size,
            "pad_audio_id": codec_cfg.get("pad_audio_id", audio_vocab_size - 2),
            "eoa": eoa,
            "eoc": codec_cfg.get("eoc", codec_cfg.get("eoc_token_id", eoa - 3)),
            "text_pad_id": codec_cfg.get("text_pad_id", text_vocab_size - 65),
        }

    def _append_audio_message(
        self,
        message,
        role,
        sep,
        num_audio_tokens,
        audio_idx,
        audio_input_id,
        audio_start_token,
        audio_end_token,
        audio_positions,
        input_ids,
        append_tokens,
    ):
        audio_token = _audio_tok_in_message(message)
        split_message = re.split(rf"({audio_token})", message)
        split_message = [part for part in split_message if part]

        for split_idx, part in enumerate(split_message):
            if part == audio_token or _audio_tok_match_message(part):
                if len(audio_start_token):
                    append_tokens(
                        [self.tokenizer.convert_tokens_to_ids(audio_start_token)],
                        [IGNORE_INDEX],
                    )

                audio_token_length = num_audio_tokens[audio_idx]
                audio_positions.append(len(input_ids))
                append_tokens([audio_input_id] * audio_token_length, [IGNORE_INDEX] * audio_token_length)

                if len(audio_end_token):
                    append_tokens(
                        [self.tokenizer.convert_tokens_to_ids(audio_end_token)],
                        [IGNORE_INDEX],
                    )

                audio_input_id -= 1
                audio_idx += 1
                if split_idx == len(split_message) - 1 and len(sep) > 0:
                    sep_ids = self.tokenizer.encode(sep, add_special_tokens=False)
                    sep_labels = [IGNORE_INDEX] * len(sep_ids)
                    if role not in [self.roles[0], self.system_role]:
                        sep_labels = sep_ids
                    append_tokens(sep_ids, sep_labels)
            else:
                if split_idx == len(split_message) - 1:
                    part = part + sep
                part_ids = self.tokenizer.encode(part, add_special_tokens=False)
                part_labels = [IGNORE_INDEX] * len(part_ids)
                if role not in [self.roles[0], self.system_role]:
                    part_labels = part_ids
                append_tokens(part_ids, part_labels)

        return audio_idx, audio_input_id, True

    def _build_codec_segment(
        self,
        codec,
        message_input_ids,
        codec_cfg,
        codec_params,
        use_eoc_ending,
        has_fake_codec,
    ):
        if codec is None and not has_fake_codec:
            return None

        common_kwargs = dict(
            pad_token_id=self.tokenizer.pad_token_id,
            ignore_id=IGNORE_INDEX,
            pad_audio_id=codec_params["pad_audio_id"],
            eoa=codec_params["eoa"],
            eoc=codec_params["eoc"],
            pattern=codec_cfg.get("pattern", "simul"),
            pattern_window=codec_cfg.get("pattern_window", 10),
            pattern_ratio=codec_cfg.get("pattern_ratio", 2),
            text_vocab_size=codec_params["text_vocab_size"],
        )

        if codec is not None:
            return get_codec_input(
                codec,
                message_input_ids,
                use_eoc_ending=use_eoc_ending,
                **common_kwargs,
            )

        return get_fake_codec_input([], [], message_input_ids, **common_kwargs)


def _to_s2s_conversation(conversation):
    return S2SConversation(
        system=conversation.system,
        system_role=conversation.system_role,
        roles=conversation.roles,
        messages=[[x, y] for x, y in conversation.messages],
        offset=conversation.offset,
        sep_style=conversation.sep_style,
        sep=conversation.sep,
        sep2=conversation.sep2,
        system_sep=conversation.system_sep,
        version=conversation.version,
        skip_next=conversation.skip_next,
        max_num_audios=conversation.max_num_audios,
        audio_token_length=conversation.audio_token_length,
        role_suffix=conversation.role_suffix,
        role_prefix=conversation.role_prefix,
    )


conv_templates = {
    name: _to_s2s_conversation(conversation)
    for name, conversation in base_conv_templates.items()
}
conv_templates["phi_o_long_s2s"] = conv_templates["phi_o_long"].copy()