# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

import torch
from transformers import BatchEncoding

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.audio_text_conversation import (
    AudioTextConversationTransform,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.s2s_utils import (
    add_audio_text_in_conversation,
    uses_fake_codec_pattern,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.s2s_conversation_utils import conv_templates
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.conversation_utils import (
    _audio_tok_in_message,
    _audio_tok_match_message,
)
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger

logger = get_logger(__name__)

IGNORE_IDX = -100


def _count_audio_placeholders(messages) -> int:
    """Count audio-token occurrences exactly as ``S2SConversation.get_prompt`` /
    ``_append_audio_message`` would consume them (one ``num_audio_tokens`` index per
    occurrence). Used to validate that a (packed) sample's messages carry exactly as
    many ``<audio>`` placeholders as it has ``audio_embed_sizes`` entries.
    """
    import re

    total = 0
    for message in messages:
        text = message[1] if isinstance(message, (list, tuple)) and len(message) > 1 else message
        if not isinstance(text, str):
            continue
        tok = _audio_tok_in_message(text)
        if not tok:
            continue
        for part in re.split(rf"({tok})", text):
            if not part:
                continue
            if part == tok or _audio_tok_match_message(part):
                total += 1
    return total


def _validate_audio_token_alignment(messages, num_audio_tokens, where: str) -> None:
    """Raise ValueError if the message audio-placeholder count != num_audio_tokens length.

    The forward-tokenize path rebuilds the sequence inside ``model.forward`` (a rank
    crash there aborts the whole distributed job). A rare per-sample data defect where the
    conversation text carries more/fewer ``<audio>`` placeholders than the sample has audios
    would trip an ``IndexError`` (``num_audio_tokens[audio_idx]``) mid-forward, or violate the
    audio-embed assert ``audio_embed_sizes.sum() == len(placeholder positions)``. Validating
    here (in the dataloader worker) lets the caller's try/except DROP the sample cleanly,
    keeping messages / audio_embed_sizes / audios mutually consistent.
    """
    placeholders = _count_audio_placeholders(messages)
    if placeholders != len(num_audio_tokens):
        raise ValueError(
            f"forward_tokenize audio-token mismatch in {where}: messages carry {placeholders} "
            f"<audio> placeholder(s) but audio_embed_sizes has {len(num_audio_tokens)} entry(ies). "
            f"Dropping this sample to avoid a mid-forward IndexError / audio-embed assert."
        )


class SpeechOutAudioTextConversationTransform(AudioTextConversationTransform):
    def __init__(self, config, **kwargs):
        s2s_conv_template = kwargs.get("conv_template")
        base_kwargs = dict(kwargs)
        if s2s_conv_template and s2s_conv_template not in conv_templates:
            raise KeyError(s2s_conv_template)
        if s2s_conv_template and s2s_conv_template.endswith("_s2s"):
            base_kwargs["conv_template"] = s2s_conv_template[: -len("_s2s")]
        super().__init__(config, **base_kwargs)
        self.conv_template = conv_templates[s2s_conv_template].copy()
        self.s2s_conv_template_name = s2s_conv_template

    def _new_conversation(self, results):
        conv_template = self.conv_template.copy()
        conv_template.tokenizer = results["tokenizer"]
        conv_template.codec_cfg = results.get("codec_cfg")
        return conv_template

    def _has_codec_output(self, codec, conv_template):
        return codec is not None or uses_fake_codec_pattern(getattr(conv_template, "codec_cfg", None))

    def _sanitize_label_codecs(self, codec_cfg, labels, label_codecs, where):
        text_pred_mask = (labels != IGNORE_IDX).unsqueeze(1).expand_as(label_codecs)
        label_codecs = label_codecs.masked_fill(text_pred_mask, IGNORE_IDX)

        audio_vocab_size = (codec_cfg or {}).get("audio_vocab_size", 6624)
        if audio_vocab_size:
            invalid_mask = (label_codecs != IGNORE_IDX) & (
                (label_codecs < 0) | (label_codecs >= audio_vocab_size)
            )
            n_invalid = int(invalid_mask.sum())
            if n_invalid > 0:
                bad_vals = torch.unique(label_codecs[invalid_mask]).tolist()
                logger.warning(
                    "[s2s_transform] masked %d out-of-range audio label token(s) to IGNORE "
                    "in %s (audio_vocab_size=%d, valid=[0,%d)); offending values=%s. "
                    "label_codecs shape=%s.",
                    n_invalid,
                    where,
                    audio_vocab_size,
                    audio_vocab_size,
                    bad_vals[:32],
                    tuple(label_codecs.shape),
                )
                label_codecs[invalid_mask] = IGNORE_IDX
        return label_codecs

    def _pad_codec_batch(self, results, input_codecs, label_codecs, max_input_id_len, max_label_len, labels=None):
        codec_cfg = results.get("codec_cfg", {})
        audio_pad_id = codec_cfg.get("audio_vocab_size", 6624) + codec_cfg.get("text_vocab_size", 200064) - 2
        codec_layer = len(input_codecs[0])
        for i in range(len(input_codecs)):
            for j in range(codec_layer):
                input_codecs[i][j] += [audio_pad_id] * (max_input_id_len - len(input_codecs[i][j]))
                label_codecs[i][j] += [IGNORE_IDX] * (max_label_len - len(label_codecs[i][j]))

        label_codecs_t = torch.tensor(label_codecs, dtype=torch.long)
        if labels is not None:
            label_codecs_t = self._sanitize_label_codecs(
                codec_cfg,
                labels,
                label_codecs_t,
                "transform_concat_to_batch",
            )

        results.update(
            {
                "input_codecs": BatchEncoding(
                    dict(
                        input_ids=torch.tensor(input_codecs, dtype=torch.long),
                        attention_mask=torch.ones(len(input_codecs), dtype=torch.long),
                    )
                ),
                "label_codecs": BatchEncoding(
                    dict(input_ids=label_codecs_t)
                ),
            }
        )

    def _update_text_outputs(self, results, audios, input_ids, labels, audio_embed_sizes_1d, is_text_sample):
        results.update(
            {
                "audios": audios,
                "inputs": BatchEncoding(
                    dict(
                        input_ids=input_ids,
                        attention_mask=torch.ones(len(input_ids), dtype=torch.long),
                    )
                ),
                "labels": BatchEncoding(dict(input_ids=labels)),
                "audio_embed_sizes": torch.tensor(audio_embed_sizes_1d),
                "audio_frames": torch.tensor(results["audio_frames"]),
                "is_text_sample": is_text_sample,
            }
        )
        return results

    def transform_concat_to_batch(self, results: Dict) -> Dict:
        assert "texts" in results, f"No texts/image_text_interleaved found in the annotations {results['annotations']}."
        assert "tokenizer" in results, "No tokenizer found in the results."

        audios = results["audios"]
        texts = results["texts"]
        audio_embed_sizes = results["audio_embed_sizes"]
        user_prompts = results["user_prompts"]
        languages = results["languages"]
        codecs = results.get("codecs", [None] * len(texts))
        contexts = results.get("contexts", [None] * len(texts))
        alignments = results.get("alignments", [None] * len(texts))
        use_eoc_endings = results.get("use_eoc_endings", [False] * len(texts))
        tokenizer = results["tokenizer"]

        input_ids = []
        labels = []
        input_codecs = []
        label_codecs = []
        audio_embed_sizes_1d = []
        is_text_sample = False

        for text, embed_size, user_prompt, language, codec, context, alignment, use_eoc_ending in zip(
            texts,
            audio_embed_sizes,
            user_prompts,
            languages,
            codecs,
            contexts,
            alignments,
            use_eoc_endings,
        ):
            conv_template = self._new_conversation(results)
            conv_template = add_audio_text_in_conversation(
                [text],
                [user_prompt],
                [language],
                [context],
                conv_template,
                self.prompt_formatter,
                self.fixed_prompt,
                bias_after_audio=self.bias_after_audio,
            )
            ret, input_id, label, codec_labels = conv_template.get_prompt(
                is_train=True,
                audio_input_id=-1,
                only_add_first_audio_occurrence=False,
                num_audio_tokens=embed_size,
                codecs_list=[codec],
                alignments_list=[alignment],
                use_eoc_endings_list=[use_eoc_ending],
            )

            if self._has_codec_output(codec, conv_template):
                input_codecs.append(ret)
                label_codecs.append(codec_labels)

            audio_embed_sizes_1d.extend(embed_size)
            input_ids.append(input_id)
            labels.append(label)

            if sum(embed_size) == 0:
                is_text_sample = True

        max_input_id_len = max(len(input_id) for input_id in input_ids)
        max_label_len = max(len(label) for label in labels)
        input_ids = [
            input_id + [tokenizer.pad_token_id] * (max_input_id_len - len(input_id))
            for input_id in input_ids
        ]
        labels = [
            label + [IGNORE_IDX] * (max_label_len - len(label))
            for label in labels
        ]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        if input_codecs:
            self._pad_codec_batch(results, input_codecs, label_codecs, max_input_id_len, max_label_len, labels=labels)

        if self.shift_labels:
            labels = labels[1:]

        return self._update_text_outputs(
            results,
            audios,
            input_ids,
            labels,
            audio_embed_sizes_1d,
            is_text_sample,
        )

    def transform_concat_to_context_length(self, results: Dict) -> Dict:
        assert "texts" in results, f"No texts/image_text_interleaved found in the annotations {results['annotations']}."
        assert "tokenizer" in results, "No tokenizer found in the results."

        audios = results["audios"]
        texts = results["texts"]
        user_prompts = results["user_prompts"]
        languages = results["languages"]
        codecs = results.get("codecs", [None] * len(texts))
        contexts = results.get("contexts", [None] * len(texts))
        alignments = results.get("alignments", [None] * len(texts))
        use_eoc_endings = results.get("use_eoc_endings", [False] * len(texts))

        audio_embed_sizes_1d = []
        is_text_sample = False
        for embed_size in results["audio_embed_sizes"]:
            if sum(embed_size) == 0:
                is_text_sample = True
            audio_embed_sizes_1d.extend(embed_size)

        conv_template = self._new_conversation(results)
        conv_template = add_audio_text_in_conversation(
            texts,
            user_prompts,
            languages,
            contexts,
            conv_template,
            self.prompt_formatter,
            self.fixed_prompt,
            self.instruction_first,
            self.bias_after_audio,
        )
        ret, input_ids, labels, codec_labels = conv_template.get_prompt(
            is_train=True,
            audio_input_id=-1,
            only_add_first_audio_occurrence=False,
            num_audio_tokens=audio_embed_sizes_1d,
            codecs_list=codecs,
            alignments_list=alignments,
            use_eoc_endings_list=use_eoc_endings,
        )

        if self.pad_to_context_length:
            input_ids = input_ids[: self.context_length]
            labels = labels[: self.context_length]
            if self._has_codec_output(codecs[0], conv_template):
                ret = [layer[: self.context_length] for layer in ret]
                codec_labels = [layer[: self.context_length] for layer in codec_labels]

        if self._has_codec_output(codecs[0], conv_template):
            label_codecs_t = torch.tensor(codec_labels, dtype=torch.long).unsqueeze(0)
            label_codecs_t = self._sanitize_label_codecs(
                results.get("codec_cfg", {}),
                torch.tensor(labels, dtype=torch.long).unsqueeze(0),
                label_codecs_t,
                "transform_concat_to_context_length",
            )
            results.update(
                {
                    "input_codecs": BatchEncoding(
                        dict(
                            input_ids=torch.tensor(ret, dtype=torch.long).unsqueeze(0),
                            attention_mask=torch.ones(len(input_ids), dtype=torch.long),
                        )
                    ),
                    "label_codecs": BatchEncoding(
                        dict(input_ids=label_codecs_t)
                    ),
                }
            )

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        if self.shift_labels:
            labels = labels[1:]
        labels = torch.tensor(labels, dtype=torch.long)

        return self._update_text_outputs(
            results,
            audios,
            input_ids,
            labels,
            audio_embed_sizes_1d,
            is_text_sample,
        )


class SpeechOutRawAudioConversationTransform(SpeechOutAudioTextConversationTransform):
    """Transform for the raw-audio forward path.

    Resolves the conversation messages ONCE (concrete strings, random prompts
    already chosen) and emits a per-sample ``seq_plan`` plus ``answer_audio``
    passthrough — it does NOT call ``get_prompt`` with codecs and does NOT build
    or pad any codec tensors. Question-audio / text handling matches the parent.
    """

    def transform(self, results: Dict) -> Dict:
        if self.batch_format == "samples_to_batch":
            return self.transform_concat_to_batch(results)
        if self.batch_format == "samples_to_sequence":
            return self.transform_concat_to_context_length(results)
        raise NotImplementedError(
            f"SpeechOutRawAudioConversationTransform: unknown batch_format {self.batch_format!r}."
        )

    def transform_concat_to_batch(self, results: Dict) -> Dict:
        assert "texts" in results, f"No texts found in the annotations {results.get('annotations')}."
        assert "tokenizer" in results, "No tokenizer found in the results."

        audios = results["audios"]
        texts = results["texts"]
        audio_embed_sizes = results["audio_embed_sizes"]
        user_prompts = results["user_prompts"]
        languages = results["languages"]
        contexts = results.get("contexts", [None] * len(texts))
        alignments = results.get("alignments", [None] * len(texts))
        use_eoc_endings = results.get("use_eoc_endings", [False] * len(texts))
        answer_audio = results.get("answer_audio", [None] * len(texts))
        infos = results.get("infos", [None] * len(texts))

        seq_plan = []
        audio_embed_sizes_1d = []
        is_text_sample = False

        for text, embed_size, user_prompt, language, context, alignment, use_eoc_ending, info in zip(
            texts,
            audio_embed_sizes,
            user_prompts,
            languages,
            contexts,
            alignments,
            use_eoc_endings,
            infos,
        ):
            conv_template = self._new_conversation(results)
            conv_template = add_audio_text_in_conversation(
                [text],
                [user_prompt],
                [language],
                [context],
                conv_template,
                self.prompt_formatter,
                self.fixed_prompt,
            )
            entry_messages = [list(message) for message in conv_template.messages]
            _validate_audio_token_alignment(entry_messages, list(embed_size), "transform_concat_to_batch")
            seq_plan.append(
                {
                    "conv_template": self.s2s_conv_template_name,
                    "messages": entry_messages,
                    "system": conv_template.system,
                    "num_audio_tokens": list(embed_size),
                    "alignment": alignment,
                    "use_eoc_ending": use_eoc_ending,
                    "codec_prefix_sample": {"info": info} if info is not None else {},
                    "batch_format": "samples_to_batch",
                    "num_samples": 1,
                }
            )

            audio_embed_sizes_1d.extend(embed_size)
            if sum(embed_size) == 0:
                is_text_sample = True

        results.update(
            {
                "audios": audios,
                "seq_plan": seq_plan,
                "answer_audio": list(answer_audio),
                "audio_embed_sizes": torch.tensor(audio_embed_sizes_1d),
                "audio_frames": torch.tensor(results["audio_frames"]),
                "is_text_sample": is_text_sample,
            }
        )
        return results

    def transform_concat_to_context_length(self, results: Dict) -> Dict:
        """Pack all K samples of ``single_batch`` into ONE context-length sequence.

        Mirrors the parent ``SpeechOutAudioTextConversationTransform.
        transform_concat_to_context_length`` (single ``add_audio_text_in_conversation``
        call with the full lists + ``instruction_first``, then ONE ``get_prompt``), but
        emits a single packed ``seq_plan`` dict (carrying the K per-answer metadata
        lists + ``num_samples=K``) and the K raw ``answer_audio`` waveforms instead of
        building/padding codec tensors here. The codec build + ``get_prompt`` happen in
        ``model.forward`` via ``build_packed_sample_sequence``, keeping the result
        bit-exact with the ``.answer_cv_token`` path.
        """
        assert "texts" in results, f"No texts found in the annotations {results.get('annotations')}."
        assert "tokenizer" in results, "No tokenizer found in the results."

        audios = results["audios"]
        texts = results["texts"]
        user_prompts = results["user_prompts"]
        languages = results["languages"]
        contexts = results.get("contexts", [None] * len(texts))
        alignments = results.get("alignments", [None] * len(texts))
        use_eoc_endings = results.get("use_eoc_endings", [False] * len(texts))
        answer_audio = results.get("answer_audio", [None] * len(texts))
        infos = results.get("infos", [None] * len(texts))

        audio_embed_sizes_1d = []
        is_text_sample = False
        for embed_size in results["audio_embed_sizes"]:
            if sum(embed_size) == 0:
                is_text_sample = True
            audio_embed_sizes_1d.extend(embed_size)

        conv_template = self._new_conversation(results)
        conv_template = add_audio_text_in_conversation(
            texts,
            user_prompts,
            languages,
            contexts,
            conv_template,
            self.prompt_formatter,
            self.fixed_prompt,
            self.instruction_first,
        )

        num_samples = len(texts)
        codec_prefix_samples = [
            {"info": info} if info is not None else {} for info in infos
        ]

        packed_messages = [list(message) for message in conv_template.messages]
        _validate_audio_token_alignment(
            packed_messages, list(audio_embed_sizes_1d), "transform_concat_to_context_length"
        )

        seq_plan = [
            {
                "conv_template": self.s2s_conv_template_name,
                "messages": packed_messages,
                "system": conv_template.system,
                "num_audio_tokens": list(audio_embed_sizes_1d),
                "alignments": list(alignments),
                "use_eoc_endings": list(use_eoc_endings),
                "codec_prefix_samples": codec_prefix_samples,
                "num_samples": num_samples,
                "batch_format": "samples_to_sequence",
                "pad_to_context_length": self.pad_to_context_length,
                "context_length": self.context_length,
            }
        ]

        results.update(
            {
                "audios": audios,
                "seq_plan": seq_plan,
                "answer_audio": list(answer_audio),
                "audio_embed_sizes": torch.tensor(audio_embed_sizes_1d),
                "audio_frames": torch.tensor(results["audio_frames"]),
                "is_text_sample": is_text_sample,
            }
        )
        return results
