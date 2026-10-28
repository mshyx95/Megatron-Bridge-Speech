# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import copy
import math
from typing import Optional

import numpy as np
import torch

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.filters import remove_utterance_by_audio_token_length
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.alm_sample_processor import BlobChunkALMSampleProcessor
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger


logger = get_logger(__name__)
ANSWER_CODEC_POSITION_MARGIN = 1.10
DEFAULT_ANSWER_AUDIO_SAMPLE_RATE = 16000

try:
    from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.emotion_utils import map_info_to_token
except ImportError:
    map_info_to_token = None


class S2SBlobChunkALMSampleProcessor(BlobChunkALMSampleProcessor):
    def __init__(self, *args, codec_layer: int = 1, codec_cfg: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.codec_layer = codec_layer
        self.codec_cfg = copy.deepcopy(codec_cfg or {})

    def apply_audio_truncation(self, sample, sample_rate=None):
        return sample

    def apply_transcription_processing(self, sample, sample_processing_state):
        if sample.get("language") is None:
            sample = dict(sample)
            sample["language"] = ""
        return super().apply_transcription_processing(sample, sample_processing_state)

    def resolve_sample_processing_state(self, sample):
        state = super().resolve_sample_processing_state(sample)
        effective_filter_options, _, _ = self._merge_sample_processing_config(sample)
        state["remove_utterance_by_audio_token_length"] = effective_filter_options.get(
            "remove_utterance_by_audio_token_length",
            False,
        )
        state["min_audio_tokens"] = effective_filter_options.get("min_audio_tokens", 25)
        state["max_audio_tokens"] = effective_filter_options.get("max_audio_tokens", 1500)
        return state

    def passes_sample_filters(self, sample, sample_processing_state, sample_rate, filter_key=None):
        retain = super().passes_sample_filters(
            sample,
            sample_processing_state,
            sample_rate,
            filter_key=filter_key,
        )
        if sample_processing_state.get("remove_utterance_by_audio_token_length", False):
            retain &= remove_utterance_by_audio_token_length(
                sample,
                min_audio_tokens=sample_processing_state["min_audio_tokens"],
                max_audio_tokens=sample_processing_state["max_audio_tokens"],
            )
        return retain

    def process_codec_tokens(self, sample):
        if "answer_cv_token" in sample:
            label_codec = self._build_target_codec_layers(sample["answer_cv_token"], sample)
            if label_codec is None:
                return None
            sample["label_codec"] = label_codec
            sample["answer_cv_token"] = label_codec

        if "answer_cv_tokens" in sample:
            answer_cv_tokens = []
            for token in sample["answer_cv_tokens"]:
                label_codec = self._build_target_codec_layers(token, sample)
                if label_codec is None:
                    return None
                answer_cv_tokens.append(label_codec)
            sample["answer_cv_tokens"] = answer_cv_tokens

        if "label_codec" in sample:
            if sample.get("answer") is None:
                sample["answer"] = sample.get("transcription", "")
            answer = sample["answer"]
            if isinstance(answer, str) and len(answer) > 1 and answer[0] == "_" and " " in answer[:20]:
                sample["answer"] = "".join(answer.split(" ")).replace("▁", " ").replace("_", " ").strip()

        return sample

    def materialize_examples(self, sample, audio_frames, audio_embed_sizes, sample_processing_state):
        if "answer" in sample and "label_codec" in sample:
            example = {
                "user_prompt": sample.get("transcription"),
                "annotations": [sample["answer"]],
                "label": sample.get("label"),
                "label_codec": sample["label_codec"],
                self.final_audio_key: sample[self.final_audio_key],
                "audio_frames": audio_frames,
                "audio_embed_sizes": audio_embed_sizes,
                "language": sample["language"],
                "answer_alignment": sample.get("answer_alignment", None),
            }
            if "alignment" in sample:
                example["alignment"] = sample["alignment"]
            if "_use_eoc_ending" in sample:
                example["_use_eoc_ending"] = sample["_use_eoc_ending"]
            return [example]

        examples = super().materialize_examples(
            sample,
            audio_frames,
            audio_embed_sizes,
            sample_processing_state,
        )
        for example in examples:
            if "label_codec" in sample:
                example["label_codec"] = sample["label_codec"]
            if "answer_cv_tokens" in sample:
                example["answer_cv_tokens"] = sample["answer_cv_tokens"]
            if "alignment" in sample:
                example["alignment"] = sample["alignment"]
            if "answer_alignment" in sample:
                example["answer_alignment"] = sample["answer_alignment"]
            if "_use_eoc_ending" in sample:
                example["_use_eoc_ending"] = sample["_use_eoc_ending"]
        return examples

    def _build_target_codec_layers(self, tokens, sample):
        return self.build_target_codec_layers(tokens, self.codec_layer, self.codec_cfg, sample)

    def _codec_prefix(self, sample):
        return self.codec_prefix(self.codec_cfg, sample)

    def _normalize_flat_codec_tokens(self, tokens):
        return self.normalize_flat_codec_tokens(tokens)

    def _parse_codec_string(self, tokens: str):
        return self.parse_codec_string(tokens)

    @staticmethod
    def build_target_codec_layers(tokens, codec_layer, codec_cfg, sample):
        """Flat codec tokens → list of ``codec_layer`` prefixed codec layers.

        Shared by the dataloader sample processor and the forward-side builder so
        both paths reproduce the same layer split + prefix exactly.
        """
        flat_tokens = S2SBlobChunkALMSampleProcessor.normalize_flat_codec_tokens(tokens)
        if flat_tokens is None:
            return None

        remain_len = len(flat_tokens) - len(flat_tokens) % codec_layer
        flat_tokens = flat_tokens[:remain_len]
        if len(flat_tokens) == 0:
            return None

        codec_layers = [flat_tokens[i::codec_layer] for i in range(codec_layer)]
        prefix = S2SBlobChunkALMSampleProcessor.codec_prefix(codec_cfg, sample)
        return [prefix + layer for layer in codec_layers]

    @staticmethod
    def codec_prefix(codec_cfg, sample):
        codec_cfg = codec_cfg or {}
        if codec_cfg.get("use_emotion", False):
            if map_info_to_token is None:
                raise ImportError("use_emotion=True requires transforms.emotion_utils.map_info_to_token")
            try:
                return map_info_to_token(sample["info"])
            except Exception:
                logger.info("Error in mapping emotion token, use neutral emotion instead.")
                return [6622, 6603, 6603, 6603, 6603]

        if codec_cfg.get("speaker_id", None) is not None:
            speaker_id = 6617 + int(codec_cfg["speaker_id"])
            return [speaker_id, 6622, 6622, 6622]

        return [6622, 6622, 6622]

    @staticmethod
    def normalize_flat_codec_tokens(tokens):
        if tokens is None:
            return None
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        elif isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        elif isinstance(tokens, str):
            tokens = S2SBlobChunkALMSampleProcessor.parse_codec_string(tokens)

        if not isinstance(tokens, list) or len(tokens) == 0:
            return None

        if len(tokens) == 1 and isinstance(tokens[0], list):
            tokens = tokens[0]

        if all(isinstance(layer, list) for layer in tokens):
            tokens = [token for layer in tokens for token in layer]

        return [int(token) for token in tokens]

    @staticmethod
    def parse_codec_string(tokens: str):
        tokens = tokens.strip()
        if not tokens:
            return None
        try:
            parsed = ast.literal_eval(tokens)
        except (SyntaxError, ValueError):
            parsed = tokens.replace(",", " ").split()
        return parsed


class SpeechOutRawAudioSampleProcessor(S2SBlobChunkALMSampleProcessor):
    """Sample processor for the raw-audio forward path.

    Carries the raw ``answer_audio`` waveform onto the example (no tokenization,
    no ``answer_cv_token`` conversion, no codec-layer build). It reproduces the
    same ``annotations`` / ``user_prompt`` / answer-text cleanup as the codec
    branch of :class:`S2SBlobChunkALMSampleProcessor.materialize_examples` so the
    forward-side sequence build is bit-identical to the old path.
    """

    def __init__(self, *args, prompt_formatter: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_formatter = prompt_formatter
        self._answer_audio_drop_count = 0

    def process_codec_tokens(self, sample):
        # No codec tokens on this path; nothing to build. Keep answer_audio as-is.
        return sample

    @staticmethod
    def _split_answer_audio(answer_audio):
        if answer_audio is None:
            return None, DEFAULT_ANSWER_AUDIO_SAMPLE_RATE
        if isinstance(answer_audio, (tuple, list)) and len(answer_audio) == 2:
            return answer_audio[0], answer_audio[1]
        return answer_audio, DEFAULT_ANSWER_AUDIO_SAMPLE_RATE

    def _estimate_answer_positions(self, answer_audio, sample):
        waveform, sample_rate = self._split_answer_audio(answer_audio)
        if waveform is None:
            return 0

        try:
            num_frames = len(waveform)
        except TypeError:
            return 0
        if num_frames <= 0:
            return 0

        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            sample_rate = DEFAULT_ANSWER_AUDIO_SAMPLE_RATE
        if sample_rate <= 0:
            sample_rate = DEFAULT_ANSWER_AUDIO_SAMPLE_RATE

        codec_layer = self.codec_cfg.get("codec_layer", self.codec_layer)
        try:
            codec_layer = int(codec_layer)
        except (TypeError, ValueError):
            codec_layer = int(self.codec_layer or 1)
        codec_layer = max(1, codec_layer)

        n_raw_codec = math.ceil(num_frames * 25 / sample_rate)
        codec_positions = math.ceil(n_raw_codec / codec_layer)
        margined_codec_positions = math.ceil(codec_positions * ANSWER_CODEC_POSITION_MARGIN)
        prefix_and_end = max(4, len(self._codec_prefix(sample)) + 1)
        return margined_codec_positions + prefix_and_end

    def materialize_examples(self, sample, audio_frames, audio_embed_sizes, sample_processing_state):
        # answer_audio resolution for the forward_tokenize path:
        #  - answer_audio present (S2S datasets with an answer_audio chunk) → use it.
        #  - else ASR only (prompt_formatter == "codec_pre") with input audio present →
        #    fall back to the input audio waveform (audio == answer audio for ASR, e.g. S1
        #    libriheavy) and tokenize it in forward.
        #  - else (non-ASR sample with no answer_audio) → DROP the example. Never raise;
        #    never substitute the question audio as the answer for non-ASR data.
        answer_audio = sample.get("answer_audio")
        if answer_audio is None:
            if self.prompt_formatter == "codec_pre" and sample.get(self.input_key) is not None:
                answer_audio = sample.get(self.input_key)
            else:
                self._answer_audio_drop_count += 1
                if (self._answer_audio_drop_count & (self._answer_audio_drop_count - 1)) == 0:
                    logger.debug(
                        "forward_tokenize: dropping sample with no answer_audio "
                        "(prompt_formatter=%s, non-ASR); total dropped=%d",
                        self.prompt_formatter,
                        self._answer_audio_drop_count,
                    )
                return []

        # Replicate the answer-text setup that process_codec_tokens performs on
        # the old path, so annotations match exactly.
        if sample.get("answer") is None:
            sample["answer"] = sample.get("transcription", "")
        answer = sample["answer"]
        if isinstance(answer, str) and len(answer) > 1 and answer[0] == "_" and " " in answer[:20]:
            sample["answer"] = "".join(answer.split(" ")).replace("▁", " ").replace("_", " ").strip()

        example = {
            "user_prompt": sample.get("transcription"),
            "annotations": [sample["answer"]],
            "label": sample.get("label"),
            "answer_audio": answer_audio,
            # Packing budget for forward_tokenize raw audio: CosyVoice2-v2 emits
            # ~25 tokens/sec; reserve 10% extra codec positions plus prefix+EOA/EOC.
            "est_answer_positions": self._estimate_answer_positions(answer_audio, sample),
            self.final_audio_key: sample[self.final_audio_key],
            "audio_frames": audio_frames,
            "audio_embed_sizes": audio_embed_sizes,
            "language": sample["language"],
            "answer_alignment": sample.get("answer_alignment", None),
            "spkid": sample.get("spkid", None),
            "info": sample.get("info", None),
        }
        if "alignment" in sample:
            example["alignment"] = sample["alignment"]
        if "_use_eoc_ending" in sample:
            example["_use_eoc_ending"] = sample["_use_eoc_ending"]
        return [example]


BlobChunkALMSampleProcessor = S2SBlobChunkALMSampleProcessor