import copy
import json
import math
import re
from typing import Dict, List, Optional

import torch

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.filters import (
    remove_utterance_by_length,
    remove_utterance_by_wer,
    remove_utterance_by_ctc_length_filter,
    illegal_char_filter,
    remove_utterance_by_comparing_number,
    remove_utterance_by_consecutive_numbers,
    remove_utterance_by_punctuation_numbers,
    remove_utterance_by_corpus_transcript_pattern,
    illegal_word_filter,
    check_zh_en_wer,
    messages_from_response,
    get_long_form_transcription,
    TNRuleWord,
    CaseNorm,
    CharMapper,
    TranscriptSelector,
    TransAbbrFix,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.filters.biasing import PieceSampler
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.sqa_helper import generate_sqa_sample, apply_transcribe_cot
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.lang_utils import LANGUAGE_PAIRS as language_pairs


logger = get_logger(__name__)

PUNCTUATION = (
    r"，。！？；：、"
    r"）】》」』"
)


def _deep_merge_dict(base: Optional[dict], override: Optional[dict]) -> dict:
    if base is None:
        base = {}
    if override is None:
        return copy.deepcopy(base)

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class BlobChunkALMSampleProcessor:
    def __init__(
        self,
        tokenizer,
        audio_processor,
        feat_stride: int,
        input_key: str,
        final_audio_key: str,
        allow_pure_text_return: bool,
        retain_first_n_turns_in_conversation: Optional[int],
        default_filter_options: Optional[dict],
        default_text_normalization: Optional[dict],
        default_sp_transcription: bool,
        compute_audio_embed_size,
    ):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.feat_stride = feat_stride
        self.input_key = input_key
        self.final_audio_key = final_audio_key
        self.allow_pure_text_return = allow_pure_text_return
        self.retain_first_n_turns_in_conversation = retain_first_n_turns_in_conversation
        self.default_filter_options = copy.deepcopy(default_filter_options or {})
        self.default_text_normalization = copy.deepcopy(default_text_normalization)
        self.default_sp_transcription = default_sp_transcription
        self.compute_audio_embed_size = compute_audio_embed_size
        self._sample_processing_state_cache = {}

    def _build_text_normalizer(self, text_normalization: Optional[dict]):
        if not text_normalization:
            return None

        tn_rule_file = text_normalization.get("tn_rule_file", None)
        if tn_rule_file is not None:
            case_normalize = text_normalization.get("case_normalize", None)
            unicode_normalize = text_normalization.get("unicode_normalize", None)
            return TNRuleWord(tn_rule_file, case_normalize, unicode_normalize)

        case_normalize = text_normalization.get("case_normalize", "lower")
        return CaseNorm(case_normalize)

    def _build_char_mapper(self, text_normalization: Optional[dict]):
        if not text_normalization:
            return None
        char_map_file = text_normalization.get("char_map_file", None)
        if char_map_file is None:
            return None
        return CharMapper(char_map_file)

    def _load_punctuation_list(self, punctuation_config: Optional[dict]):
        if punctuation_config is None:
            return None

        punctuation_list_file = punctuation_config["punctuation_list_file"]
        if punctuation_list_file.startswith("az://"):
            import blobfile as bf
            with bf.BlobFile(punctuation_list_file, "r") as f:
                return f.read().splitlines()

        with open(punctuation_list_file, "r") as f:
            return f.read().splitlines()

    def _normalize_illegal_word_list(self, illegal_word_list):
        if illegal_word_list is None:
            return None
        if isinstance(illegal_word_list, list):
            return illegal_word_list
        return [illegal_word_list]

    def _merge_sample_processing_config(self, sample: Dict[str, any]):
        sample_overrides = sample.get("_sample_overrides") or {}

        effective_filter_options = _deep_merge_dict(
            self.default_filter_options,
            sample_overrides.get("filter_options", {}),
        )

        effective_text_normalization = _deep_merge_dict(
            self.default_text_normalization,
            sample_overrides.get("text_normalization"),
        )
        if not effective_text_normalization:
            effective_text_normalization = None

        effective_sp_transcription = sample_overrides.get("sp_transcription", self.default_sp_transcription)
        return effective_filter_options, effective_text_normalization, effective_sp_transcription

    def resolve_sample_processing_state(self, sample: Dict[str, any]) -> Dict[str, any]:
        effective_filter_options, effective_text_normalization, effective_sp_transcription = self._merge_sample_processing_config(sample)
        cache_key = json.dumps(
            {
                "filter_options": effective_filter_options,
                "text_normalization": effective_text_normalization,
                "sp_transcription": effective_sp_transcription,
            },
            sort_keys=True,
            default=str,
        )
        if cache_key in self._sample_processing_state_cache:
            return self._sample_processing_state_cache[cache_key]

        transcript_selector_params = effective_filter_options.get("transcript_selector")
        transcript_selector = TranscriptSelector(**transcript_selector_params) if transcript_selector_params is not None else None
        trans_abbr_fix_params = effective_filter_options.get("trans_abbr_fix")
        trans_abbr_fix = TransAbbrFix(**trans_abbr_fix_params) if trans_abbr_fix_params is not None else None
        remove_utterance_by_punctuation_numbers_cfg = effective_filter_options.get("remove_utterance_by_punctuation_numbers")
        state = {
            "max_utterance_frames": math.ceil(effective_filter_options.get("max_utterance_frames", 3000) / self.feat_stride),
            "min_utterance_frames": math.ceil(effective_filter_options.get("min_utterance_frames", 0) / self.feat_stride),
            "max_utterance_labels": effective_filter_options.get("max_utterance_labels", 50000000),
            "min_utterance_labels": effective_filter_options.get("min_utterance_labels", 0),
            "remove_utterance_by_wer": effective_filter_options.get("remove_utterance_by_wer", False),
            "response_options": effective_filter_options.get("response_options", {}),
            "sqa_sample": effective_filter_options.get("sqa_sample", {}),
            "filter_on_all": effective_filter_options.get("filter_on_all", False),
            "transcript_selector": transcript_selector,
            "trans_abbr_fix": trans_abbr_fix,
            "max_wer": effective_filter_options.get("max_wer", 0.5),
            "remove_utterance_by_ctc_length": effective_filter_options.get("remove_utterance_by_ctc_length", False),
            "ctc_filter_subsampling_factor": effective_filter_options.get("ctc_filter_subsampling_factor", 8) / self.feat_stride,
            "illegal_char_filter": effective_filter_options.get("illegal_char_filter", False),
            "remove_utterance_by_comparing_number": effective_filter_options.get("remove_utterance_by_comparing_number", False),
            "remove_utterance_by_consecutive_numbers": effective_filter_options.get("remove_utterance_by_consecutive_numbers", False),
            "remove_utterance_by_punctuation_numbers": remove_utterance_by_punctuation_numbers_cfg,
            "punctuation_list": self._load_punctuation_list(remove_utterance_by_punctuation_numbers_cfg),
            "illegal_word_list": self._normalize_illegal_word_list(effective_filter_options.get("illegal_word_list")),
            "remove_utterance_by_corpus_transcript_pattern": effective_filter_options.get("remove_utterance_by_corpus_transcript_pattern"),
            "recover_long_form_block": effective_filter_options.get("recover_long_form", {}),
            "text_normalizer": self._build_text_normalizer(effective_text_normalization),
            "char_mapper": self._build_char_mapper(effective_text_normalization),
            "sp_transcription": effective_sp_transcription,
            "mix_normal_sr": effective_filter_options.get("mix_normal_sr", 0.0),
            "biasing_sampler": PieceSampler(**effective_filter_options.get("biasing", {})) if effective_filter_options.get("biasing") else None,
        }
        self._sample_processing_state_cache[cache_key] = state
        return state

    def prepare_multiaudio_sample(self, sample: Dict[str, any], sample_processing_state: Dict[str, any]) -> Optional[Dict[str, any]]:
        if "audios" not in sample:
            return sample

        assert "sft" in sample
        if sample_processing_state["sqa_sample"]:
            sample = generate_sqa_sample(sample, **sample_processing_state["sqa_sample"])
        if len(sample["sft"]) == 0 or len(sample["audios"]) == 0:
            return None
        if any(audio is None for audio in sample["audios"]):
            logger.info("None audio in the sample, skip")
            return None

        sample["audio"] = sample["audios"][0]
        messages = sample["sft"][0]["messages"]
        if self.retain_first_n_turns_in_conversation is not None:
            messages = messages[: 2 * self.retain_first_n_turns_in_conversation]

        predictions = self.tokenizer.encode(
            "".join([messages[i]["content"] for i in range(len(messages)) if messages[i]["role"] == "assistant"]),
            add_special_tokens=False,
        )
        if self.allow_pure_text_return or sample_processing_state["filter_on_all"]:
            sample["label"] = self.tokenizer.encode(
                "".join([messages[i]["content"] for i in range(len(messages))]),
                add_special_tokens=False,
            )
            if len(predictions) == 0:
                return None
        else:
            sample["label"] = predictions

        return sample

    def apply_transcription_processing(self, sample: Dict[str, any], sample_processing_state: Dict[str, any]) -> Optional[Dict[str, any]]:
        if "transcription" not in sample:
            return sample

        transcript_selector = sample_processing_state["transcript_selector"]
        if transcript_selector is not None:
            sample = transcript_selector(sample)

        trans_abbr_fix = sample_processing_state["trans_abbr_fix"]
        if trans_abbr_fix is not None:
            sample = trans_abbr_fix(sample)

        if sample_processing_state["recover_long_form_block"]:
            sample = get_long_form_transcription(
                sample,
                sample_processing_state["max_utterance_frames"],
                sample_processing_state["min_utterance_frames"],
                **sample_processing_state["recover_long_form_block"],
            )
            if sample is None:
                return None

        if sample_processing_state["text_normalizer"]:
            sample["transcription"] = sample_processing_state["text_normalizer"].process_a_line(sample["transcription"])

        if sample_processing_state["char_mapper"]:
            sample["transcription"] = sample_processing_state["char_mapper"].replace_chars(sample["transcription"])

        if sample_processing_state["sp_transcription"]:
            sample["transcription"] = "".join(sample["transcription"].split(" ")).replace("▁", " ").replace("_", " ").strip()

        language = sample.get("language") or ""
        if ("zh" in language or "CN" in language) and not sample_processing_state["sp_transcription"]:
            pattern = rf'(?<=[\u4e00-\u9fff{PUNCTUATION}])\s+'
            sample["transcription"] = re.sub(pattern, '', sample["transcription"])
            pattern = rf'(?<=[a-zA-Z0-9])\s+(?=[{PUNCTUATION}\u4e00-\u9fff])'
            sample["transcription"] = re.sub(pattern, '', sample["transcription"])
            pattern = r'(?<=[A-Z])\s+(?=[0-9])'
            sample["transcription"] = re.sub(pattern, '', sample["transcription"])

        if len(sample["transcription"]) == 0 and sample_processing_state["min_utterance_labels"] <= 0:
            sample["transcription"] = "<nonspeech>"

        if sample_processing_state["illegal_word_list"] is not None and not illegal_word_filter(sample, sample_processing_state["illegal_word_list"]):
            return None
        assert "<abnormal>" not in sample["transcription"]

        sample["label"] = self.tokenizer.encode(sample["transcription"], add_special_tokens=False)
        return sample

    def passes_sample_filters(self, sample: Dict[str, any], sample_processing_state: Dict[str, any], sample_rate: Optional[int], filter_key: Optional[str] = None) -> bool:
        if filter_key is not None:
            # Caller explicitly specifies the key to filter on (e.g. pre-extracted features)
            max_feat_len = sample_processing_state["max_utterance_frames"]
            min_feat_len = sample_processing_state["min_utterance_frames"]
            ctc_filter_subsampling_factor = sample_processing_state["ctc_filter_subsampling_factor"]
        elif self.audio_processor and sample_rate is not None:
            filter_key = self.input_key
            max_feat_len = math.ceil(sample_processing_state["max_utterance_frames"] * self.feat_stride / 100 * sample_rate)
            min_feat_len = math.ceil(sample_processing_state["min_utterance_frames"] * self.feat_stride / 100 * sample_rate)
            ctc_filter_subsampling_factor = sample_processing_state["ctc_filter_subsampling_factor"] / 100 * sample_rate
        else:
            filter_key = self.final_audio_key
            max_feat_len = sample_processing_state["max_utterance_frames"]
            min_feat_len = sample_processing_state["min_utterance_frames"]
            ctc_filter_subsampling_factor = sample_processing_state["ctc_filter_subsampling_factor"]

        retain = remove_utterance_by_length(
            sample,
            max_feat_len=max_feat_len,
            max_label_len=sample_processing_state["max_utterance_labels"],
            min_feat_len=min_feat_len,
            min_label_len=sample_processing_state["min_utterance_labels"],
            feature_key=filter_key,
            label_key="label",
        )

        if sample_processing_state["remove_utterance_by_wer"]:
            skip_wer = False
            transcript_selector = sample_processing_state["transcript_selector"]
            language = sample.get("language") or ""
            if ("zh" in language or "CN" in language) and transcript_selector is not None:
                transcription_keys = transcript_selector.get_transcription_key()
                skip_wer, res = check_zh_en_wer(sample, transcription_keys)
            if not skip_wer:
                retain &= remove_utterance_by_wer(sample, max_wer=sample_processing_state["max_wer"])
            else:
                retain &= res
        if sample_processing_state["remove_utterance_by_ctc_length"]:
            retain &= remove_utterance_by_ctc_length_filter(sample, feature_key=filter_key, label_key="label", subsampling_factor=ctc_filter_subsampling_factor)
        if sample_processing_state["illegal_char_filter"]:
            retain &= illegal_char_filter(sample)
        if sample_processing_state["remove_utterance_by_comparing_number"]:
            retain &= remove_utterance_by_comparing_number(sample)
        if sample_processing_state["remove_utterance_by_consecutive_numbers"]:
            retain &= remove_utterance_by_consecutive_numbers(sample)
        if sample_processing_state["remove_utterance_by_punctuation_numbers"] is not None:
            threshold = sample_processing_state["remove_utterance_by_punctuation_numbers"].get("threshold", 20)
            retain &= remove_utterance_by_punctuation_numbers(sample, sample_processing_state["punctuation_list"], threshold=threshold)
        if sample_processing_state["remove_utterance_by_corpus_transcript_pattern"] is not None:
            cfg = sample_processing_state["remove_utterance_by_corpus_transcript_pattern"]
            retain &= remove_utterance_by_corpus_transcript_pattern(
                sample,
                corpus_patterns=cfg["corpus_patterns"],
                original_transcript_patterns=cfg["original_transcript_patterns"],
            )
        if len(sample_processing_state["response_options"]) > 0:
            msgs = messages_from_response(sample, **sample_processing_state["response_options"])
            retain &= len(msgs) > 0
            sample["sft"] = msgs

        if "sft" in sample:
            sample.pop("label", None)
        return retain

    def extract_audio_features(self, sample: Dict[str, any], sample_rate: Optional[int]):
        if self.allow_pure_text_return:
            sample[self.final_audio_key] = [torch.tensor([])]
            return [0], [0]

        if self.audio_processor:
            if "audios" in sample:
                sample[self.final_audio_key] = [self.audio_processor.extract_features(audio[0], audio[1]) for audio in sample["audios"]]
                if self.retain_first_n_turns_in_conversation is not None:
                    sample[self.final_audio_key] = sample[self.final_audio_key][: self.retain_first_n_turns_in_conversation]
            elif "audio" in sample:
                sample[self.final_audio_key] = [self.audio_processor.extract_features(sample["audio"], sample_rate)]
            elif sample.get(self.final_audio_key) is not None:
                # Features already pre-extracted (e.g. by _maybe_segment_podcast)
                sample[self.final_audio_key] = [sample[self.final_audio_key]]
            else:
                assert False, "No audio or pre-extracted feature found in sample"
        else:
            sample[self.final_audio_key] = [sample[self.input_key]]

        audio_frames = [len(audio) * self.feat_stride for audio in sample[self.final_audio_key]]
        audio_embed_sizes = [self.compute_audio_embed_size(audio_frame) for audio_frame in audio_frames]

        if hasattr(self.audio_processor, "pad_to_thirty_seconds_encoder_only") and self.audio_processor.pad_to_thirty_seconds_encoder_only:
            sample[self.final_audio_key] = [self.audio_processor.pad_to_chunk_length(audio) for audio in sample[self.final_audio_key]]

        return audio_frames, audio_embed_sizes

    def materialize_examples(self, sample: Dict[str, any], audio_frames: List[int], audio_embed_sizes: List[int], sample_processing_state: Dict[str, any]) -> List[dict]:
        examples = []
        if "single_qa" in sample:
            annotations = sample["single_qa"]
            questions, answers, scores = annotations["questions"], annotations["answer"], annotations["score"]

            if isinstance(scores, list):
                questions = [q.lstrip()[2:] for q in questions]
            else:
                questions = questions if isinstance(questions, str) else ""
                questions, answers, scores = [questions], [answers], [scores]

            for i, score in enumerate(scores):
                if score < 9:
                    continue
                query = questions[i].strip()
                answer = answers[i].strip()
                if sample_processing_state["sqa_sample"]:
                    query, answer = apply_transcribe_cot(query, answer, sample["transcription"], **sample_processing_state["sqa_sample"])
                examples.append({
                    "user_prompt": query,
                    "annotations": [answer],
                    "label": None,
                    self.final_audio_key: sample[self.final_audio_key],
                    "audio_frames": audio_frames,
                    "audio_embed_sizes": audio_embed_sizes,
                    "language": sample["language"],
                })
            return examples

        if "sft" in sample:
            for i in range(len(sample["sft"])):
                messages = sample["sft"][i]["messages"]
                if self.retain_first_n_turns_in_conversation is not None:
                    messages = messages[: 2 * self.retain_first_n_turns_in_conversation]
                examples.append({
                    "user_prompt": None,
                    "annotations": messages,
                    "label": None,
                    self.final_audio_key: sample[self.final_audio_key],
                    "audio_frames": audio_frames,
                    "audio_embed_sizes": audio_embed_sizes,
                    "language": sample["language"],
                })
            return examples

        context = None
        if sample_processing_state["biasing_sampler"] is not None and torch.rand(1).item() > sample_processing_state["mix_normal_sr"]:
            locale = sample.get("language", "unknown")
            language = language_pairs.get(locale, 'None')
            context, _ = sample_processing_state["biasing_sampler"].sample(sample["transcription"], language=language)
        examples.append({
            "user_prompt": None,
            "annotations": [sample["transcription"]],
            "label": sample["label"],
            self.final_audio_key: sample[self.final_audio_key],
            "audio_frames": audio_frames,
            "audio_embed_sizes": audio_embed_sizes,
            "language": sample["language"],
            "spkid": sample.get("spkid", None),
            "context": context,
        })
        return examples