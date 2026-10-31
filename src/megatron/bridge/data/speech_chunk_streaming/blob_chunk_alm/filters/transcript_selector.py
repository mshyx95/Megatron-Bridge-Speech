# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import string
from typing import List, Dict

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.filters.align_cap_punc import align_utterance
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger

LOG = get_logger(__name__)

# Constants used for updating DPP trans with GPT puncuation and capitalization
PUNCTUATION_DEFAULT = string.punctuation  # punctuation set used for applying GPT punc/cap to DPP
PUNCTUATION_TO_REMOVE_DEFAULT = "LEADING_TRAILING"  # must be either "LEADING_TRAILING" or "ALL"
ALIGNMENT_ERR_THRESHOLD_DEFAULT = 20  # threshold for alignment error rate fallback

class TranscriptSelector:
    """Choose a transcript from the alternative transcription data in chunk data.
        By default, this will fetch the transcript in the info chunk's item:
        chunk_example['info']['alternative_transcription']

    Args:
        transcription_keys (List[str]): the keys that indicate transcription data souces
            from info chunk data. It will search the data from the list order.
            If the data are found, stop the search and use it.
            Defaults to [
                "display_human_caption_adjusted_with_filler_tag",
                "display_dpp"
            ]
        info_key (str, optional): the info key.
            Defaults to 'info'
        input_key (str, optional): the alternative transcription key.
            Defaults to 'alternative_transcription'
        output_key (str, optional): the output field key.
            Defaults to 'transcription'
        tag_stats_key (str, optional): key to the tag stats.
            This will be used for filtering utterances with tagging failure.
            Defaults to 'tag_stats'
        filter_failed_verbatm_tagger (bool, optional):
            skip utterances marked as tagging failure. Defaults to True
        grep_pattern (str, optional): only pass a sample if the transcription matches
            to `grep_pattern`
        apply_gpt_dpp_fall_back_logic (bool, optional): If True, apply punctuation and
            capitalization from GPT trans to DPP trans. Defaults to False.
        apply_gpt_dpp_fall_back_punctuation (str, optional): punctuation set used for applying
            GPT punc/cap to DPP. Defaults to string.punctuation (English punctuation set).
        apply_gpt_dpp_fall_back_punctuation_to_remove (str, optional): what punctuation to remove
            from the DPP transcription. Must be either "LEADING_TRAILING" or "ALL". Defaults to
            "LEADING_TRAILING".
        apply_gpt_dpp_fall_back_enable_gpt_fallback (bool, optional): If True, fallback to GPT
            output for transcripts with URLs or emails. Defaults to False.
        apply_gpt_dpp_fall_back_enable_alignment_err_fallback (bool, optional): If True, enable
            logic to fallback to DPP when alignment error > given threshold. Defaults to False.
        apply_gpt_dpp_fall_back_alignment_err_threshold (int, optional): The threshold for the
            alignment error rate fallback logic. Defaults to 20.
        require_transcription_key (bool, optional): If True, clear output transcription when none
            of the configured alternative transcription keys exists. Defaults to False to preserve
            historical fallback behavior.
        log_debug (bool, optional): If True, print the debug information. Defaults to False.
    """

    # Here, define the keys in .info chunk data
    alternative_transcription_keys = {
        # raw transcription data in display format
        "display_human_caption",
        # non-verbatim display format
        "display_human_caption_adjusted",
        # non-verbatim (lexical) format converted from display format
        "lexical_tned_mixed_case_human_caption_adjusted",
        # verbatim display format
        "display_human_caption_adjusted_with_filler_tag",
        # verbatim display format for R%d datasets
        "display_dpp",
        # lexical format
        "lexical_tned_mixed_case",
        # lexical format with filler tags
        "lexical_tned_mixed_case_human_caption_adjusted_with_filler_tag",
        # transcription data with speaker change tags
        "original_human_caption_with_spk_change_tag",
        "display_human_caption_adjusted_with_spk_change_tag",
        "lexical_tned_mixed_case_human_caption_adjusted_with_spk_change_tag",
        # copied from the N-best result
        "lexical_hybrid_decode",
        "display_dpp_es-es_DisfluencyRemoved",
        "display_dpp_es-es",
        "display_dpp_es-mx_DisfluencyRemoved",
        "display_dpp_es-mx",
        "display_GPT4o_from_dpp_es-es_DisfluencyRemoved",
        "display_GPT4o_from_dpp_es-mx_DisfluencyRemoved",
    }

    def __init__(
        self,
        transcription_keys: List[str] = None,
        info_key: str = "info",
        input_key: str = "alternative_transcription",
        output_key: str = "transcription",
        tag_stats_key: str = "tag_stats",
        filter_tagging_failure_data: bool = True,
        grep_pattern: str = None,
        apply_gpt_dpp_fall_back_logic: bool = False,
        apply_gpt_dpp_fall_back_punctuation: str = PUNCTUATION_DEFAULT,
        apply_gpt_dpp_fall_back_punctuation_to_remove: str = PUNCTUATION_TO_REMOVE_DEFAULT,
        apply_gpt_dpp_fall_back_enable_gpt_fallback: bool = False,
        apply_gpt_dpp_fall_back_enable_alignment_err_fallback: bool = False,
        apply_gpt_dpp_fall_back_alignment_err_threshold: int = ALIGNMENT_ERR_THRESHOLD_DEFAULT,
        require_transcription_key: bool = False,
        log_debug: bool = False,
    ):
        """Init"""
        self.transcription_keys = transcription_keys
        if transcription_keys is None:
            self.transcription_keys = [
                "display_human_caption_adjusted_with_filler_tag",
                "display_dpp",
            ]
        self._info_key = info_key
        self._input_key = input_key
        self._output_key = output_key
        self._tag_stats_key = tag_stats_key
        self._filter_tagging_failure_data = filter_tagging_failure_data
        if set(self.transcription_keys).issubset(set(self.alternative_transcription_keys)) is False:
            LOG.warning("Using an unregistered transcription key: %s", self.transcription_keys)
            LOG.warning("Not in the allowable key list: %s", self.alternative_transcription_keys)
        self._grep_pattern = re.compile(grep_pattern) if grep_pattern else None
        self.apply_gpt_dpp_fall_back_logic = apply_gpt_dpp_fall_back_logic
        self.apply_gpt_dpp_fall_back_punctuation = apply_gpt_dpp_fall_back_punctuation
        self.apply_gpt_dpp_fall_back_punctuation_to_remove = (
            apply_gpt_dpp_fall_back_punctuation_to_remove
        )
        self.apply_gpt_dpp_fall_back_enable_gpt_fallback = (
            apply_gpt_dpp_fall_back_enable_gpt_fallback
        )
        self.apply_gpt_dpp_fall_back_enable_alignment_err_fallback = (
            apply_gpt_dpp_fall_back_enable_alignment_err_fallback
        )
        self.apply_gpt_dpp_fall_back_alignment_err_threshold = (
            apply_gpt_dpp_fall_back_alignment_err_threshold
        )
        self.require_transcription_key = require_transcription_key
        self.log_debug = log_debug

    def __call__(self, example):

        alt_trans, transcription_key = self.get_transcription(example[self._info_key])

        if transcription_key and self.should_filter(example[self._info_key], transcription_key):
            LOG.debug(
                "Filter utterance: %s",
                example[self._info_key].get("filename", example[self._info_key]),
            )
            example[self._output_key] = ""
            # filter later

        if alt_trans is not None:  # Use the alternative transcription data
            example[self._output_key] = alt_trans
        elif self.require_transcription_key:
            example[self._output_key] = ""

        if self._grep_pattern and not re.search(self._grep_pattern, example[self._output_key]):
            # filter a sample if the transcription does not match self._grep_pattern
            example[self._output_key] = ""
            # filter later

        return example

    def should_filter(self, info_chunk: Dict, transcription_key: str):
        """Filter the utterances that a tagging API failed to process"""
        if self._filter_tagging_failure_data:
            if (
                self._tag_stats_key not in info_chunk
                or transcription_key not in info_chunk[self._tag_stats_key]
            ):
                return False

            # check if the tagger API failed during data-prep
            tagging_result = info_chunk[self._tag_stats_key][transcription_key]
            if int(tagging_result.get("tagging_api_failed_sent", -1)) == 1:
                return True  # API communication failed

            if (
                int(tagging_result.get("tagging_required_sent", -1)) == 1
                and int(tagging_result.get("tagging_succeed_sent", -1)) == 0
            ):
                return True  # The API failed to tag the data

        return False

    def get_transcription(self, info_chunk: Dict):
        """Extract context information from an input example

        Args:
            info_chunk (Dict): .info chunk data

        Returns:
            transcription (str) and its key (str)
        """

        if self._input_key in info_chunk:
            if not self.apply_gpt_dpp_fall_back_logic:  # default behavior
                for transcription_key in self.transcription_keys:
                    if transcription_key in info_chunk[self._input_key]:
                        trans_text = info_chunk[self._input_key][transcription_key]
                        # Handle list type: take first element, this happens for some zh-CN data
                        if isinstance(trans_text, list) and len(trans_text) > 0:
                            trans_text = trans_text[0]
                        return trans_text, transcription_key
            else:
                dpp_trans = None
                gpt_trans = None
                dpp_key = None
                gpt_key = None
                for transcription_key in self.transcription_keys:
                    if transcription_key in info_chunk[self._input_key]:
                        if "display_GPT4o" in transcription_key:
                            gpt_trans = info_chunk[self._input_key][transcription_key]
                            gpt_key = transcription_key
                        elif "display_dpp" in transcription_key:
                            dpp_trans = info_chunk[self._input_key][transcription_key]
                            dpp_key = transcription_key

                def contains_number_with_dot_or_comma(s):
                    pattern = r"\d*[\.\,]\d+"
                    return bool(re.search(pattern, s))

                if gpt_trans is None or contains_number_with_dot_or_comma(gpt_trans):
                    # If gpt_trans doesn't exist, or gpt_trans contains a number with dot or comma
                    # (i.e. dubious wether it's floating point number or thousand separator),
                    # then use dpp_trans
                    if gpt_trans is not None and self.log_debug:
                        print("gpt contains number with dot or comma")
                        print(gpt_trans)
                    return dpp_trans, dpp_key

                if dpp_trans is not None and gpt_trans is not None:
                    # Both gpt trans and dpp trans are available. Try to update dpp trans.
                    if self.log_debug:
                        print("Start alignment for")
                        print(f"dpp_trans:         {dpp_trans}")
                        print(f"gpt_trans:         {gpt_trans}")
                        print(f"punctuation:      {self.apply_gpt_dpp_fall_back_punctuation}")
                        print(
                            f"punctuation to remove: "
                            f"{self.apply_gpt_dpp_fall_back_punctuation_to_remove}"
                        )
                        print(
                            f"enable gpt fall back logic:"
                            f"{self.apply_gpt_dpp_fall_back_enable_gpt_fallback}"
                        )
                        print(
                            f"enable alignment error fall back logic:"
                            f"{self.apply_gpt_dpp_fall_back_enable_alignment_err_fallback}"
                        )
                        print(
                            f"alignment error threshold:"
                            f"{self.apply_gpt_dpp_fall_back_alignment_err_threshold}"
                        )

                    if self.apply_gpt_dpp_fall_back_enable_gpt_fallback and (
                        "@" in dpp_trans
                        or any(etld in dpp_trans for etld in [".com", ".org", ".edu"])
                    ):
                        # special case: email or URL. Fallback to GPT result directly.
                        if self.log_debug:
                            print("Email or URL detected in DPP output: falling back to GPT result")
                        return gpt_trans, gpt_key

                    align_info, _, _ = align_utterance(
                        dpp_trans,
                        gpt_trans,
                        self.apply_gpt_dpp_fall_back_punctuation,
                        self.apply_gpt_dpp_fall_back_punctuation_to_remove,
                        self.apply_gpt_dpp_fall_back_enable_alignment_err_fallback,
                        self.apply_gpt_dpp_fall_back_alignment_err_threshold,
                        log_debug=self.log_debug,
                    )

                    if align_info is None:
                        # If align_utterance fails, align_info will be None, so use dpp_trans as-is
                        if self.log_debug:
                            print("Alignment failed")
                        return dpp_trans, dpp_key
                    updated_dpp_trans = " ".join(align_info["updated_dpp"])
                    if self.log_debug:
                        print(f"updated_dpp_trans: {updated_dpp_trans}")
                    return updated_dpp_trans, "display_dpp_GPT4o_punc_cap"

        return None, None

    def get_transcription_key(self):
        """Get the transcription key used in the last get_transcription call.

        Returns:
            transcription key (str)
        """
        return self.transcription_keys