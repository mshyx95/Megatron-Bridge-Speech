# pylint: disable=W0102,C0103,W0718

"""Crop a longform audio into training sampes"""
import logging
import copy
import re
import json
import numpy as np
from .audioutil import AudioUtil

LOG = logging.getLogger(__name__)


class GenWordSpeakerLabel:
    """A conversation cropper for cutting a long form audio into chunks for
    speaker diarization training.
    """
    def __init__(
        self,
        input_speaker_label_key="spkid",
        output_speaker_label_key="spkid",
        pattern = r'speaker\d+ ',
    ):
        self._input_speaker_label_key = input_speaker_label_key
        self._output_speaker_label_key = output_speaker_label_key
        self._pattern = pattern

    def __call__(self, examples):
        """This is the loop to generate chunked clips from long form audio."""
        if not isinstance(examples, list):
            examples = [examples]

        for example in examples:
            example[self._output_speaker_label_key] = self.gen_word_speaker_label(
                example[self._input_speaker_label_key]
            )

        return examples

    def split_by_speaker_markers(self, words, one_word_per_segment=False):
        """
        Splits a sequence of words into segments, each starting with a <SPK#> token.

        Args:
            words (list of str): A list of word strings, including <SPK#> markers.

        Returns:
            list of list of str: A list where each sublist is a segment starting with <SPK#>.
        """
        segments = []
        current_segment = []

        for word in words:
            if re.fullmatch(self.pattern, word):
                if current_segment:
                    if one_word_per_segment:
                        for w in current_segment[1:]:
                            segments.append([current_segment[0], w])
                    else:
                        segments.append(current_segment)
                current_segment = [word]
            else:
                current_segment.append(word)

        if current_segment:
            if one_word_per_segment:
                for w in current_segment[1:]:
                    segments.append([current_segment[0], w])
            else:
                segments.append(current_segment)

        return segments

    def gen_word_speaker_label(self, word_speaker_labels):
        """Generate word-level speaker labels from transcription with speaker tags."""

        # extract speaker IDs from transcription
        # segments = self.split_by_speaker_markers(transcription.split(), one_word_per_segment=True)
        # word_speaker_labels = [segment[0] for segment in segments]

        # sort the speaker IDs and map them to consecutive integers
        spk_mapping = {}
        for spk in word_speaker_labels:
            if spk not in spk_mapping:
                spk_mapping[spk] = len(spk_mapping)

        spk_label = [spk_mapping[seg] for seg in word_speaker_labels]
        spk_label2 = np.array(spk_label)

        return spk_label2[:, np.newaxis]
