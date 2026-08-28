# pylint: disable=W0102,C0103,W0718

"""Crop a longform audio into training sampes"""
import os
import logging
import copy
import re
import json
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import html
import shutil
from .audioutil import AudioUtil
from .prepare_data.speaker_label_converter import SpeakerLabelConverter
LOG = logging.getLogger(__name__)
from typing import List, Optional, Tuple, Union


def sample_from_range(interval):
    """Sample a random integer from a given range."""
    assert interval[1] >= interval[0], "max of interval should not be smaller than min."
    return np.random.choice(list(range(int(interval[0]), int(interval[1]) + 1)))


TAG_RE = re.compile(r'<SPK_CHANGE>(.*?)</SPK_CHANGE>')          # capture the speaker name
SPLIT_RE = re.compile(r'<SPK_CHANGE>.*?</SPK_CHANGE>')            # split on speaker tags


def spk_change_to_spk_id(line: str, speaker_map: dict, last_spk_id: int = 1):
    """
    Yield (<tag>, word) tuples for one caption line and update speaker_map.
    """
    # split keeps the text *between* tags; finditer gives the tags themselves
    segments = SPLIT_RE.split(line)
    tags = TAG_RE.findall(line)

    if not tags:    # no speaker tags found, only words. Use the last speaker ID
        for word in line.split():
            yield last_spk_id, word
        return

    if segments[0]:  # text before the first speaker tag, e.g. "OK. <SPK_CHANGE>spk1</SPK_CHANGE> hello"
        for word in segments[0].strip().split():
            yield last_spk_id, word

    num_words_yielded = 0
    for seg_text, spk_name in zip(segments[1:], tags):   # skip leading '' segment
        spk_name = spk_name.strip()
        if spk_name not in speaker_map:                  # assign new ID
            speaker_map[spk_name] = len(speaker_map) + 1
        spk_id = speaker_map[spk_name]

        for word in seg_text.strip().split():
            yield spk_id, word
            num_words_yielded += 1

    if num_words_yielded == 0:  # no words in this segment, only speaker change tag, e.g. "<SPK_CHANGE>spk1</SPK_CHANGE>"
        spk_name = tags[-1].strip()     # # use the last speaker name
        if spk_name not in speaker_map:                  # assign new ID
            speaker_map[spk_name] = len(speaker_map) + 1
        yield speaker_map.get(spk_name, last_spk_id), ""


class Chunker:
    """Base class for chunking long form audio into short chunks."""
    def __init__(
            self,
            output_feat_key,
            output_audio_key,
            output_spkid_key,
            output_label_key,
            num_spk_range,
            gain_range,
            conv_duration_range,
            strict_duration,
            speaker_label_converter=None,
            speaker_label_decision_dir=None,
            profile_cfg=None,
    ):
        self._output_feat_key = output_feat_key
        self._output_audio_key = output_audio_key
        self._output_spkid_key = output_spkid_key
        self._output_label_key = output_label_key
        self._num_spk_range = num_spk_range
        self._gain_range = gain_range
        self._conv_duration_range = conv_duration_range
        self._strict_duration = strict_duration
        self._profile_cfg = profile_cfg

        if speaker_label_converter == "SpeakerLabelConverter":
            self._speaker_label_converter = SpeakerLabelConverter(speaker_label_decision_dir)
        else:
            self._speaker_label_converter = None

        if self._profile_cfg is not None:
            dump_dir = self._profile_cfg.get("debug_dump_dir", None)  # only used
            if dump_dir:
                print("Debug dump dir for profile chunker:", dump_dir)
                self._debug_cnt = 0
                if os.path.exists(dump_dir):
                    try:
                        shutil.rmtree(dump_dir)
                        LOG.info("Removed existing debug dump dir:", dump_dir)
                    except Exception as e:
                        LOG.info("Failed to remove existing debug dump dir:", dump_dir)
                        LOG.info(e)
                os.makedirs(dump_dir, exist_ok=True)

    def __call__(self):
        pass

    def _sort_spkid(self, curr_label):
        """Sort speaker ID according to their appearance time in the transcription."""
        spk_mapping = {}
        for seg in curr_label:
            if seg[2] not in spk_mapping:
                spk_mapping[seg[2]] = f"<SPK{len(spk_mapping) + 1}>"

        new_label = copy.deepcopy(curr_label)
        for seg in new_label:
            seg[2] = spk_mapping[seg[2]]

        spk_label = [seg[2] for seg in new_label]

        return new_label, spk_label

    def _gain_norm(self, wav):
        """Normalize the gain of the audio to a random value in the gain range.
        Also check the original gain of the audio, if it is too low, return None.
        """
        if wav is None or wav.size == 0:
            return None

        max_amplitude = np.max(np.abs(wav))
        if max_amplitude < 0.01:  # abnormally low amplitude, maybe it's empty signal
            return None

        # global gain perturbation
        if self._gain_range[0] <= 0:  # no gain perturbation
            return wav

        global_gain = np.random.uniform(self._gain_range[0], self._gain_range[1])
        try:
            wav = wav / max_amplitude * global_gain
        except Exception as e:  # pylint: disable=W0703
            LOG.info(e)
            LOG.info("global_gain: %f", global_gain)
            LOG.info("output_wav: %f", wav)
            LOG.info("np.max(np.abs(wav)): %f", np.max(np.abs(wav)))
            return None
        return wav


class ChunkerBySegment(Chunker):
    """Chunk long form audio by segments, which are usually words or sentences."""
    def __call__(self, wav, fs, fbanks, segments):
        if self._speaker_label_converter is not None:
            try:
                segments = self.gen_speaker_label(segments)
            except Exception as e:
                LOG.info(e)
                LOG.info("Error in generating speaker labels for segments. Skip this example.")
                return []
        segments_word_timing = self._generate_word_level_spkid_and_timing(segments)
        clips = self._sample_human_captions_by_segment(wav, fs, fbanks, segments_word_timing)

        return clips

    def gen_speaker_label(self, segments):
        """Convert speaker names to speaker IDs in the segments."""
        segments2 = ["\t".join([str(seg["start"]), str(seg["end"]), seg["original_words"]])
                     for seg in segments]
        segments_with_spkid = self._speaker_label_converter(segments2)
        # Change <SPK-*> to <SPK_CHANGE>*</SPK_CHANGE>
        final_segments = []
        for seg in segments_with_spkid:
            if len(seg.split("\t")) != 3:
                continue
            start, end, text = seg.split("\t")
            text = re.sub(r'<SPK-(.*?)>', r'<SPK_CHANGE>\1</SPK_CHANGE>', text)
            final_segments.append({
                "start": float(start),
                "end": float(end),
                "duration": float(end) - float(start),
                "word": text,
            })

        return final_segments

    def _generate_word_level_spkid_and_timing(self, segments):
        """ Replace speaker tags with unique speaker IDs.
        The speaker IDs start from 1, and are assigned in the order of appearance.
        Input:
        text: a string containing speaker tags

        Output:
        a string with speaker tags replaced by unique speaker IDs
        """
        speaker_map = {}
        for seg in segments:
            seg["word"] = html.unescape(seg["word"]).replace('\xa0', ' ')
            seg["word"] = seg["word"].replace("\n", " ").strip()
            # replace multiple spaces with a single space
            seg["word"] = re.sub(r'\s+', ' ', seg["word"])

        # remove non-alphanumeric characters at the beginning of the transcription before speaker change tag
        # e.g. ">><SPK_CHANGE>spk1</SPK_CHANGE> hello" -> "<SPK_CHANGE>spk1</SPK_CHANGE> hello"
        for seg in segments:
            sub_segments = SPLIT_RE.split(seg["word"])
            # if sub_segments[0] does not contain alphanumeric characters, e.g. spaces only, skip it
            if sub_segments[0] and not re.search(r'\w', sub_segments[0]):
                seg["word"] = seg["word"][len(sub_segments[0]):].strip()

        last_spk_id = 1
        new_segments = []
        for seg in segments:
            new_label = []
            label_with_spk_change = seg["word"]
            if not label_with_spk_change.startswith("<SPK_CHANGE>"):
                # if the label does not start with <SPK_CHANGE>, prepend a speaker change tag
                last_spk_name = [spk_name for spk_name, spk_id in speaker_map.items()
                                 if spk_id == last_spk_id]
                if not last_spk_name:  # only happens when the first line does not start with <SPK_CHANGE>.
                    # We should skip this audio as the speaker information is incomplete.
                    return []
                last_spk_name = last_spk_name[0]
                label_with_spk_change = \
                    f"<SPK_CHANGE>{last_spk_name}</SPK_CHANGE> " + label_with_spk_change

            for spk_id, word in spk_change_to_spk_id(label_with_spk_change, speaker_map, last_spk_id):
                new_label.append([spk_id, word])
                last_spk_id = spk_id

            start_time = seg["start"]
            avg_word_duration = (seg["end"] - start_time) / len(new_label)
            final_label = []
            for i, (spk_id, word) in enumerate(new_label):
                final_label.append([start_time + i * avg_word_duration,
                                    start_time + (i + 1) * avg_word_duration,
                                    f"<SPK{spk_id}>",
                                    word,
                                    ])
            new_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "word": final_label
            })
            # seg["word"] = final_label

        # debug
        # print("\nnew transcription with speaker ID tags:")
        # for seg in segments:
        #     spk_id = -1
        #     new_trans = []
        #     for word in seg["word"]:
        #         if word[2] != spk_id:
        #             spk_id = word[2]
        #             new_trans.append(spk_id)
        #         new_trans.append(word[3])
        #     print(" ".join(new_trans))

        # print("\noriginal transcription with speaker change tags:")
        # for seg in original_segments:
        #     print(seg["word"].replace("\n", " ").replace("</SPK_CHANGE>", " ").replace("<SPK_CHANGE>", "<SPK-"))

        # assert that the words are not changed during the conversion
        # all_words = [word[3] for seg in segments for word in seg["word"] if word[3]]
        # original_trans = " ".join([seg["word"] for seg in original_segments])
        # original_trans = TAG_RE.sub(' ', original_trans).strip()
        # all_words_original = original_trans.split()
        # assert all_words == all_words_original, \
        #     f"Transcription with speaker ID tags does not match original transcription.\n"

        # remove empty segments
        final_segments = []
        for seg in new_segments:
            words = [word for word in seg["word"] if word[3]]
            if words:
                final_segments.append(seg)

        return final_segments

    def _gen_new_clip_human_caption(self, clip_start_time, clip_end_time,
                                    wav, fs, fbanks, to_be_concatenated):
        """This function generates a new clip from the long form audio, which is
        a concatenation of several segments from the human caption.
        It also convert speaker change tags to speaker ID tags."""

        curr_label = [word for seg in to_be_concatenated for word in seg["word"]]
        curr_label, curr_spk_label = self._sort_spkid(curr_label)
        example_dict = {
            self._output_label_key: curr_label,
            self._output_spkid_key: curr_spk_label,
        }
        if wav is not None:
            output_wav = wav[int(clip_start_time * fs) : int(clip_end_time * fs)]
            example_dict[self._output_audio_key] = (output_wav, fs)
        if fbanks is not None:
            output_fbanks = fbanks[int(clip_start_time * 100) : int(clip_end_time * 100)]
            example_dict[self._output_feat_key] = output_fbanks

        if wav is not None and fbanks is not None:
            # make sure that the number of frames in fbanks matches the number
            # of frames computed from wav, because when both exists, the BatchToTensors block will
            # use the wav to compute the number of frames.
            frame_hop = int(fs / 100)
            num_frames = output_wav.size // frame_hop
            if output_fbanks.shape[0] > num_frames:
                output_fbanks = output_fbanks[:num_frames]
            elif output_fbanks.shape[0] < num_frames:
                # pad the fbanks with zeros to match the number of frames
                output_fbanks = np.pad(output_fbanks,
                                       ((0, num_frames - output_fbanks.shape[0]), (0, 0)),
                                       mode='constant', constant_values=0)

        return example_dict

    def _sample_human_captions_by_segment(
            self, wav, fs, fbanks, segments, max_margin=0.3, max_initial_sil=3,
    ):
        """
        This function samples clips from a long form audio. Each clip will be
        a training sample of diarization and contains one or more segments from the human caption.
            - If a human caption segment is longer than maximum allowed clip length, the segment
              is discarded. This is rare for human captions.
            - For the first clip, a random initial silence is added before the first segment when
              possible.
            - A small margin is added to the start and end time of each clip, when possible.
        """
        if wav is not None:
            session_duration = wav.size / fs
        else:
            session_duration = fbanks.shape[0] * 0.01
        max_super_segment_len = self._conv_duration_range[1]
        min_super_segment_len = self._conv_duration_range[2]

        clips = []
        clip_start_time = -1
        to_be_concatenated = []
        for seg_i, seg in enumerate(segments):
            margin = np.random.uniform(0, max_margin)
            # some a small margin to the segment start and end time
            seg_start_time = max(0, seg["start"] - margin)
            if seg_i > 0:
                # if this is not the first segment, we need to make sure
                # the start time is at least the end time of the previous segment
                seg_start_time = min(seg["start"], max(seg_start_time, segments[seg_i - 1]["end"]))
            seg_end_time = min(session_duration, seg["end"] + margin)
            if seg_i < len(segments) - 1:
                # if this is not the last segment, we need to make sure
                # the end time is at most the start time of the next segment
                seg_end_time = max(seg["end"], min(seg_end_time, segments[seg_i + 1]["start"]))

            # initialize clip_start_time if it is not set
            if clip_start_time < 0:
                if not clips:
                    # if this is the first clip, try to add a random initial silence
                    initial_silence = np.random.uniform(0, max_initial_sil)
                    clip_start_time = max(0, seg_start_time - initial_silence)
                    if seg_i > 0:
                        # if this is not the first segment, we need to make sure
                        # the clip_start_time is after the end time of the previous segment
                        # This could happen if the first segments are very long and discarded.
                        clip_start_time = max(clip_start_time, segments[seg_i - 1]["end"])
                else:
                    clip_start_time = seg_start_time

            if seg_end_time - clip_start_time < max_super_segment_len:
                # maximum super segment length is not reached yet, we can
                # concatenate this segment to the current super segment
                to_be_concatenated.append(seg)
                continue

            if to_be_concatenated:
                # try to save the current super segment
                clip_end_time = min(session_duration, to_be_concatenated[-1]["end"] + margin)
                if seg_i < len(segments) - 1:
                    # if this is not the last segment, we need to make sure
                    # the clip end time is at most the start time of the next segment
                    clip_end_time = max(to_be_concatenated[-1]["end"],
                                        min(clip_end_time, segments[seg_i]["start"]))

                if clip_end_time - clip_start_time < min_super_segment_len:
                    # this clip is too short, skip it
                    clip_start_time = -1
                    to_be_concatenated = []
                    continue

                # we have a super segment to be saved
                clips.append(self._gen_new_clip_human_caption(clip_start_time, clip_end_time,
                             wav, fs, fbanks, to_be_concatenated))

                # reset the start time and to_be_concatenated
                if seg_end_time - seg_start_time > max_super_segment_len:
                    # this segment is too long, we need to start a new super segment
                    clip_start_time = -1
                    to_be_concatenated = []
                else:
                    clip_start_time = seg_start_time
                    to_be_concatenated = [seg]
            else:
                # we are here because this segment is the first candidate segment for a new clip
                # and it is too long.
                clip_start_time = -1
                to_be_concatenated = []

        if to_be_concatenated:
            # we have some segments left to be sampled
            clip_end_time = min(session_duration, to_be_concatenated[-1]["end"] + margin)
            if clip_end_time - clip_start_time >= min_super_segment_len:
                clips.append(self._gen_new_clip_human_caption(clip_start_time, clip_end_time,
                             wav, fs, fbanks, to_be_concatenated))

        return clips


@dataclass
class Word:
    start_time: float
    end_time: float
    # duration: float
    speaker: str
    text: str

    def __post_init__(self):
        self.duration = self.end_time - self.start_time


class ProfileSegment:
    """A class to represent a speaker profile segment."""
    def __init__(self, start_time, end_time, speaker, words, wav=None, fs=16000, fbanks=None):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.speaker = speaker
        self.words = words
        self.wav = wav
        self.fs = fs
        self.fbanks = fbanks

    def extract_subsegment(self, max_duration):
        """Extract a subsegment from the profile segment with the given maximum duration."""
        # return self
        if self.duration <= max_duration:
            return self

        if not self.words:
            return None

        if len(self.words) == 1:
            if self.duration > max_duration:
                return None
            return self

        for i in range(len(self.words)):
            for j in range(i + 1, len(self.words)):
                if self.words[j].end_time - self.words[i].start_time > max_duration:
                    break
            subsegment_words = self.words[i:j]
            subsegment_start_time = subsegment_words[0].start_time
            subsegment_end_time = subsegment_words[-1].end_time

            if subsegment_end_time - subsegment_start_time <= max_duration:
                # get start and end time relative to the original segment
                relative_start_time = subsegment_start_time - self.start_time
                relative_end_time = subsegment_end_time - self.start_time

                if self.fbanks is not None:
                    fbanks_start_frame = int(relative_start_time * 100)
                    fbanks_end_frame = int(relative_end_time * 100)
                    subsegment_fbanks = self.fbanks[fbanks_start_frame:fbanks_end_frame]
                else:
                    subsegment_fbanks = None

                if self.wav is not None:
                    wav_start_sample = int(relative_start_time * self.fs)
                    wav_end_sample = int(relative_end_time * self.fs)
                    subsegment_wav = self.wav[wav_start_sample:wav_end_sample]
                else:
                    subsegment_wav = None

                return ProfileSegment(
                    start_time=subsegment_start_time,
                    end_time=subsegment_end_time,
                    speaker=self.speaker,
                    words=subsegment_words,
                    wav=subsegment_wav,
                    fs=self.fs,
                    fbanks=subsegment_fbanks,
                )
            else:
                continue
        # if no subsegment found, return None
        return None


def convert_words_to_profile_segments(words):
    """Convert a list of word dicts to a list of ProfileSegment, each segment contains only one word."""
    segments = []
    for word in words:
        word2 = Word(start_time=word["start"],
            end_time=word["end"],
            speaker=word["speaker"],
            text=word["word"]
        )
        segments.append(ProfileSegment(
            start_time=word["start"],
            end_time=word["end"],
            speaker=word["speaker"],
            words=[word2],
        ))
    return segments


class ProfileBuilder:
    """A class to build speaker profiles according to the profile configuration."""
    def __init__(self, profile_cfg,
                 wav, fs, fbanks, segments: List[ProfileSegment], time_resolution=0.02):
        self._profile_cfg = profile_cfg
        self._time_resolution = time_resolution
        self.spk_list = sorted(list({seg.speaker for seg in segments}))
        self.spk2vad = self._build_spk2vad(segments)
        self.spk2profiles = self._build_spk_profiles(wav, fs, fbanks, segments)

    def _build_spk2vad(self, segments):
        """Build speaker VAD according to the segments. It is useful to detect overlapping speech."""
        max_time = max([seg.end_time for seg in segments])
        max_frame = int(max_time / self._time_resolution) + 1
        spk2vad = np.zeros((len(self.spk_list), max_frame), dtype=np.bool8)
        for seg in segments:
            start_frame = int(seg.start_time / self._time_resolution)
            end_frame = int(seg.end_time / self._time_resolution)
            end_frame = max(end_frame, start_frame + 1)  # make sure at least one frame is marked as active
            spk2vad[self.spk_list.index(seg.speaker), start_frame:end_frame] = 1

        return spk2vad

    def has_overlaping_speech(self, spk, start_time, end_time):
        """Check if there is overlapping speech from other speakers in the given time range."""
        spk_idx = self.spk_list.index(spk)
        start_frame = int(start_time / self._time_resolution)
        end_frame = int(end_time / self._time_resolution)
        seg_vad = self.spk2vad[:, start_frame:end_frame]
        seg_vad[spk_idx, :] = 0  # ignore the target speaker

        return np.any(seg_vad)

    def has_speech(self, start_time, end_time):
        """Check if there is speech in the given time range."""
        start_frame = int(start_time / self._time_resolution)
        end_frame = int(end_time / self._time_resolution)
        seg_vad = self.spk2vad[:, start_frame:end_frame]

        return np.any(seg_vad)

    def _get_continuous_segments_of_speaker(self, spk, segs_of_spk, max_duration=10, max_sil_gap=1.0, min_num_words=3):
        """Merge segments of a speaker that are close enough to form continuous segments.
        Input:
            spk: speaker ID
            segs_of_spk: segments of a speaker
            max_sil_gap: maximum silence gap between two segments to be merged
            min_num_words: minimum number of words in a continuous segment
        """
        continuous_segs = []
        start_idx = 0
        for i in range(1, len(segs_of_spk)):
            sil_gap_satisfied = segs_of_spk[i].start_time - segs_of_spk[i - 1].end_time <= max_sil_gap
            max_duration_satisfied = segs_of_spk[i].end_time - segs_of_spk[start_idx].start_time <= max_duration
            if sil_gap_satisfied and max_duration_satisfied and not self.has_overlaping_speech(
                    spk, segs_of_spk[i-1].start_time, segs_of_spk[i].end_time):
                continue
            else:
                continuous_segs.append([start_idx, i - 1])
                start_idx = i

        if start_idx < len(segs_of_spk):
            continuous_segs.append([start_idx, len(segs_of_spk) - 1])

        # Filter out segments that do not meet the minimum number of words requirement
        continuous_segs = [seg for seg in continuous_segs
                           if sum(len(segs_of_spk[i].words) for i in range(seg[0], seg[1] + 1)) >= min_num_words]

        return continuous_segs

    def merge_segments(self, segments_to_merge, max_time, margin=0.1):
        """Merge a list of segments into one segment."""
        start_time = segments_to_merge[0].start_time
        end_time = segments_to_merge[-1].end_time

        # add a small margin to both ends of the merged segment if there is no speech in the margin area
        start_with_margin = max(0, start_time - margin)
        if not self.has_speech(start_with_margin, start_time):
            start_time = start_with_margin
        end_with_margin = min(end_time + margin, max_time)
        if not self.has_speech(end_time, end_with_margin):
            end_time = end_with_margin

        speaker = segments_to_merge[0].speaker
        words = []
        for seg in segments_to_merge:
            words.extend(seg.words)

        return ProfileSegment(start_time, end_time, speaker, words)

    def _build_spk_profiles(self, wav, fs, fbanks, segments):
        """Build speaker profiles according to the profile configuration.
        Each profile contains several utterances from the same speaker. And each utteranes contains
        its start time, end time, speaker ID, word transcription, and waveform or fbank features.
        """
        if wav is not None:
            max_time = wav.shape[0] / fs
        elif fbanks is not None:
            max_time = fbanks.shape[0] * 0.01
        else:
            max_time = max([seg.end_time for seg in segments])

        spk2profiles = {spk: [] for spk in self.spk_list}
        for spk in self.spk_list:
            segs_of_spk = [seg for seg in segments if seg.speaker == spk]
            continuous_segs = self._get_continuous_segments_of_speaker(
                spk, segs_of_spk, max_duration=self._profile_cfg["profile_duration_range"][1],
                max_sil_gap=1.0, min_num_words=3,
            )

            # only keep segments that are long enough
            valid_segments = []
            for seg in continuous_segs:
                end_time = segs_of_spk[seg[-1]].end_time
                start_time = segs_of_spk[seg[0]].start_time
                seg_duration =  end_time - start_time
                if seg_duration >= self._profile_cfg["profile_duration_range"][0]:
                    merged_segment = self.merge_segments(segs_of_spk[seg[0]:seg[1] + 1], max_time=max_time)
                    merged_segment.wav = wav[int(merged_segment.start_time * fs):int(merged_segment.end_time * fs)] if wav is not None else None
                    merged_segment.fs = fs
                    merged_segment.fbanks = fbanks[int(merged_segment.start_time * 100):int(merged_segment.end_time * 100)] if fbanks is not None else None
                    valid_segments.append(merged_segment)

            spk2profiles[spk] = valid_segments

        return spk2profiles

    def get_spk_profiles(self, spk, exclude_time_range=None):
        """Get speaker profiles for a given speaker.
        Exclude the profiles that overlap with the exclude_time_ranges.
        """
        candidates = self.spk2profiles.get(spk, [])

        # exclude segments that overlap with the exclude_time_range
        non_excluded_candidates = []
        for seg in candidates:
            if not exclude_time_range or (seg.end_time <= exclude_time_range[0] or seg.start_time >= exclude_time_range[1]):
                non_excluded_candidates.append(seg)

        if not non_excluded_candidates:
            return None
        chosen_profile = np.random.choice(non_excluded_candidates)
        valid_seg = chosen_profile.extract_subsegment(max_duration=self._profile_cfg["profile_duration_range"][1])

        return valid_seg


class ProfileApplier:
    """A class to apply speaker profiles to the training examples."""
    def __init__(self,
                 output_feat_key,
                 output_audio_key,
                 output_label_key,
                 output_spkid_key,
                 profile_cfg):
        self._output_feat_key = output_feat_key
        self._output_audio_key = output_audio_key
        self._output_label_key = output_label_key
        self._output_spkid_key = output_spkid_key
        self._profile_cfg = profile_cfg

    def apply(self, example: dict, profile_builder: ProfileBuilder, final_start_time: float):
        """Prepend speaker profiles to the segments according to the profile configuration."""
        words = example[self._output_label_key]
        profiles = []
        start_time = words[0][0] + final_start_time
        end_time = words[-1][1] + final_start_time
        local_spk_list = sorted(list({seg[2] for seg in words}))
        for spk in local_spk_list:
            spk_profile = profile_builder.get_spk_profiles(spk, exclude_time_range=(start_time, end_time))
            if spk_profile:
                profiles.append(spk_profile)

        num_spk_in_example = len(local_spk_list)
        num_spk_with_profile = len(profiles)

        # decide on the number of guest speakers
        num_guest_range = self._profile_cfg.get("num_guest_range", (0, 0))
        if num_guest_range[1] > 0:
            num_guest = sample_from_range(num_guest_range)
            num_profile_to_use = max(1, num_spk_in_example - num_guest)  # at least one profile should be used
            num_profile_to_use = min(num_profile_to_use, num_spk_with_profile)
            profiles_to_use = np.random.choice(profiles, num_profile_to_use, replace=False).tolist()
        else:
            profiles_to_use = profiles

        # decide on the number of no-show profiles
        num_noshow_profile_range = self._profile_cfg.get("num_noshow_profile_range", (0, 0))
        if num_noshow_profile_range[1] > 0:
            num_noshow_profile = sample_from_range(num_noshow_profile_range)
            available_noshow_spks = [spk for spk in profile_builder.spk_list
                                         if spk not in local_spk_list]
            noshow_profiles = [profile_builder.get_spk_profiles(spk, exclude_time_range=(start_time, end_time))
                               for spk in available_noshow_spks]
            noshow_profiles = [prof for prof in noshow_profiles if prof is not None]
            num_noshow_profile = min(num_noshow_profile, len(noshow_profiles))
            noshow_profiles_to_use = np.random.choice(noshow_profiles, num_noshow_profile, replace=False).tolist()
        else:
            noshow_profiles_to_use = []

        all_profiles = profiles_to_use + noshow_profiles_to_use
        np.random.shuffle(all_profiles)

        # normalize the time stamps in the profiles to start from zero
        for profile in all_profiles:
            time_offset = profile.start_time
            for word in profile.words:
                word.start_time -= time_offset
                word.end_time -= time_offset
            profile.start_time -= time_offset
            profile.end_time -= time_offset

        self._merge_profiles_to_example(example, all_profiles)

    def _merge_profiles_to_example(self, example: dict, profiles: List[ProfileSegment]):
        """Merge speaker profiles to the example dict."""
        if not profiles:
            return

        profile_trans = []
        time_offset = 0.0
        for p in profiles:
            for word in p.words:
                profile_trans.append([word.start_time + time_offset, word.end_time + time_offset, word.speaker, word.text])
            time_offset += p.duration

        wav_available = profiles[0].wav is not None if profiles else False
        wav_available = wav_available and example[self._output_audio_key][0] is not None
        fbank_available = profiles[0].fbanks is not None if profiles else False
        if not wav_available and not fbank_available:
            raise ValueError("No waveform or fbank features available in the profiles.")

        if fbank_available:
            fbank_durations = [p.fbanks.shape[0] * 0.01 for p in profiles]
            profile_fbanks = [p.fbanks for p in profiles]
            example[self._output_feat_key] = np.concatenate(profile_fbanks + [example[self._output_feat_key]], axis=0)

        if wav_available:
            wav_durations = [p.wav.size / p.fs for p in profiles]
            profile_wavs = [p.wav for p in profiles]
            example_wav, fs = example[self._output_audio_key]
            if profile_wavs[0].ndim != example_wav.ndim:
                raise ValueError("Dimension mismatch between profile wav and example wav.")

            concatenated_wav = np.concatenate(profile_wavs + [example_wav], axis=0)
            example[self._output_audio_key] = (concatenated_wav, fs)

        if fbank_available and wav_available:
            if abs(sum(fbank_durations) - max(0.01, sum(wav_durations))) > 1.0:
                print("Fbank and waveform durations do not match in the profiles.")
                print(f"Fbank durations: {fbank_durations}, total: {sum(fbank_durations)}")
                print(f"Wav durations: {wav_durations}, total: {sum(wav_durations)}")
                trans_to_print = [f"{w[0]:.2f} {w[1]:.2f} {w[2]} {w[3]}" for w in example[self._output_label_key]]
                print("Transcription =\n", "\n".join(trans_to_print))

        profile_duration = sum(fbank_durations) if fbank_available else sum(wav_durations)
        # update the time stamps in the example dict
        example[self._output_label_key] = [
            [round(seg[0] + profile_duration, 2),
             round(seg[1] + profile_duration, 2),
             seg[2],
             seg[3],
             ]
            for seg in example[self._output_label_key]
        ]
        # concatenate profile transcription to the front of the example transcription
        example[self._output_label_key] = profile_trans + example[self._output_label_key]
        example["profile_label"] = profile_trans
        example[self._output_spkid_key] = [seg[2] for seg in example[self._output_label_key]]


class ChunkerByWord(Chunker):
    """Chunk long form audio by words, which are usually the smallest unit in transcription."""
    def __call__(self, wav, fs, fbanks, segments):
        clips = self._sample_clips_by_word(wav, fs, fbanks, segments)
        return clips

    def _sample_clips_by_word(self, wav, fs, fbanks, segments):
        # this function samples clips from a long form audio. Each clip will be
        # a training sample of diarization.
        if wav is not None:
            session_duration = wav.size / fs
        else:
            session_duration = fbanks.shape[0] * 0.01
        min_super_segment_len = self._conv_duration_range[
            2
        ]  # this is the minimum duration of a clip. Usually, at the end of the
        # long form audio, the remaining part of the audio could be quite short.

        if self._profile_cfg is not None:
            apply_profile_prob = self._profile_cfg.get("apply_profile_prob", 0.0)
            profile_builder = ProfileBuilder(self._profile_cfg, wav, fs, fbanks,
                                             convert_words_to_profile_segments(segments))
            profile_applier = ProfileApplier(self._output_feat_key, self._output_audio_key, self._output_label_key,
                                             self._output_spkid_key, copy.deepcopy(self._profile_cfg))
        else:
            apply_profile_prob = 0.0

        super_segments = []
        used_duration = 0
        while True:
            conv_duration = sample_from_range(self._conv_duration_range[:2])
            conv_duration = np.minimum(session_duration - used_duration, conv_duration)
            if conv_duration >= min_super_segment_len:
                super_segments.append([used_duration, used_duration + conv_duration])
            elif session_duration - used_duration >= min_super_segment_len:
                continue # try again to sample a valid conv_duration
            else:
                break
            used_duration += conv_duration

        clips = []
        for start, end in super_segments:
            curr_label = []
            for seg_i, seg in enumerate(segments):  # a segment is usually a word, or a sentence
                start_time_within_super_segment = start <= seg["start"] and seg["start"] < end
                end_time_within_super_segment = start < seg["end"] and seg["end"] <= end
                if start_time_within_super_segment or end_time_within_super_segment:
                    curr_seg = [seg_i, seg["start"], seg["end"], seg["speaker"]]
                    if "word" in seg:
                        curr_seg.append(seg["word"])
                    curr_label.append(curr_seg)
            if len(curr_label) < 1:     # no word in this segment
                continue
            curr_nspk = len({seg[3] for seg in curr_label})
            if curr_nspk > self._num_spk_range[1]:
                continue

            final_start_time = min([s[1] for s in curr_label])
            final_start_time = min(final_start_time, start)
            final_start_time = max(0, final_start_time - 0.1)  # add a small margin
            if curr_label[0][0] > 0:    # make sure no overlap with previous word
                prev_seg_end_time = segments[curr_label[0][0] - 1]["end"]
                final_start_time = max(final_start_time, prev_seg_end_time)

            final_end_time = max([s[2] for s in curr_label])
            final_end_time = max(final_end_time, end)
            final_end_time = min(session_duration, final_end_time + 0.1)  # add a small margin
            if curr_label[-1][0] < len(segments) - 1:   # make sure no overlap with next word
                next_seg_start_time = segments[curr_label[-1][0] + 1]["start"]
                final_end_time = min(final_end_time, next_seg_start_time)

            if final_end_time - final_start_time < min_super_segment_len:
                continue

            # update curr_label so that the start and end time are relative to the start of the clip
            curr_label = [
                [np.maximum(0, seg[1] - final_start_time),
                 np.maximum(0, seg[2] - final_start_time),
                 ] + seg[3:]
                for seg in curr_label
            ]

            if wav is not None:
                output_wav = wav[int(final_start_time * fs) : int(final_end_time * fs)]
                output_wav = self._gain_norm(output_wav)
            else:
                output_wav = None

            if fbanks is not None:
                output_fbanks = fbanks[int(final_start_time * 100) : int(final_end_time * 100)]
            else:
                output_fbanks = None

            if output_fbanks is None and output_wav is None:
                continue  # skip this segment if both audio and fbanks are None

            example_dict = {
                self._output_label_key: curr_label,
                self._output_spkid_key: [seg[2] for seg in curr_label],
            }

            if wav is not None:
                example_dict[self._output_audio_key] = (output_wav, fs)
            if fbanks is not None:
                example_dict[self._output_feat_key] = output_fbanks

            if apply_profile_prob > 0 and np.random.rand() < apply_profile_prob:
                profile_applier.apply(example_dict, profile_builder, final_start_time)

            # sort the speaker IDs such that the first speaker is always "Spk1", the second is "Spk2", etc.
            sorted_label, sorted_spk_label = self._sort_spkid(example_dict[self._output_label_key])
            example_dict[self._output_label_key] = sorted_label
            example_dict[self._output_spkid_key] = sorted_spk_label

            try:
                self.debug_dump(example_dict)
            except Exception as e:
                LOG.info("Failed to dump debug example:", e)

            clips.append(example_dict)

        return clips

    def debug_dump(self, example: dict):
        """Dump the final audio and label for debugging."""
        # For debugging: dump final audio and label for manual checking
        if self._profile_cfg is None:
            return
        dump_dir = self._profile_cfg.get("debug_dump_dir", None)
        if dump_dir:
            if self._debug_cnt > 50:   # limit the number of dumped examples
                return
            # only dump 1% of the examples
            if np.random.rand() > 0.01:
                return

            rank = os.environ.get("RANK", "0")

            example_wav, fs = example[self._output_audio_key]
            if "profile_label" in example:
                dump_wav_path = os.path.join(dump_dir, f"debug_profile_example_rank{rank}_{self._debug_cnt}.wav")
            else:
                dump_wav_path = os.path.join(dump_dir, f"debug_example_rank{rank}_{self._debug_cnt}.wav")
            sf.write(dump_wav_path, example_wav, fs)

            dump_label_path = dump_wav_path.replace(".wav", ".txt")
            with open(dump_label_path, "w") as f:
                if "profile_label" in example:
                    for seg in example["profile_label"]:
                        f.write(f"{seg[0]:.2f}\t{seg[1]:.2f}\t{seg[2]}:{seg[3]}\n")
                    f.write(f"{seg[1]:.2f}\t{seg[1]:.2f}\END_OF_PROFILES\n")
                    num_profile_words = len(example["profile_label"])
                else:
                    num_profile_words = 0
                for seg in example[self._output_label_key][num_profile_words:]:
                    f.write(f"{seg[0]:.2f}\t{seg[1]:.2f}\t{seg[2]}:{seg[3]}\n")
            print(f"Dumped debug example to {dump_wav_path} and {dump_label_path}")
            self._debug_cnt += 1


class ConversationCropperCore:
    """A cropper class that chunk long form audios into short chunks for training."""
    def __init__(
        self,
        speaker_first=True,  # whether to put speaker ID before the word/segments in transcription
        word_level_spkid=False,    # whether to use word level speaker ID tags in transcription
        segmentation_unit="word",  # "word" or "segment", which unit to use for chunking
        num_spk_range=[1, 5],
        conv_duration_range=[10, 60, 5],    # min/max of duration distribution to sample from,
        # and min conversation duration (usually for last chunk)
        strict_duration=True,
        gain_range=[0.1, 1.0],
        chunk_sampling_prob_by_spk_cnt=[1, 1, 1, 1, 1, 1, 1, 1],  # probability of sampling a chunk
        speaker_label_converter=None,
        speaker_label_decision_dir=None,
        # with a given number of speakers.
        profile_cfg=None,
    ):
        self._speaker_first = speaker_first
        self._word_level_spkid = word_level_spkid
        self._segmentation_unit = segmentation_unit
        self._num_spk_range = num_spk_range
        self._conv_duration_range = conv_duration_range
        # self._strict_duration = strict_duration
        # self._gain_range = gain_range
        self._chunk_sampling_prob_by_spk_cnt = chunk_sampling_prob_by_spk_cnt

        self._audio_util = AudioUtil()
        self._chunker_by_segment = ChunkerBySegment(
            output_feat_key="feature",
            output_audio_key="audio",
            output_spkid_key="spkid",
            output_label_key="transcription",
            num_spk_range=num_spk_range,
            gain_range=gain_range,
            conv_duration_range=conv_duration_range,
            strict_duration=strict_duration,
            speaker_label_converter=speaker_label_converter,
            speaker_label_decision_dir=speaker_label_decision_dir,
            profile_cfg=profile_cfg,
        )
        self._chunker_by_word = ChunkerByWord(
            output_feat_key="feature",
            output_audio_key="audio",
            output_spkid_key="spkid",
            output_label_key="transcription",
            num_spk_range=num_spk_range,
            gain_range=gain_range,
            conv_duration_range=conv_duration_range,
            strict_duration=strict_duration,
            profile_cfg=profile_cfg,
        )

    def get_estimated_num_of_chunks(self, session_duration):
        """Estimate the number of chunks for a given session duration."""
        average_chunk_size = np.mean(self._conv_duration_range[:2])
        num_chunks = int(session_duration / average_chunk_size)

        # Adjust the number of chunks due to speaker count-based downsampling
        # Assume the probability of N-speakers in a chunk is proportional to
        # 1, 1/2, 1/4, 1/8, ... for N = 1, 2, 3, 4, ... This is a very rough estimate.
        if self._chunk_sampling_prob_by_spk_cnt:
            prob_by_num_spk = [1 / (2 ** i) for i in range(10)]
            cnt_selected = []
            for i in range(len(prob_by_num_spk)):
                if i < len(self._chunk_sampling_prob_by_spk_cnt):
                    cnt_selected.append(self._chunk_sampling_prob_by_spk_cnt[i] * prob_by_num_spk[i])
                else:
                    cnt_selected.append(prob_by_num_spk[i])
            total_selection_ratio = sum(cnt_selected) / sum(prob_by_num_spk)
            num_chunks = int(num_chunks * total_selection_ratio)

        return max(1, num_chunks)

    def __call__(self, input_audio, fbanks, segments, has_word_timing=False):
        """This is the main function to crop long form audio into short chunks."""
        wav, fs, fbanks, skip = self._verify_audio_feats(input_audio, fbanks)
        if skip:
            return []

        if has_word_timing:
            chunker = self._chunker_by_word
        else:
            chunker = self._chunker_by_segment

        clips = chunker(wav, fs, fbanks, segments)

        self._format_transcriptions(clips)

        valid_clips = [clip for clip in clips if self._is_valid_clip(clip)]

        return valid_clips

    def _format_transcriptions(self, clips):
        for clip in clips:
            transcription = " ".join([" ".join(word[2:]) for word in clip["transcription"]])
            clip["transcription"] = self.format_speaker_token(transcription)

    def _is_valid_clip(self, clip):
        is_valid = True

        num_speakers = len(set(clip["spkid"]))
        if num_speakers > self._num_spk_range[1]:
            is_valid = False

        if len(self._chunk_sampling_prob_by_spk_cnt) >= num_speakers and \
           self._chunk_sampling_prob_by_spk_cnt[num_speakers - 1] < np.random.rand():
            is_valid = False

        if not clip["transcription"].strip():
            is_valid = False

        return is_valid

    def _verify_audio_feats(self, input_audio, fbanks):
        """Get the audio and/or fbank features from the example."""
        if input_audio is not None:
            if isinstance(input_audio, tuple):
                wav, fs = input_audio
            else:
                wav, fs = self._audio_util.read_wavebytes(input_audio)
        else:
            wav, fs = None, None

        assert fbanks is not None or wav is not None, \
            "Both wav and fbanks are None. At least one of them needs to be present"

        skip = False
        if wav is not None and fbanks is not None:
            wav_duration = wav.size / fs
            feat_duration = fbanks.shape[0] / 100
            if abs(wav_duration - feat_duration) > 0.5:
                LOG.info(
                    "Wav duration (%.2fs) is different from fbank duration (%.2fs). Skip!",
                    wav_duration, feat_duration,
                )
                skip = True

        return wav, fs, fbanks, skip

    def format_speaker_token(self, text: str):
        """Format the transcription text by having desired speaker ID tags."""
        tokens = text.strip().split()

        segments = []
        current_spk = None
        current_words = []

        def _flush():
            if current_spk and current_words:
                if self._word_level_spkid:
                    if self._speaker_first:
                        segments.extend([f"{current_spk} {w}" for w in current_words])
                    else:
                        segments.extend([f"{w} {current_spk}" for w in current_words])
                else:
                    if self._speaker_first:
                        segments.append(f"{current_spk} {' '.join(current_words)}")
                    else:
                        segments.append(f"{' '.join(current_words)} {current_spk}")

        for token in tokens:
            if re.match(r"<SPK\d+>", token):
                if token != current_spk:
                    _flush()
                    current_spk = token
                    current_words = []
            else:
                current_words.append(token)
        _flush()  # flush the last segment

        content_token1 = [token for token in tokens if not re.match(r"<SPK\d+>", token)]
        content_token2 = [token for token in " ".join(segments).split()
                          if not re.match(r"<SPK\d+>", token)]
        if content_token1 != content_token2:
            LOG.warning("The transcription is changed while merging speaker segments.")
            LOG.warning("Original transcription: %s", ' '.join(content_token1))
            LOG.warning("Merged transcription: %s", ' '.join(content_token2))

        return " ".join(segments)


class ConversationCropper:
    """A conversation cropper for cutting a long form audio into chunks for
    speaker diarization training.
    """
    def __init__(
        self,
        input_audio_key="audio",
        input_feat_key="feature",
        input_label_key="info.segmentation_with_spkid",
        output_audio_key="audio",
        output_feat_key="feature",
        output_label_key="transcription",
        output_alternate_label_key="display_human_caption_adjusted_with_spk_id_tag",
        output_spkid_key="spkid",
        segmentation_fields=["info.segmentation_with_spkid", "info.Meta"],  # first found field used
        **kwargs
    ):
        super().__init__()
        self._input_audio_key = input_audio_key
        self._input_feat_key = input_feat_key
        self._input_label_key = input_label_key
        self._output_audio_key = output_audio_key
        self._output_feat_key = output_feat_key
        self._output_label_key = output_label_key
        self._output_alternate_label_key = output_alternate_label_key
        self._output_spkid_key = output_spkid_key
        self.segmentation_fields = segmentation_fields

        self.cropper = ConversationCropperCore(**kwargs)
        self._example_cnt = 0

    def get_estimated_num_of_chunks(self, session_duration):
        """Estimate the number of chunks for a given session duration."""
        return self.cropper.get_estimated_num_of_chunks(session_duration)

    def _get_cascaded_key(self, cascaded_key_str, example):
        terms = cascaded_key_str.split(".")
        v = example
        for i in range(len(terms) - 1):
            v = v[terms[i]]
        if terms[-1] in v:
            return v[terms[-1]]
        else:
            return None

    def _has_a_segmentation_field(self, example):
        """Check whether any segmentation field exist in current example."""
        apply = False
        for seg_field in self.segmentation_fields:
            if self._get_cascaded_key(seg_field, example) is not None:
                apply = True
                break

        return apply, seg_field

    def __call__(self, example):
        """This is the loop to generate chunked clips from long form audio."""
        has_seg_field, seg_field = self._has_a_segmentation_field(example)
        if not has_seg_field:
            return [example]

        input_audio = example.get(self._input_audio_key, None)
        fbanks = example.get(self._input_feat_key, None)
        clips = self._apply_chunking(input_audio, fbanks, example, seg_field)

        clips = [self._postprocess_clip(example, clip) for clip in clips]

        return clips

    def _apply_chunking(self, input_audio, fbanks, example, seg_field):
        """Apply chunking to long form audio."""
        seg_field_value = self._get_cascaded_key(seg_field, example)
        if seg_field == "info.segmentation_with_spkid":
            segments = [
                {
                    "start": seg[0],
                    "end": seg[1],
                    "duration": seg[1] - seg[0],
                    "word": seg[2],
                    "speaker": seg[3],
                }
                for seg in seg_field_value
            ]
            has_word_timing = True
        elif seg_field == "info.Meta":
            meta_data = json.loads(seg_field_value)
            segments = [
                {
                    "start": seg["time"][0],
                    "end": seg["time"][1],
                    "duration": seg["time"][1] - seg["time"][0],
                    "original_words": seg["transcription"],
                    "word": seg["original_transcription_with_spk_change"],
                }
                for seg in meta_data["segmentation"]
            ]
            has_word_timing = False
        clips = self.cropper(input_audio, fbanks, segments, has_word_timing)

        # dump transcription and wav for debug
        # out_dir = "debug/vox_simu"
        # os.makedirs(out_dir, exist_ok=True)
        # for clip in clips:
        #     out_file = "{}/example_{}.txt".format(out_dir, self._example_cnt)
        #     with open(out_file, "w") as f:
        #         f.write(clip["transcription"])
        #     wav, fs = clip["audio"]
        #     out_wav_file = "{}/example_{}.wav".format(out_dir, self._example_cnt)
        #     try:
        #         sf.write(out_wav_file, wav, fs)
        #     except Exception as e:  # pylint: disable=W0703
        #         LOG.info(e)
        #         LOG.info("fs: %d", fs)
        #         LOG.info("wav: %s", wav)
        #     self._example_cnt += 1

        return clips

    def _postprocess_clip(self, example, clip):
        """Postprocess a clip and decide whether to skip current clip."""
        # copy fields from example to clips
        new_clip = {}
        for key in ["language"]:
            if key in example:
                new_clip[key] = example[key]

        new_clip["info"] = {}
        for key in ["filename", "corpusname", "sessionid"]:
            if key in example["info"]:
                new_clip["info"][key] = example["info"][key]

        new_clip[self._output_label_key], new_clip[self._output_spkid_key] = self._replace_spkid(clip)
        if "audio" in clip:
            new_clip[self._output_audio_key] = clip["audio"]
        if "feature" in clip:
            new_clip[self._output_feat_key] = clip["feature"]

        return new_clip

    def _replace_spkid(self, clip):
        """Replace speaker ID tags from <SPK#> to speaker#, so we don't need to add
        special token for speaker ID"""
        transcription2 = re.sub(r"<SPK(\d+)>", r"speaker\1", clip["transcription"])
        speaker_id2 = [re.sub(r"<SPK(\d+)>", r"speaker\1", sid) for sid in clip["spkid"]]

        return transcription2.strip(), speaker_id2

    def _dump_clip_for_manual_checking(self, output_wav, fs, curr_label, clips):
        # pylint: disable=C0415
        """dump audio and label for manual checking for correctness"""
        import soundfile as sf

        dump_dir = "/mrfs/PRIVATE/spbarn/am_s3/users/xioxiao/temp/"
        sf.write(dump_dir + f"wav{len(clips)}.wav", output_wav, fs)
        lines = [
            "\t".join([f"{seg[0]:.3f}", f"{seg[1]:.3f}", str(seg[2]) + ":" + seg[3]])
            for seg in curr_label
        ]
        with open(dump_dir + f"label{len(clips)}.txt", "w", encoding="utf8") as file:
            for line in lines:
                file.write(line + "\n")
