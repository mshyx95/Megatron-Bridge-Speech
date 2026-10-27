"""chunk audio into segments"""

# pylint: disable=redefined-builtin,relative-beyond-top-level,logging-fstring-interpolation
import json
import logging

from whisper import load_audio

from .svad import SVadChunker
from .webrtc_vad import WebRtcChunker


# from webrtc_vad import webrtc_segment
LOG = logging.getLogger(__name__)


class FixedWindowChunker:
    """chunk audio into segments with a non-overlapping sliding window of fixed length"""

    def __init__(self, max_len_sec=30):
        self.max_len_sec = max_len_sec

    def chunk(self, audio, sr):
        """chunk audio into segments"""
        max_len = self.max_len_sec * sr
        segments = []
        start = 0
        while start < len(audio):
            end = min(start + max_len, len(audio))
            segments.append((start / sr, end / sr))
            start = end
        return segments


class ExternalChunker:
    """chunk audio into segments according to external segmentations"""

    def __init__(self, segmentation_json_file):
        self.segmentation_json_file = segmentation_json_file
        with open(self.segmentation_json_file, "r") as f:
            self.session2segmentation = json.load(f)

    def chunk(self, audio_file):
        """chunk audio into segments"""
        # locate the session for the audio file
        target_session = None
        for session in self.session2segmentation:
            if session in audio_file:
                target_session = session
                break

        if target_session is None:
            raise ValueError(f"session not found for audio file: {audio_file}")

        segments = [(seg["start_time"], seg["end_time"]) for seg in self.session2segmentation[target_session]]

        return segments


class Chunker:
    """a chunker class to wrap different chunking methods"""

    def __init__(self, max_len, type="svad", **args):
        self.max_len = max_len
        self.type = type.lower()
        self._chunker = None

        self._chunker = None
        if type == "none":
            pass
        elif type == "fixed":
            self._chunker = FixedWindowChunker(max_len_sec=max_len)
        elif type == "svad":
            self._chunker = SVadChunker(max_len_sec=max_len, **args)
        elif type == "webrtc":
            self._chunker = WebRtcChunker(max_len, mode=args.get("chunk_mode", 0))
        elif type == "external":
            self._chunker = ExternalChunker(args["vad_file"])
        else:
            raise ValueError(f"unknown chunker type: {type}")

    def __call__(self, audio_file):
        """chunk audio into segments"""
        segments = self._chunk(audio_file)
        seg_str = ", ".join([f"{s[0]:.2f}-{s[1]:.2f}" for s in segments])
        # print(f"chunked {len(segments)} segments: {seg_str}")
        LOG.info(f"chunked {len(segments)} segments: {seg_str}")

        large_seg_str = ", ".join([f"{s:.2f}-{e:.2f}" for s, e in segments if e - s > 30])
        if large_seg_str:
            # print(f"large segments: {large_seg_str}")
            LOG.info(f"large segments: {large_seg_str}")
        return segments

    def _chunk(self, audio_file):
        """chunk audio into segments"""
        if isinstance(self._chunker, WebRtcChunker):
            return self._chunker.chunk(audio_file)
        # the load_audio function will try to upsample the audio to 16k by default.
        # so that the sr is always 16k
        audio, sr = load_audio(audio_file), 16000
        if self._chunker is None:
            return [(0, len(audio) / sr)]
        if isinstance(self._chunker, ExternalChunker):
            return self._chunker.chunk(audio_file)
        return self._chunker.chunk(audio, sr)
