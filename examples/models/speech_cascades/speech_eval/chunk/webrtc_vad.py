"""chunk audio using webrtcvad"""

import collections
import contextlib
import logging
import sys
import wave
from os import path

import webrtcvad


# pylint: disable=logging-fstring-interpolation,redefined-builtin
# pylint: disable=undefined-loop-variable
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.WARNING)


def read_wave(wav_file):
    """Reads wave file and returns PCM audio data and sample rate."""
    with contextlib.closing(wave.open(wav_file, "rb")) as f:
        num_channels = f.getnchannels()
        assert num_channels == 1
        sample_width = f.getsampwidth()
        assert sample_width == 2
        sr = f.getframerate()
        assert sr in (8000, 16000, 32000, 48000)
        pcm_data = f.readframes(f.getnframes())
        return pcm_data, sr


def write_wave(wav_file, audio, sr):
    """Writes wave file from PCM audio data, and sample rate."""
    with contextlib.closing(wave.open(wav_file, "wb")) as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(audio)


class Frame:
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_ms, audio, sr):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sr * (frame_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sr) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sr, frame_ms, padding_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sr - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_ms / frame_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sr)

        LOG.info("1" if is_speech else "0")
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                LOG.info(f"+({ring_buffer[0][0].timestamp:.4f})")
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, _ in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                LOG.info(f"-({frame.timestamp + frame.duration:.4f})")
                triggered = False
                # yield b"".join([f.bytes for f in voiced_frames])
                yield (
                    voiced_frames[0].timestamp,
                    voiced_frames[-1].timestamp + voiced_frames[-1].duration,
                )
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        LOG.info(f"-({frame.timestamp + frame.duration:4f})")
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield (voiced_frames[0].timestamp, voiced_frames[-1].timestamp + voiced_frames[-1].duration)
        # yield b"".join([f.bytes for f in voiced_frames])


def webrtc_chunk(wav_file, mode=3):
    """segment wav file using webrtcvad"""
    if not path.exists(wav_file):
        return None
    audio, sr = read_wave(wav_file)
    max_len = len(audio) / sr
    segments = WebRtcChunker(30, mode=mode).chunk(audio, sr)
    return segments, max_len


class WebRtcChunker:
    """Chunker using WebRTC VAD"""

    def __init__(self, max_len, max_sil=1, long_sil=5, buf_len=0.03, mode=3):
        self.mode = mode if 3 >= mode >= 0 else None
        self.max_len = max_len
        self.buf_len = buf_len
        self.max_sil = max_sil
        self.long_sil = long_sil

    def chunk(self, audio, sr=None):
        """chunk audio data"""
        if isinstance(audio, str):
            audio, sr = read_wave(audio)
        audio_len = len(audio) / sr / 2  # due to 16bit
        if audio_len <= self.max_len:
            return [(0, audio_len)]
        vad = webrtcvad.Vad(self.mode)
        frames = frame_generator(10, audio, sr)
        segments = list(vad_collector(sr, 10, self.buf_len * 1000, vad, frames))
        if len(segments) == 0 or segments is None:
            return [(0, audio_len)]
        return self.extend_segments(segments, audio_len)

    def extend_segments(self, segments, max_len):
        """extend segments to max_len"""
        s_0 = segments[0]
        ext_segs = [[max(0, s_0[0] - self.max_sil), s_0[1]]]
        for s, e in segments[1:]:
            sil_len = max(0, s - ext_segs[-1][1])
            seg_len = e - ext_segs[-1][0]
            if sil_len >= self.long_sil or seg_len >= self.max_len:
                side_sil = min(self.max_sil, sil_len)
                ext_segs[-1][1] += side_sil
                ext_segs.append([s - side_sil, e])
            else:
                ext_segs[-1][1] = e
        ext_segs[-1][1] = min(max_len, ext_segs[-1][1] + self.max_sil)
        return ext_segs


def main(args):
    """main entry point"""
    if len(args) != 2:
        sys.stderr.write("Usage: example.py <aggressiveness> <path to wav file>\n")
        sys.exit(1)
    segments, _ = webrtc_chunk(args[1], mode=args[0])
    for i, segment in enumerate(segments):
        print(f"Segment {i}: {segment[0]:.2f} - {segment[1]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
