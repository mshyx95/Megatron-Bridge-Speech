# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Union
import scipy.signal
import numpy as np
import torch
import torch.nn.functional as F

# Audio hyperparameters and functions used in Whisper
from whisper.audio import (
    N_FFT,
    HOP_LENGTH,
    pad_or_trim,
    mel_filters,
)

from .utils import bytes2wav

LOG = logging.getLogger(__name__)
# pylint: disable=too-many-arguments
# Audio hyperparameters used in Whisper
SAMPLE_RATE = 16000
N_MELS = 128
CHUNK_LENGTH = 30
# 480000: number of samples in a chunk
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE


def pad(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


def trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

    return array


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    filters: Union[str, np.ndarray, torch.Tensor],
):
    """
    Compute the log-Mel spectrogram of
    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or
        Tensor containing the audio waveform in 16 kHz
    filters: torch.tensor
        The number of Mel-frequency filters, only 80 is supported
    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        # This assume that audio samples are normalized [-1.0, 1.0]
        # np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        audio = torch.from_numpy(audio)

    window = torch.hann_window(N_FFT)
    stft = torch.stft(audio.float(), N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[:, :-1].abs() ** 2

    mel_spec = filters @ magnitudes.to(dtype=filters.dtype)

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class WhisperProcessor:
    """Extract the log fbank feature in the same configuration as Whisper

    Any 8 kHz input data will be upsampled to 16 kHz
    and then processed the same as 16 kHz data.

    Args:
        audio_key (str, optional): name of the key in the input data containing
            the audio data. Defaults to "audio".
        feat_key (str, optional): name of the key in the output data where the
            features will be put. Defaults to "feature".
        trim_or_pad (bool): pad/trim audio to fit 30 seconds.
            Defaults to False.
        feat_dim (int): number of mel filters.
            Defaults to 80.
    """

    def __init__(
        self,
        audio_key: str = "audio",
        feat_key: str = "feature",
        trim: bool = False,
        pad: bool = False,
        feat_dim: int = 128,
        pad_features_inside: int = False,
        pad_features_outside: int = False,
    ):
        """Initialize the Whisper feature extractor.
        Args:
            audio_key (str): name of the key in the input data containing the
                audio data.
            feat_key (str): name of the key in the output data where the
                features will be put.
            trim_or_pad (bool): pad/trim audio to fit 30 seconds.
            feat_dim (int): number of mel filters.
        """
        self._audio_key = audio_key
        self._feat_key = feat_key
        self.trim = trim
        self.pad = pad
        self.feat_dim = feat_dim
        self.filters = None  # This will be initialized with mel_filters()
        # pad to 30s for encoder forward and LLM attend to all frames
        self.pad_to_thirty_seconds = pad_features_inside
        # pad to 30s for encoder forward and LLM attend to real frames only
        self.pad_to_thirty_seconds_encoder_only = pad_features_outside
        assert (pad_features_inside and pad_features_outside) != True, "don't pad twice"

    def _decode_waveform(self, data):
        if isinstance(data, tuple):
            return data

        return bytes2wav(data)

    def duration_to_frames(self, duration):
        """duration in s, estimated frames"""
        frame_rate = 10

        num_frames = duration * 1000 // frame_rate
        return num_frames

    def __next__(self):
        _input = self.next_input()
        if isinstance(self._audio_key, list):
            assert isinstance(self._feat_key, list) and len(self._audio_key) == len(
                self._feat_key
            ), (
                "If self._audio_key is a list, "
                + "self._feat_key should be a list with the same size."
            )

            for audio_key, feat_key in zip(self._audio_key, self._feat_key):
                self._process_key(_input, audio_key, feat_key)
        else:
            assert isinstance(self._feat_key, str) and isinstance(self._audio_key, str), (
                "self._audio_key and self._feat_key " + "should be string or list of strings."
            )

            self._process_key(_input, self._audio_key, self._feat_key)

        return _input

    def _process_key(self, data, audio_key, feat_key):
        if feat_key in data:
            return
        input_audio = data[audio_key]
        if isinstance(input_audio, list):
            data[feat_key] = [
                self.extract_features(*self._decode_waveform(audio)) for audio in input_audio
            ]
        else:
            wav, fs = self._decode_waveform(input_audio)
            data[feat_key] = self.extract_features(wav, fs)

    def extract_features(self, wav, fs):
        """Extract log filterbank features from waveform.
        Args:
            wav (1D array): waveform input normalized within [-1, 1]
            fs (int): sampling rate of the waveform, 16000 or 8000.
                If fs=8000, the waveform will be resampled to 16000Hz.
        Output:
            log_fbank (2D array): a TxD matrix of log Mel filterbank features.
                D=80, and T is the number of frames.
        """
        if wav.ndim > 1:
            wav = np.squeeze(wav)
        
        # by default, we extract the mean if stereo
        if len(wav.shape) == 2:
            wav = wav.mean(1)

        # Resample to 16000 or 8000 if needed
        if fs > 16000:
            wav = scipy.signal.resample_poly(wav, 16000, fs)
            fs = 16000
        elif 8000 < fs < 16000:
            wav = scipy.signal.resample_poly(wav, 8000, fs)
            fs = 8000
        elif fs < 8000:
            raise RuntimeError(f"Unsupported sample rate {fs}")

        if fs == 8000:
            # Input audio is 8 kHz. Convert to 16 kHz before feature
            # extraction
            wav = scipy.signal.resample_poly(wav, 2, 1)
            fs = 16000
            # Do nothing here for fillzero method

        assert fs == 16000, "Invalid sampling rate: {fs}"
        # Note that Whisper does not do pre-emphasis on the raw audio
        if self.trim:
            wav = trim(wav)
        if self.pad:
            wav = pad(wav)
        if self.filters is None:
            self.filters = mel_filters(torch.device("cpu"), self.feat_dim)
        log_spec = log_mel_spectrogram(wav, self.filters)

        if self.pad_to_thirty_seconds:
            padding = CHUNK_LENGTH * 100 - log_spec.shape[1]
            if padding > 0:
                log_spec = F.pad(log_spec, (0, padding), "constant", 0)

        return log_spec.transpose(0, 1).numpy()

    def pad_to_chunk_length(self, feature):
        # pad features after obtaining audio_embed_sizes in dataset
        # in this case, the padded frames are used for encoder forward only
        # The LLM decode will use the real frames only
        
        if feature.shape[0] == N_MELS:
            feature = feature.transpose()
        
        padding = CHUNK_LENGTH * 100 - feature.shape[0]
        
        if padding > 0:
            feature = np.pad(feature, ((0, padding), (0, 0)), "constant")

        return feature
