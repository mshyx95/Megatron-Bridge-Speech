# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Extract feature from audio for w2vbert encoder"""

import logging
import numpy as np
import scipy.fft
import scipy.signal

from .base_processor import BaseAudioProcessor

LOG = logging.getLogger(__name__)


def speechlib_mel(sample_rate, n_fft, n_mels, fmin=None, fmax=None):
    """Create a Mel filter-bank the same as SpeechLib FbankFC.

    Args:
        sample_rate (int): Sample rate in Hz. number > 0 [scalar]
        n_fft (int): FFT size. int > 0 [scalar]
        n_mel (int): Mel filter size. int > 0 [scalar]
        fmin (float): lowest frequency (in Hz). If None use 0.0.
            float >= 0 [scalar]
        fmax: highest frequency (in Hz). If None use sample_rate / 2.
            float >= 0 [scalar]

    Returns
        out (numpy.ndarray): Mel transform matrix
            [shape=(n_mels, 1 + n_fft/2)]
    """

    bank_width = int(n_fft // 2 + 1)
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0
    assert fmin >= 0, "fmin cannot be negtive"
    assert fmin < fmax <= sample_rate / 2, "fmax must be between (fmin, samplerate / 2]"

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f):
        return int((f * n_fft / sample_rate) + 0.5)

    # Spec 1: FFT bin range [f2bin(fmin) + 1, f2bin(fmax) - 1]
    klo = f2bin(fmin) + 1
    khi = f2bin(fmax)

    khi = max(khi, klo)

    # Spec 2: SpeechLib uses trianges in Mel space
    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)

    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(0, n_mels):
        left = m_centers[m]
        center = m_centers[m + 1]
        right = m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms

    return matrix


class LogFbankProcessor(BaseAudioProcessor):
    def __init__(self, eightk_method="resample"):

        if eightk_method in ("fillzero", "resample"):
            self._eightk_method = eightk_method
        else:
            raise ValueError(f"Invalid value for eightk_method parameter: {eightk_method}")

        self._mel = speechlib_mel(16000, 512, 80, fmin=None, fmax=7690).T

        self._hamming400 = np.hamming(400)  # for 16k audio
        self._hamming200 = np.hamming(200)  # for 8k audio

    def extract_spectrogram(self, wav, fs):
        """Extract spectrogram features from waveform.
        Args:
            wav (1D array): waveform of the input
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
            if self._eightk_method == "resample":
                # Input audio is 8 kHz. Convert to 16 kHz before feature
                # extraction
                wav = scipy.signal.resample_poly(wav, 2, 1)
                fs = 16000
            # Do nothing here for fillzero method
        elif fs != 16000:
            # Input audio is not a supported sample rate.
            raise RuntimeError(f"Input data using an unsupported sample rate: {fs}")

        preemphasis = 0.97

        if fs == 8000:
            n_fft = 256
            win_length = 200
            hop_length = 80
            fft_window = self._hamming200
        elif fs == 16000:
            n_fft = 512
            win_length = 400
            hop_length = 160
            fft_window = self._hamming400

        # Spec 1: SpeechLib cut remaining sample insufficient for a hop
        n_batch = (wav.shape[0] - win_length) // hop_length + 1

        # Efficient framing using stride_tricks (zero-copy view)
        wav_f32 = wav.astype(np.float32) if wav.dtype != np.float32 else np.ascontiguousarray(wav) if not wav.flags['C_CONTIGUOUS'] else wav
        y_frames = np.lib.stride_tricks.as_strided(
            wav_f32,
            shape=(n_batch, win_length),
            strides=(wav_f32.strides[0] * hop_length, wav_f32.strides[0]),
        )

        # Spec 2: SpeechLib applies preemphasis within each batch
        # Compute preemphasis directly from strided view (avoids full .copy())
        # result[0] = (frame[0] - 0.97*frame[0]) * 32768
        # result[i] = (frame[i] - 0.97*frame[i-1]) * 32768
        prev = np.empty((n_batch, win_length), dtype=np.float32)
        prev[:, 0] = y_frames[:, 0]
        prev[:, 1:] = y_frames[:, :-1]
        y_frames_preemp = (y_frames - preemphasis * prev) * 32768

        S = scipy.fft.rfft(fft_window * y_frames_preemp, n=n_fft, axis=1).astype(np.complex64)

        if fs == 8000:
            # Need to pad the output to look like 16 kHz data but with zeros in
            # the 4 to 8 kHz bins.
            frames, bins = S.shape
            padarray = np.zeros((frames, bins))
            S = np.concatenate((S[:, 0:-1], padarray), axis=1)  # Nyquist bin gets set to zero

        spec = np.abs(S).astype(np.float32)
        return spec

    def extract_features(self, wav, fs):
        """Extract log filterbank features from waveform.
        Args:
            wav (1D array): waveform of the input
            fs (int): sampling rate of the waveform, 16000 or 8000.
                If fs=8000, the waveform will be resampled to 16000Hz.
        Output:
            log_fbank (2D array): a TxD matrix of log Mel filterbank features.
                D=80, and T is the number of frames.
        """
        spec = self.extract_spectrogram(wav, fs)
        spec_power = spec**2

        fbank_power = np.clip(spec_power.dot(self._mel), 1.0, None)
        log_fbank = np.log(fbank_power).astype(np.float32)

        return log_fbank
