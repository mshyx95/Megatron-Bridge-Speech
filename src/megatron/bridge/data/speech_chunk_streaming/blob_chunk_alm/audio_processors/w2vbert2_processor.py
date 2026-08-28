# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Extract feature from audio for w2vbert encoder"""

import numpy as np
import scipy.signal

from transformers.audio_utils import mel_filter_bank, spectrogram, window_function 
from .base_processor import BaseAudioProcessor


class W2vBERTProcessor(BaseAudioProcessor):
    def __init__(self):
        
        self._eightk_method = "resample"

        mel_filters = mel_filter_bank(
            num_frequency_bins=256,
            num_mel_filters=80,
            min_frequency=20,
            max_frequency=8000,
            sampling_rate=16000,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )

        self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))
        self.window = window_function(400, "povey", periodic=False)
        self.stride = 2
    
    def duration_to_frames(self, duration):
        """duration in s, estimated frames"""
        frame_rate = 10

        num_frames = duration * 1000 // frame_rate // self.stride
        return num_frames

    def extract_features(self, wav, fs):

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

        wav = wav * (2 ** 15)
        feature = spectrogram(
            wav,
            self.window,
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            center=False,
            preemphasis=0.97,
            mel_filters=self.mel_filters,
            log_mel="log",
            mel_floor=1.192092955078125e-07,
            remove_dc_offset=True,
        ).T

        feature = (feature - np.expand_dims(feature.mean(0), 0)) / np.sqrt(np.expand_dims(feature.var(0, ddof=1), 0) + 1e-7)
        
        num_frames, num_channels = feature.shape

        remainder = num_frames % self.stride
        if remainder != 0:
            n_pad = self.stride - remainder
            pad_to_stride = np.zeros((n_pad, num_channels))
            feature = np.vstack([feature, pad_to_stride])

        num_frames = feature.shape[0]
        feature = np.reshape(feature, (num_frames // self.stride, num_channels * self.stride))

        return feature
