# flake8: noqa
# pylint: skip-file
import numpy as np
import argparse
from typing import List, Union


class MSAudioFeaturizer:
    """
    This class is used to extract log mel spectrogram features for SVAD inference.
    please note this code is shared with the svad.py script by Khuram Shahid, this
    feature extraction here is the same as the function in
    cascades/data/block/logfbank_extractor.py. we should consider to further
    refactor the code here to avoid code duplication.
    """

    scale = 1 << 15
    n_mels = 80
    pre_emphasis = 0.97
    max_win_size = 400

    def __init__(self, sample_rate, device="cpu"):
        assert sample_rate in [16000, 8000]
        self.sample_rate = sample_rate
        self.device = device

        self.fft_size = 512
        self.n_fft_bins = self.fft_size // 2 + 1

        if sample_rate == 16000:
            self.n_samples = 400  # 25ms
            self.hop_len = 160  # 10ms
            self.n_fft = 512
        elif sample_rate == 8000:
            self.n_samples = 200  # 25ms
            self.hop_len = 80  # 10ms
            self.n_fft = 256

        assert self.n_samples <= self.max_win_size, "Window size too large"

        # 16kHzStaticStreamE14LFB80 impl uses a 512 FFT length but with an actual 400 sample signal, which hence gets padded with 0 to reach 512.
        # We extend the hamming window here to 512 with 0s to replicate that behavior in a moving-window stft
        # Note that this is not needed if you call torch.fft.fft directly with just the 400 samples.
        self.stft_window = np.hamming(self.n_samples)

        # Note that these are independent of input sample rate since for 8khz
        # we make the spectrum look like 16khz input audio by 0 padding before
        # applying filters
        self.filters = self._get_mel_filters_16khz()

    def log_mel_spectrogram(
        self, audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]
    ) -> np.ndarray:
        """
        Computes and returns log mel spectogram features for given audio after chunking the audio in to several windows

        Args:
            audio (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence of float audio samples.

        Returns:
            (log_mel_features, unused_samples)

            log_mel_features: (`np.ndarray`) of shape [audio_slices, self.n_mels] where audio_slices depends on the
            length of input audio.T

            unused_samples: (`int`) Number of samples that were not consumed at the end of the audio stream and should be
            sent again in a subsequent call if this is a streaming scenario. Can be ignored otherwise.

        """
        if len(audio) < self.n_samples:
            return np.empty((0, self.n_mels), np.float32), len(audio)

        # due to how border conditions are treated we cannot do preemphasis on the signal first and must
        # get the sliding window first

        audio_view = np.lib.stride_tricks.sliding_window_view(audio, self.n_samples)[:: self.hop_len]
        audio_slices = audio_view.copy()  # so we can modify

        audio_slices[:, 1:] = audio_slices[:, 1:] - self.pre_emphasis * audio_slices[:, :-1]
        audio_slices[:, 0] -= audio_slices[:, 0] * self.pre_emphasis
        audio_slices = audio_slices * self.scale

        stft = np.fft.rfft(audio_slices * self.stft_window, self.n_fft)

        magnitudes = np.square(np.abs(stft), dtype=np.float32)

        if magnitudes.shape[1] != self.n_fft_bins:
            # # Need to pad the output to look like 16 kHz data but with zeros in the 4 to 8 kHz bins.
            # # note that the final/high freq bin in the fft is also set to zero
            mag_pad = np.zeros((magnitudes.shape[0], self.n_fft_bins), dtype=magnitudes.dtype)
            mag_pad[:, : magnitudes.shape[1] - 1] = magnitudes[:, :-1]
            magnitudes = mag_pad

        mel_spec = self.filters @ magnitudes.T
        # note: following uses in place updates
        log_spec = np.log(np.clip(mel_spec, 1.0, np.inf, out=mel_spec), out=mel_spec)

        num_win = audio_view.shape[0]
        next_start = self.hop_len * num_win
        return log_spec.T, next_start

    def _get_mel_filters_16khz(self):
        LoFreqCutOff = 0
        HiFreqCutOff = 7690.608442
        sample_rate = 16000

        f2bin = lambda f: int((f * self.fft_size / sample_rate) + 0.5)
        mel = lambda f: 1127 * np.log(1.0 + f / 700.0)
        melk = lambda k: 1127 * np.log(1.0 + k * sample_rate / (self.fft_size * 700.0))

        # frequency cutoffs
        k_lo = 1
        k_hi = f2bin(HiFreqCutOff)

        # frequency to mel conversion table
        mel_k = melk(np.arange(self.n_fft_bins))

        # mel filter bank weights:
        m_lo = mel(LoFreqCutOff)
        m_hi = mel(HiFreqCutOff)
        freq_centers = np.linspace(m_lo, m_hi, self.n_mels + 2)
        weights = np.zeros((self.n_mels, self.n_fft_bins), dtype=np.float32)
        for m in range(self.n_mels):
            # mel band left boundary
            mlb = next(idx for idx, melk in enumerate(mel_k[k_lo:k_hi]) if melk > freq_centers[m]) + k_lo

            # mel band right boundary
            mrb = (
                k_hi
                - 1
                - next(idx for idx, melk in enumerate(mel_k[k_hi - 1 : k_lo - 1 : -1]) if melk < freq_centers[m + 2])
            )

            # compute the weights, non zero only within the band boundaries
            p = freq_centers[m]
            c = freq_centers[m + 1]
            k = np.arange(mlb, mrb + 1)
            tt = 1.0 - np.abs(c - mel_k[k]) / (c - p)
            weights[m, k] = tt

        return weights


def run_tests():
    sampling_rate = 16000
    fe = MSAudioFeaturizer(sampling_rate)
    dummy_wavform = np.random.rand(16000 * 30 + 37) * 2 - 1  # 37 extra samples at end to test a non-aligned case

    # Test 1 : sanity test
    _ = fe.log_mel_spectrogram(dummy_wavform)

    # Test 2 : check that processing 1 long stream gives same results as featurizing in chunks
    all_together, _ = fe.log_mel_spectrogram(dummy_wavform)
    parts = []
    remaining = np.empty(0, dtype=np.float32)
    chunk_size = 700
    for i in range(0, len(dummy_wavform), chunk_size):
        chunk = dummy_wavform[i : i + chunk_size]
        chunk = np.append(remaining, chunk)
        features, next_start = fe.log_mel_spectrogram(chunk)
        # print("in", type(chunk[0]))
        # print("out", type(view[0][0]))
        # print(i, i+chunk_size, "input", len(chunk),  "num features " , len(features), "start", next_start, "remainig", chunk.size - next_start)
        remaining = chunk[next_start:]
        parts.extend(features)

    parts = np.array(parts)
    assert np.allclose(parts, all_together)
    assert all_together.dtype == np.float32

    print("Test passed")


def ffmpeg_read_audio(filename, sample_rate=16000):
    import ffmpeg

    stream = ffmpeg.input(filename)
    stream = ffmpeg.output(stream, "pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
    out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=True)
    return np.frombuffer(out, np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio_file", help="Path to audio file")
    parser.add_argument(
        "-o",
        "--output_file",
        help="Path to output txt file where each line contains 1 feature vector with entries separated by tabs",
    )
    parser.add_argument("-t", "--test", help="Run tests", action="store_true")
    args = parser.parse_args()

    if args.audio_file:
        sampling_rate = 16000
        wavform = ffmpeg_read_audio(args.audio_file)
        fe = MSAudioFeaturizer(sampling_rate)
        features, _ = fe.log_mel_spectrogram(wavform)
        lines = [str.join("\t", [str(f) for f in feature]) for feature in features]
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(str.join("\n", lines))
        else:
            print(lines)
    elif args.test:
        run_tests()
