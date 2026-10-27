# noqa: E402
# flake8: noqa
# pylint: skip-file
#
import math
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from .featurizer import MSAudioFeaturizer


class SVadChunker:
    """
    Identifies segments in audio after finding vad segments and combining greedily to configurable (e.g. 30 second) chunks
    Uses Microsoft Smart VAD

    Please note that this part SVAD code is shared from Khuram Shahid, confirmed that this is modified based on the code from
    https://msasg.visualstudio.com/Bing_and_IPG/_git/speech_transducers?version=GBakash/guoye/datafilt-ictc-longform-s2s_eval_refactor&_a=contents&path=/speech_transducers/data/prepare_eval/vad/vad_utils.py


    Determines segment boundaries for decoding a wav file by running a VAD and processing its raw output. This allows to
    (i) reject large silence/noise to reduce decoding cost (ii) split arbitrary length wav files into model-appropriate segments

    It runs in 2 stages - one to mark candidate silence_points while running the VAD, and one to refine and create final segments.
        1. `mark_silence_points`: processes VAD speech probability (`spProb`) using hysteresis thresholding (uses two onset/offset thresholds).
            It returns a list of `[(start, duration)]` silence_points used as candidates to segment. This step can be implemented
            in streaming L->R fashion while Pasco is reading and processing as stream of VAD frames.
        2. `buffer_segments`: refines silence_points down into smaller list of `[(start, end)]` final_segments that aims to create as long
            segments as possible, while rejecting obvious silence/noise regions. this is implemented by different methods, depdending on the `version`

    The `vad-v0` baseline strategy only rejects obvious silence/noise regions, while splitting remaining segments into max_segment_len fixed-length windows
    The `vad-v1` strategy rejects obvious silence/noise regions, and uses a greedy approach to create as long as possible segments that are always bounded
    by silence_points avoiding any mid-speech segmentation. It yields similar / better accuracy than an oracle segmentation from a hybrid 1st pass model.

    There are two sets of hyperparameters - 3 used by mark_silence_points, 4 used by buffer_segments. Details:
        5/7 *_len parameters are physical time values. We got good WER after setting conservative values without tuning
            max_segment_len gives best results when set to the largest size S2S model is trained on
            min_sil_region_len may reduce cost by rejecting ~10-15% more silence if reduced upto 1/2s, but will create shorter segments
            min_[sil_point/segment]_len are set to small values that we expect to consider meaningful enough for speech
            min_buffer_len is likely an insignificant param, just haven't explicitly evaluated its impact
        2/7 _threshold parameters are set after some exploration of the specific VAD prob distribution on some examples
            could probably be tuned further for heavy noise, else these defaults worked well on most dev/tuning sets

    """

    SVAD_ONNX_BLOB_PATH = os.environ.get("SPEECH_SVAD_MODEL", "")

    def __init__(self, max_len_sec, verbose=False, onnx_model_path=None, **config):
        self.verbose = verbose

        if onnx_model_path is None:
            if os.path.exists(self.SVAD_ONNX_BLOB_PATH):
                onnx_model_path = self.SVAD_ONNX_BLOB_PATH
            else:
                onnx_model_path = str(Path(__file__).parent / "resource/svad_seq_quant.onnx")

        self.seq_session = self._get_session(onnx_model_path)
        self.ort_initializer_names = [
            x.name for x in self.seq_session.get_overridable_initializers() if "PastValue" in x.name
        ]
        self.ort_out_prob_idx = [
            i for i, x in enumerate(self.seq_session.get_outputs()) if x.name == "Plus5184_Output_0_attach_noop_"
        ][0]

        self.featurizer_16k = MSAudioFeaturizer(16000)
        self.featurizer_8k = MSAudioFeaturizer(8000)

        self.smooth_probs = True
        self.smooth_window = config["svad_smooth_window"] if "svad_smooth_window" in config else 5
        self.version = config["svad_version"] if "svad_version" in config else "vad-v1"

        # used by mark_silence_points
        self.sil_onset_threshold = (
            config["svad_sil_onset_threshold"] if "svad_sil_onset_threshold" in config else 0.05,
        )
        self.sil_offset_threshold = (
            config["svad_sil_offset_threshold"] if "svad_sil_offset_threshold" in config else 0.25,
        )
        self.min_sil_point_len = (
            config["svad_min_sil_point_len_ms"] if "svad_min_sil_point_len_ms" in config else 100
        ) / 10  # 100ms

        # used by buffer_segments
        self.min_sil_region_len = (
            config["svad_min_sil_region_len_ms"] if "svad_min_sil_region_len_ms" in config else 5000
        ) / 10  # 5 seconds
        self.max_segment_len = int(max_len_sec * 100)
        self.min_segment_len = 10  # 100ms
        self.min_buffer_len = 1000  # this is likely an unnecessary param, just haven't explicitly tested its impact
        self.leading_collar = 100
        self.trailing_collar = 100  # NOTE: this is not implemented yet

        self.step_ms = 10.0  # each frame is 10ms and we advance by 1 frame at a time
        self.fn_print = lambda *args: print(*args) if verbose else None

    def chunk(self, data, sr):
        """get chunks of audio data from a wav file"""
        spans = self._get_vad_segments(data, sr)
        self.fn_print(" VAD merged spans:", spans)
        return spans

    def _moving_average(self, x, w, na_fill_value=0):
        # Notes this returns the same length as the input with boundary effects (edges clamped to na_fill_value)
        avg = np.convolve(x, np.ones(w), "same") / w

        # set invalid values to 0
        avg[np.isnan(avg)] = na_fill_value
        avg[: w // 2] = na_fill_value
        avg[-w // 2 :] = na_fill_value
        return avg

    def _get_vad_segments(self, wavform, sample_rate=16000):
        """
        segments the audio in to chunks using a VAD. For any segments that are longer than max_len_sec,
        it will do a second round of thresholding on the VAD score to sub-chunk to multiple segments
        returns [(start_sec, end_sec)]
        """
        if len(wavform) <= self.max_segment_len / 100 * sample_rate:  # max_segment_len in units of 10ms
            return [(0, len(wavform) / sample_rate)]

        features_view = self._get_vad_audio_features_view(wavform, sample_rate)

        vad_output = self._run_sequence_batched_vad(
            self.seq_session, features_view
        )  # returns [('nonSpProb', 'spProb', 'eosProb')]
        spProb = [x[1] for x in vad_output]

        if self.smooth_probs:
            spProb = self._moving_average(spProb, self.smooth_window)

        silence_points = self._mark_silence_points(spProb)
        assert len(silence_points) > 0, "Silence points must contain min 1 point marking the end"

        final_segments = self._greedy_buffer_segments(silence_points)
        self._assert_monotonic(final_segments)

        # Logging only
        if self.verbose:
            n_silence_frames = sum(x[1] for x in silence_points)
            n_frames = len(spProb)
            n_final_frames = sum((x[1] - x[0]) for x in final_segments)
            self.fn_print(
                f"Marked {len(silence_points)} silence points with total silence frames: {n_silence_frames} .."
            )
            self.fn_print(
                f"Could reject {n_silence_frames}/{n_frames} silence frames i.e. {n_silence_frames / n_frames * 100:.1f}%"
            )
            self.fn_print(f"Buffered into {len(final_segments)} segments")
            self.fn_print(
                f"Actually rejected {n_frames - n_final_frames}/{n_frames} silence frames i.e. {(n_frames - n_final_frames) / n_frames * 100:.1f}%"
            )

        segments = [(x[0] * self.step_ms / 1000, (x[1] - 1) * self.step_ms / 1000) for x in final_segments]

        return segments

    def _get_session(self, onnx_file):
        import psutil

        ort_ep = "CPUExecutionProvider"  # "CUDAExecutionProvider"
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        opts.inter_op_num_threads = math.ceil(psutil.cpu_count(logical=False) / 2)
        opts.intra_op_num_threads = math.ceil(psutil.cpu_count(logical=False) / 2)
        seq_session = ort.InferenceSession(onnx_file, opts, providers=[ort_ep])
        return seq_session

    def _get_vad_audio_features_view(self, wavform, sample_rate):
        featurizer = (
            self.featurizer_16k if sample_rate == 16000 else self.featurizer_8k if sample_rate == 8000 else None
        )
        features, _ = featurizer.log_mel_spectrogram(wavform)
        hop_len = 1
        num_frames = 11
        features_view = np.lib.stride_tricks.sliding_window_view(features, num_frames, axis=0)[::hop_len]
        features_view = np.swapaxes(features_view, 1, 2)
        return features_view

    def _run_sequence_batched_vad(self, seq_session, features_view, step_size=64):
        def softmax(x, axis=0):
            return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

        # Matching S2S settings:
        reset_eos_prob_threshold = 0.7
        reset_eos_time_threshold = 3  # aka vad-reset-smooth-frames
        reset_sil_threshold = 0  # Note default in code is 0.05
        reset_time_threshold = 3  # Note default in code is 9000

        def should_reset(vadOutputs):
            if len(vadOutputs) < reset_time_threshold:
                return False
            totalEosProb = 0.0
            totalSilProb = 0.0
            for i in range(len(vadOutputs) - reset_eos_time_threshold, len(vadOutputs)):
                totalEosProb += vadOutputs[i][2]  # 2 = eosProb
                totalSilProb += vadOutputs[i][0]  # 0 = nonSpeechProb
            return (
                totalEosProb / reset_eos_time_threshold > reset_eos_prob_threshold
                or totalSilProb / reset_eos_time_threshold < reset_sil_threshold
            )

        results_seq_batch = []
        inputs = {}

        i = 0
        while i < len(features_view):
            input_batch = features_view[i : i + step_size, :]
            inputs["features"] = np.expand_dims(
                input_batch.reshape(-1, 880), 1
            )  # TODO: replace 880 with computed value

            out = seq_session.run(None, inputs)
            reset = False
            for r in softmax(out[self.ort_out_prob_idx].squeeze(1), axis=1):
                results_seq_batch.append(r.reshape(-1).tolist())
                i += 1
                if should_reset(results_seq_batch):
                    reset = True
                    break

            if reset:
                # print("Reset at ", i)
                inputs.clear()
            else:
                for i2, n in enumerate(self.ort_initializer_names):
                    inputs[n] = out[i2]

        # results_seq_batch = np.concatenate(results_seq_batch, axis=0)

        return results_seq_batch

    def _mark_silence_points(self, spProb):
        """
        Iterates through VAD speech probabilities `spProb` in L->R way marking the start and
        duration of pauses/silences longer than self.min_sil_point_len frames (defaults to 100ms).

        We mark regions with low `spProb` using two onset/offset thresholds, based on hysteresis thresholding
        references:    https://scikit-image.org/docs/stable/auto_examples/filters/plot_hysteresis.html
                       https://github.com/pyannote/pyannote-audio/issues/82

        Onset/offset threshold logic:
            listen for when spProb goes below onset threshold and make silence region active
            end silence region and add point when spProb increases above offset threshold and make inactive again

            Use of two thresholds allows using a tight (onset) threshold, while minimizing impact of short transient noise spikes
            upto (offset) that would otherwise create disjoint parts or require loosening the onset threshold.
            this yields more reliable durations of silence regions.

        NOTE: if the wav doesn't end with a silence, this function adds a dummy silence point to mark the ending speech frame

        Returns:
            list of silence points [(start_idx, duration)]
        """

        start, active = 0, spProb[0] < self.sil_onset_threshold
        points = []

        # iterate through spProb with enumerate but skip index 0
        for i, p in enumerate(spProb[1:], start=1):
            if not active:
                # listen for when drops below self.sil_onset_threshold
                if p < self.sil_onset_threshold:
                    start, active = i, True  # make active
            else:
                # listen for when crosses above self.sil_offset_threshold
                if p > self.sil_offset_threshold:
                    duration = i - start
                    if duration > self.min_sil_point_len:
                        points.append((start, duration))  # trigger marking of point
                    active = False  # reset to inactive

        if active:  # trailing silence
            points.append((start, i - start))  # add last point before trailing silence
        else:  # trailing speech
            points.append((len(spProb) - 1, 1000))  # add dummy silence to mark last frame

        return points

    def _greedy_buffer_segments(self, silence_points):
        """
        VAD-v1:
        Keeps extending the end of a buffer segment upto the next silence point and yields it before it would cross max_segment_len.
        The buffer is also force yielded on encountering an obvious noise/silence > self.min_sil_region_len, and the silence is excluded.

        This is a greedy strategy to create as long as possible segments while always being bounded
        by silence_points to avoid any mid-speech segmentation.

        When adding a segment (see _add_segment for details):
            we filter out segments < min_segment_len that are mostly short noise/spikes from the VAD.
            it also takes care of a blank (0, 0) segment that can get added if we have a large leading silence.
            if the segment somehow still exceeds the max_segment_len, we fall back to splitting it with a fixed window
                (e.g. buffer is 5s and next point is 60s away - unlikely as the VAD is very sensitive to even short pauses)

        A leading collar can also be added to segments to add context around tight segment boundaries

        Assumes silence_points contains at least 1 point marking the end point of the wav,
        which is used to end the last segment.

        Returns:
            list of [start, end) left-inclusive segment intervals, units are 10ms frame #s
        """

        # initialize buffer at start of utt
        buffer, final_segments, reset_point = [0, 0], [], 0
        _leading_collar = 0
        _ending_collar = min(silence_points[0][1], self.trailing_collar) if len(silence_points) > 0 else 0
        b_len = (
            lambda x: x[1] - x[0] + _leading_collar + _ending_collar
        )  # make sure that collar is included in len comparisons

        for i, sp in enumerate(silence_points):
            # dont yield buffer until it crosses minimum len
            buffer_short = b_len(buffer) < self.min_buffer_len
            # yield buffer if extending to this point would cross maximum len
            buffer_yield = b_len([buffer[0], sp[0]]) > self.max_segment_len and not buffer_short
            # force yield buffer if its current end point has a long silence
            curr_ending_sil = reset_point - buffer[-1]
            long_silence = curr_ending_sil > self.min_sil_region_len

            if buffer_yield or long_silence:
                self._add_segment(
                    final_segments,
                    *buffer,
                    leading_collar=_leading_collar,
                    ending_collar=_ending_collar,
                )  # NOTE: trailing collar can be added here
                if long_silence:
                    self.fn_print(f"\tlong silence: frame {buffer[-1]}, duration {curr_ending_sil}, end {reset_point}")
                else:
                    self.fn_print(f"\tpause: frame {buffer[-1]}, duration {curr_ending_sil}, end {reset_point}")
                # start a new segment at [end of last silence point, current point]
                _leading_collar = min(
                    curr_ending_sil, self.leading_collar
                )  # collar shouldn't extend into prev segment
                buffer = [reset_point, sp[0]]

            # extend buffer to current point
            buffer[-1] = sp[0]
            reset_point = sp[0] + sp[1]  # end of the silence point
            _ending_collar = min(sp[1], self.trailing_collar)

        if b_len(buffer) > 0:  # last segment
            self._add_segment(
                final_segments,
                *buffer,
                leading_collar=_leading_collar,
                ending_collar=_ending_collar,
            )
            self.fn_print("\tadded last segment")

        return final_segments

    def _add_segment(self, seg_list, start, end, leading_collar=None, ending_collar=None):
        # filters out empty and very short segments
        if (end - start) <= self.min_segment_len:
            return

        if leading_collar is not None:
            assert leading_collar <= self.leading_collar
            start -= leading_collar

        if ending_collar is not None:
            assert ending_collar <= self.trailing_collar
            if end - start >= self.max_segment_len - ending_collar and end - start <= self.max_segment_len:
                end = start + self.max_segment_len
            else:
                end += ending_collar

        if (end - start) > self.max_segment_len:
            subsegments = self._split_with_fixed_window(start, end)  # contains a small depth-1 recursive call
            seg_list.extend(subsegments)
            self.fn_print(f"added segment {len(seg_list) - 1:03d} split into {len(subsegments)} fixed windows")
        else:
            seg_list.append((start, end))
            self.fn_print(f"added segment {len(seg_list) - 1:03d}, [{start:06d} : {end:06d}]")

    def _split_with_fixed_window(self, start, end):
        subsegments = []
        while start < end:
            self._add_segment(subsegments, start, min(start + self.max_segment_len, end))
            start += self.max_segment_len
        return subsegments

    def _assert_monotonic(self, segments):
        monotonic, end = True, 0
        for s in segments:
            if end - s[0] > self.trailing_collar:
                monotonic = False
                break
            end = s[1]
        assert monotonic, f"Segment {s} has starts less than previous end time {end}"


if __name__ == "__main__":

    def get_audio(file: str, sr: int = 16000):
        import ffmpeg

        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    segmenter = SVadChunker(config={}, max_len_sec=30, verbose=True)

    sample_rate = 16000
    wavform = get_audio("D:\\GitRepos\\Whisper\\WhisperAdaptation\\audio-1min.wav", sample_rate)
    segments = segmenter.chunk(wavform, sample_rate)
    import datetime

    for s in segments:
        print(
            f"[ {datetime.timedelta(seconds=s[0])} -->  {datetime.timedelta(seconds=s[1])}]",
            s[1] - s[0],
        )
