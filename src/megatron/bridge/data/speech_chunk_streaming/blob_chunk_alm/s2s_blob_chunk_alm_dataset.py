# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import threading
import traceback
from queue import Full, Queue
from typing import Callable, List, Optional

import numpy as np
import torch

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.blob_chunk_alm_dataset import (
    BlobChunkALMDataset,
    FrameBatchConditionChecker,
    _group_utterance_to_batches,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.s2s_alm_sample_processor import (
    S2SBlobChunkALMSampleProcessor,
    SpeechOutRawAudioSampleProcessor,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.streaming_blobchunk import (
    ChunkSpeechTextSeq2SeqDataset as S2SChunkSpeechTextSeq2SeqDataset,
)
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger


logger = get_logger(__name__)


class S2SFrameBatchConditionChecker(FrameBatchConditionChecker):
    def __init__(self, batch_size, max_utterances=None):
        super().__init__(batch_size)
        self.max_utterances = max_utterances

    def check_constraint(self):
        if super().check_constraint():
            return True
        return self.max_utterances is not None and self.num_utt >= self.max_utterances

    def add_utt(self, utt_idx, sorted_idx_len_pair):
        self.max_frames = max(self.max_frames, sorted_idx_len_pair[utt_idx][1])
        self.num_utt += 1


class SpeechOutBlobChunkALMDataset(BlobChunkALMDataset):
    def __init__(
        self,
        dataset_spec: dict,
        num_chunks_for_shuffling: int,
        transforms: Callable = None,
        tokenize: Callable = None,
        context_length: int = 77,
        num_replicas: int = None,
        rank: int = None,
        drop_last: bool = False,
        seed: int = 0,
        multi_threaded_download: bool = False,
        use_num_examples_as_durations: bool = False,
        tsv_chunk_size: int = 256,
        sp_transcription: bool = False,
        debug: bool = False,
        tokenize_padding="max_length",
        path_prefix_in_orange: Optional[str] = None,
        seq_len: int = 2048,
        shift_labels: bool = False,
        audio_processor: Optional[str] = None,
        compression_rate: int = 4,
        streaming_compression: bool = False,
        qformer_compression_rate: int = 1,
        batcher_input_key: str = "audio",
        final_audio_key: str = "feature",
        padding_constraint: int = -1,
        batcher_buffer_size: int = 500,
        batcher_batch_size: int = 1000,
        batcher_criterion: str = "frame",
        batcher_max_utterances: Optional[int] = None,
        filter_options: dict = None,
        text_normalization: dict = None,
        conv_template: str = "phi_3",
        retain_first_n_turns_in_conversation: Optional[int] = None,
        allow_pure_text_return: bool = False,
        chunk_processing_pipe: Optional[List] = None,
        prefetch_queue_size: int = 64,
        use_storage_api: bool = False,
        codec_layer: int = 3,
        codec_cfg: dict = None,
    ):
        base_conv_template = conv_template
        if base_conv_template and base_conv_template.endswith("_s2s"):
            base_conv_template = base_conv_template[: -len("_s2s")]
        super().__init__(
            dataset_spec=dataset_spec,
            num_chunks_for_shuffling=num_chunks_for_shuffling,
            transforms=transforms,
            tokenize=tokenize,
            context_length=context_length,
            num_replicas=num_replicas,
            rank=rank,
            drop_last=drop_last,
            seed=seed,
            multi_threaded_download=multi_threaded_download,
            use_num_examples_as_durations=use_num_examples_as_durations,
            tsv_chunk_size=tsv_chunk_size,
            sp_transcription=sp_transcription,
            debug=debug,
            tokenize_padding=tokenize_padding,
            path_prefix_in_orange=path_prefix_in_orange,
            seq_len=seq_len,
            shift_labels=shift_labels,
            audio_processor=audio_processor,
            compression_rate=compression_rate,
            streaming_compression=streaming_compression,
            qformer_compression_rate=qformer_compression_rate,
            batcher_input_key=batcher_input_key,
            final_audio_key=final_audio_key,
            padding_constraint=padding_constraint,
            batcher_buffer_size=batcher_buffer_size,
            batcher_batch_size=batcher_batch_size,
            batcher_criterion=batcher_criterion,
            filter_options=filter_options,
            text_normalization=text_normalization,
            conv_template=base_conv_template,
            retain_first_n_turns_in_conversation=retain_first_n_turns_in_conversation,
            allow_pure_text_return=allow_pure_text_return,
            chunk_processing_pipe=chunk_processing_pipe,
            prefetch_queue_size=prefetch_queue_size,
            use_storage_api=use_storage_api,
        )

        s2s_audio_dataset = S2SChunkSpeechTextSeq2SeqDataset(
            dataset_spec=dataset_spec,
            num_chunks_for_shuffling=num_chunks_for_shuffling,
            transforms=transforms,
            tokenize=tokenize,
            context_length=context_length,
            num_replicas=num_replicas,
            rank=rank,
            drop_last=drop_last,
            seed=seed,
            multi_threaded_download=multi_threaded_download,
            use_num_examples_as_durations=use_num_examples_as_durations,
            tsv_chunk_size=tsv_chunk_size,
            debug=debug,
            skip_blob_file_path=None,
            skip_missing_data=False,
            tokenize_padding=tokenize_padding,
            allow_pure_text_return=allow_pure_text_return,
            chunk_processing_pipe=chunk_processing_pipe,
            path_prefix_in_orange=path_prefix_in_orange,
            use_storage_api=use_storage_api,
        )
        self.audio_dataset = s2s_audio_dataset.audio_dataset
        self.num_samples = s2s_audio_dataset.num_samples
        self.total_durations = s2s_audio_dataset.total_durations

        self.codec_layer = codec_layer
        self.codec_cfg = copy.deepcopy(codec_cfg or {})
        self._batcher_max_utterances = batcher_max_utterances
        if self._condition_checker_class is FrameBatchConditionChecker:
            self._condition_checker_class = S2SFrameBatchConditionChecker
        self._sample_processor = S2SBlobChunkALMSampleProcessor(
            tokenizer=self.tokenize,
            audio_processor=self.audio_processor,
            feat_stride=self.feat_stride,
            input_key=self._input_key,
            final_audio_key=self._final_audio_key,
            allow_pure_text_return=allow_pure_text_return,
            retain_first_n_turns_in_conversation=retain_first_n_turns_in_conversation,
            default_filter_options=self._default_filter_options,
            default_text_normalization=self._default_text_normalization,
            default_sp_transcription=self._default_sp_transcription,
            compute_audio_embed_size=self._compute_audio_embed_size,
            codec_layer=codec_layer,
            codec_cfg=self.codec_cfg,
        )

    def _producer_loop(self):
        try:
            self._example_buffer = []
            self._batch_buffer = []
            consecutive_empty_batch_creations = 0
            self._first_iteration = True
            self._max_prompt_len = self.get_max_prompt_len()
            self.worker_id = 0
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self.worker_id = worker_info.id

            self.audio_iter = iter(self.audio_dataset)

            while not self._prefetch_stop:
                if self._first_iteration:
                    self._first_iteration = False
                    self._fill_buffer_by_length(self._example_buffer, self._buffer_size)

                if not self._batch_buffer and not self._example_buffer:
                    break

                if not self._batch_buffer:
                    self._batch_buffer, self._example_buffer = self._create_batch_list()
                    if not self._batch_buffer:
                        consecutive_empty_batch_creations += 1
                        logger.warning(
                            "No batch could be formed from %d buffered examples. "
                            "Discarding buffered examples and refilling.",
                            len(self._example_buffer),
                        )
                        if consecutive_empty_batch_creations >= 3:
                            break
                        self._example_buffer = []
                        self._fill_buffer_by_length(self._example_buffer, self._buffer_size)
                        continue
                    consecutive_empty_batch_creations = 0

                single_batch = self._batch_buffer.pop()
                if len(self._example_buffer) < self._buffer_size:
                    self._fill_buffer_by_length(self._example_buffer, len(single_batch))

                raw_sample = self._build_raw_sample(single_batch)
                if self.transforms.prompt_formatter == "asr-lid-codeswitch":
                    raw_sample = self.sythetic_codeswitch(raw_sample)

                try:
                    transformed_sample = self.transforms(raw_sample)
                except Exception as e:
                    logger.info(f"Transform failed, error: {e}")
                    logger.info(f"Transform failed, traceback: {traceback.format_exc()}")
                    continue

                sample = self._build_model_sample(transformed_sample, raw_sample["codecs"])
                if sample is None:
                    continue

                if self.worker_id == 0:
                    sample["_resume_epoch"] = self.audio_iter.epoch
                    sample["_resume_step"] = self.audio_iter.worker_step_in_epoch

                if not self._queue_put_with_stop(sample):
                    break

            self._queue_put_with_stop(None)
        except BaseException as e:
            self._prefetch_exception = e
            self._queue_put_with_stop(None)

    def _build_raw_sample(self, single_batch):
        audios, texts, user_prompts, languages, spkid, contexts = [], [], [], [], [], []
        audio_embed_sizes, audio_frames = [], []
        codecs, alignments, use_eoc_endings = [], [], []

        for example in single_batch:
            audios.extend(example[self._final_audio_key])
            audio_embed_sizes.append(example["audio_embed_sizes"])
            audio_frames.extend(example["audio_frames"])
            texts.append(example["annotations"])
            user_prompts.append(example["user_prompt"])
            languages.append(example["language"])
            spkid.append(example.get("spkid"))
            contexts.append(example.get("context", None))

            if "answer_cv_tokens" in example:
                codecs.extend(example["answer_cv_tokens"])
            else:
                codecs.append(example.get("label_codec"))
            alignments.append(example.get("answer_alignment") or example.get("alignment"))
            use_eoc_endings.append(example.get("_use_eoc_ending", False))

        return {
            "audios": audios,
            "texts": texts,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_frames": audio_frames,
            "tokenizer": self.tokenize,
            "user_prompts": user_prompts,
            "languages": languages,
            "spkid": spkid,
            "contexts": contexts,
            "codecs": codecs,
            "alignments": alignments,
            "use_eoc_endings": use_eoc_endings,
            "codec_cfg": self.codec_cfg,
        }

    def _build_model_sample(self, sample, codecs):
        input_embeds = sample["audios"]
        labels = sample["labels"].input_ids
        assert sample["is_text_sample"] == self.allow_pure_text_return, (
            "please make sure this dataloader is with pure text data"
        )
        ignore_idx = -100
        if all(labels.eq(ignore_idx).view(-1)):
            logger.info("All labels are IGNORE_IDX, skip")
            return None

        output = self._codec_output_fields(sample, codecs)
        output.update(
            {
                "input_ids": sample["inputs"]["input_ids"],
                "input_embeds": input_embeds,
                "labels": labels,
                "audio_embed_sizes": sample["audio_embed_sizes"],
                "audio_frames": sample["audio_frames"],
                "is_text_sample": sample["is_text_sample"],
                "spkid": sample.get("spkid", [None] * len(input_embeds)),
            }
        )

        for key in ("audio_token_mask", "duration_target", "stop_target", "num_chunks", "view"):
            if sample.get(key) is not None:
                output[key] = sample[key]

        if "ctc_labels" in sample:
            output.update(
                {
                    "ctc_labels": sample["ctc_labels"].input_ids,
                    "ctc_label_lens": sample["ctc_label_lens"],
                }
            )
        return output

    def _codec_output_fields(self, sample, codecs):
        pattern_mode = self.codec_cfg.get("pattern") if isinstance(self.codec_cfg, dict) else None
        if not codecs or (codecs[0] is None and pattern_mode is None):
            return {}

        if pattern_mode in {"interleave", "interleave_fake", "interleave_short", "interleave_short_fake"}:
            return {"label_codecs": sample.get("label_codecs")}
        if pattern_mode == "codec_only":
            return {}
        return {
            "input_codecs": sample.get("input_codecs"),
            "label_codecs": sample.get("label_codecs"),
        }

    def _queue_put_with_stop(self, item) -> bool:
        while not self._prefetch_stop:
            try:
                self._prefetch_queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def __iter__(self):
        self._prefetch_stop = False
        self._prefetch_queue = Queue(maxsize=self._prefetch_queue_size)
        self._prefetch_exception = None
        self._prefetch_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self._prefetch_thread.start()
        try:
            while True:
                batch = self._prefetch_queue.get()
                if batch is None:
                    if self._prefetch_exception is not None:
                        raise self._prefetch_exception
                    break
                if "_resume_epoch" in batch:
                    self.shared_state["epoch"] = batch.pop("_resume_epoch")
                    self.shared_state["generator_state"] = batch.pop("_resume_step")
                yield batch
        finally:
            self._prefetch_stop = True
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                self._prefetch_thread.join(timeout=1.0)
            self._prefetch_thread = None
            self._prefetch_queue = None

    def _create_batch_list(self):
        sorted_idx_len_pair, leftover_buffer = self._build_sorted_idx(self._example_buffer)
        if self._condition_checker_class is S2SFrameBatchConditionChecker:
            checker = S2SFrameBatchConditionChecker(
                self._batch_size_object,
                max_utterances=self._batcher_max_utterances,
            )
        else:
            checker = self._condition_checker_class(self._batch_size_object)
        batch_list, leftover_buffer2 = _group_utterance_to_batches(
            self._example_buffer,
            sorted_idx_len_pair,
            self._padding_constraint,
            checker,
        )
        leftover_buffer.extend(leftover_buffer2)
        if self._shuffle:
            np.random.shuffle(batch_list)
        else:
            batch_list = batch_list[::-1]
        return batch_list, leftover_buffer

    def _fill_buffer_by_length(self, example_buffer, length):
        try:
            buffer = []
            while len(buffer) < length:
                raw_sample = next(self.audio_iter)
                if len(raw_sample) == 0:
                    continue

                for sample in self._maybe_segment_podcast(raw_sample):
                    sample_processing_state = self._sample_processor.resolve_sample_processing_state(sample)
                    sample = self._sample_processor.prepare_multiaudio_sample(sample, sample_processing_state)
                    if sample is None:
                        continue

                    sample_rate = None
                    input_value = sample.get(self._input_key)
                    if isinstance(input_value, tuple) and len(input_value) == 2:
                        sample[self._input_key], sample_rate = input_value

                    sample = self._sample_processor.apply_transcription_processing(sample, sample_processing_state)
                    if sample is None or not self._has_qualified_qa(sample):
                        continue

                    filter_key = "feature" if ("audio" not in sample and sample.get("feature") is not None) else None
                    if not self._sample_processor.passes_sample_filters(
                        sample,
                        sample_processing_state,
                        sample_rate,
                        filter_key=filter_key,
                    ):
                        continue

                    if "answer_cv_token" in sample or "answer_cv_tokens" in sample:
                        sample = self._sample_processor.apply_audio_truncation(sample, sample_rate)
                        if sample is None:
                            continue
                        sample = self._sample_processor.process_codec_tokens(sample)
                        if sample is None:
                            continue

                    audio_frames, audio_embed_sizes = self._sample_processor.extract_audio_features(sample, sample_rate)
                    buffer.extend(
                        self._sample_processor.materialize_examples(
                            sample,
                            audio_frames,
                            audio_embed_sizes,
                            sample_processing_state,
                        )
                    )
                    if len(buffer) >= length:
                        break

            example_buffer.extend(buffer)
        except StopIteration:
            if len(buffer) > 0:
                example_buffer.extend(buffer)

    def _has_qualified_qa(self, sample):
        if "single_qa" not in sample:
            return True
        scores = sample["single_qa"]["score"]
        if not isinstance(scores, list):
            scores = [scores]
        return any(score > 8 for score in scores)


class SpeechOutInfiniteBlobChunkALMDataset(SpeechOutBlobChunkALMDataset):
    def __iter__(self):
        while True:
            yield from super().__iter__()
            logger.info(f"Epoch {self.audio_dataset.epoch} finished, starting new epoch")
            self.set_epoch(self.audio_dataset.epoch + 1)


class SpeechOutRawAudioBlobChunkALMDataset(SpeechOutBlobChunkALMDataset):
    """Dataset for the raw-audio forward path.

    Emits raw ``answer_audio`` waveforms (one per sample, no concat/pad, no
    tokenization) plus the unpadded sequence structure, so ``model.forward`` can
    tokenize + build + pad. Skips all codec-token construction. The existing
    ``.answer_cv_token`` dataset behavior is untouched.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap in the raw-audio sample processor (carries answer_audio, skips codec).
        self._sample_processor = SpeechOutRawAudioSampleProcessor(
            tokenizer=self.tokenize,
            audio_processor=self.audio_processor,
            feat_stride=self.feat_stride,
            input_key=self._input_key,
            final_audio_key=self._final_audio_key,
            allow_pure_text_return=self.allow_pure_text_return,
            retain_first_n_turns_in_conversation=self.retain_first_n_turns_in_conversation,
            default_filter_options=self._default_filter_options,
            default_text_normalization=self._default_text_normalization,
            default_sp_transcription=self._default_sp_transcription,
            compute_audio_embed_size=self._compute_audio_embed_size,
            codec_layer=self.codec_layer,
            codec_cfg=self.codec_cfg,
            prompt_formatter=getattr(self.transforms, "prompt_formatter", None),
        )

    @staticmethod
    def _extract_waveform(answer_audio):
        if answer_audio is None:
            return None
        if isinstance(answer_audio, (tuple, list)) and len(answer_audio) == 2:
            return answer_audio[0]
        return answer_audio

    def _build_raw_sample(self, single_batch):
        audios, texts, user_prompts, languages, spkid, contexts = [], [], [], [], [], []
        audio_embed_sizes, audio_frames = [], []
        answer_audio, alignments, use_eoc_endings, infos = [], [], [], []

        for example in single_batch:
            audios.extend(example[self._final_audio_key])
            audio_embed_sizes.append(example["audio_embed_sizes"])
            audio_frames.extend(example["audio_frames"])
            texts.append(example["annotations"])
            user_prompts.append(example["user_prompt"])
            languages.append(example["language"])
            spkid.append(example.get("spkid"))
            contexts.append(example.get("context", None))
            answer_audio.append(self._extract_waveform(example.get("answer_audio")))
            alignments.append(example.get("answer_alignment") or example.get("alignment"))
            use_eoc_endings.append(example.get("_use_eoc_ending", False))
            infos.append(example.get("info"))

        return {
            "audios": audios,
            "texts": texts,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_frames": audio_frames,
            "tokenizer": self.tokenize,
            "user_prompts": user_prompts,
            "languages": languages,
            "spkid": spkid,
            "contexts": contexts,
            "answer_audio": answer_audio,
            "alignments": alignments,
            "use_eoc_endings": use_eoc_endings,
            "infos": infos,
            "codec_cfg": self.codec_cfg,
            "codecs": None,
        }

    def _build_model_sample(self, sample, codecs=None):
        input_embeds = sample["audios"]
        output = {
            "input_embeds": input_embeds,
            "audio_embed_sizes": sample["audio_embed_sizes"],
            "audio_frames": sample["audio_frames"],
            "is_text_sample": sample["is_text_sample"],
            "spkid": sample.get("spkid", [None] * len(input_embeds)),
            "answer_audio": sample["answer_audio"],
            "seq_plan": sample["seq_plan"],
            "codec_cfg": sample.get("codec_cfg"),
        }
        return output


class SpeechOutRawAudioInfiniteBlobChunkALMDataset(SpeechOutRawAudioBlobChunkALMDataset):
    def __iter__(self):
        while True:
            yield from super().__iter__()
            logger.info(f"Epoch {self.audio_dataset.epoch} finished, starting new epoch")
            self.set_epoch(self.audio_dataset.epoch + 1)
