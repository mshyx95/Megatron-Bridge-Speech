# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import re
import copy
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from multiprocessing import Manager

import numpy as np
import torch
from queue import Full, Queue
import threading
import random
from collections import deque
from torch.utils.data import Dataset

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.blobchunk import ChunkSpeechTextSeq2SeqDataset
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.audio_processors import W2vBERTProcessor, WhisperProcessor, LogFbankProcessor

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.alm_sample_processor import BlobChunkALMSampleProcessor

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.utils import (
    ASR_PROMPTS,
)

from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger


logger = get_logger(__name__)


def _fill_batch(cur_idx, sorted_idx_len_pair, padding_constraint, condition_checker):
    """Will return start_idx and end_idx for the current batch."""
    start_idx = cur_idx
    end_idx = cur_idx
    condition_checker.reset()
    condition_checker.add_utt(end_idx, sorted_idx_len_pair)
    while not condition_checker.check_constraint():
        min_len = sorted_idx_len_pair[end_idx][1]
        max_len = sorted_idx_len_pair[start_idx][1]
        # if padding constraint is violated, let's remove longest utterance
        # otherwise, will add another utterance and check constraints again
        if padding_constraint != -1 and max_len - min_len > padding_constraint:
            condition_checker.remove_utt(start_idx, sorted_idx_len_pair)
            start_idx += 1
        else:
            end_idx += 1
            if end_idx == len(sorted_idx_len_pair):
                if padding_constraint != -1:
                    return -1, -1  # this means we were not able to fill up batch
                # if no constraints, we have to return last batch as it is
                return start_idx, end_idx - 1
            condition_checker.add_utt(end_idx, sorted_idx_len_pair)
    if start_idx == end_idx:
        return start_idx, end_idx
    return start_idx, end_idx - 1


def _collect_utters(start_idx, end_idx, buffer, sorted_idx_len_pair):
    utts = [None] * (end_idx - start_idx + 1)
    for utt_idx in range(start_idx, end_idx + 1):
        utts[utt_idx - start_idx] = buffer[sorted_idx_len_pair[utt_idx][0]]
    return utts


def _group_utterance_to_batches(buffer, sorted_idx_len_pair, padding_constraint, condition_checker):
    batch_list = []

    cur_idx = 0
    leftover_buffer = []
    while True:
        if cur_idx == len(sorted_idx_len_pair):
            break
        start_idx, end_idx = _fill_batch(cur_idx, sorted_idx_len_pair, padding_constraint, condition_checker)
        # adding unused utterances to leftover buffer
        if start_idx > cur_idx:
            leftover_buffer.extend(_collect_utters(cur_idx, start_idx - 1, buffer, sorted_idx_len_pair))
        # couldn't fill-up current batch
        if start_idx == -1:
            leftover_buffer.extend(_collect_utters(cur_idx, len(sorted_idx_len_pair) - 1, buffer, sorted_idx_len_pair))
            break
        batch_list.append(_collect_utters(start_idx, end_idx, buffer, sorted_idx_len_pair))
        cur_idx = end_idx + 1

    # if no batch was formed, let's return everything that we have
    # ignoring padding constraint
    if len(batch_list) == 0 and padding_constraint != -1:
        return _group_utterance_to_batches(buffer, sorted_idx_len_pair, -1, condition_checker)

    return batch_list, leftover_buffer


class UttBatchConditionChecker:
    """Will check if utterance-based batch forming condition is satisfied."""

    def __init__(self, batch_size):
        self.num_utt = 0
        self.batch_size = batch_size

    def reset(self):
        """Resets all counts."""
        self.num_utt = 0

    def check_constraint(self):
        """Returns if constraint is satisfied."""
        return self.num_utt > self.batch_size

    # pylint: disable=unused-argument
    def remove_utt(self, utt_idx, sorted_idx_len_pair):
        """Removes an utterance from the current batch."""
        self.num_utt -= 1

    # pylint: disable=unused-argument
    def add_utt(self, utt_idx, sorted_idx_len_pair):
        """Adds an utterance to the current batch."""
        self.num_utt += 1


class FrameBatchConditionChecker:
    """Will check if frame-based batch forming condition is satisfied."""

    def __init__(self, batch_size):
        self.max_frames = 0
        self.num_utt = 0
        self.batch_size = batch_size

    def reset(self):
        """Resets all counts."""
        self.max_frames = 0
        self.num_utt = 0

    def check_constraint(self):
        """Returns if constraint is satisfied."""
        return self.num_utt * self.max_frames > self.batch_size

    def remove_utt(self, utt_idx, sorted_idx_len_pair):
        """Removes an utterance from the current batch."""
        # removing current one, meaning max frames is in the next utt
        self.max_frames = sorted_idx_len_pair[utt_idx + 1][1]
        self.num_utt -= 1

    def add_utt(self, utt_idx, sorted_idx_len_pair):
        """Adds an utterance to the current batch."""
        if self.max_frames == 0:
            self.max_frames = sorted_idx_len_pair[utt_idx][1]
        self.num_utt += 1


class SumFrameBatchConditionChecker:
    """Packs utterances until the *sum* of their token lengths exceeds ``batch_size``.

    Unlike ``FrameBatchConditionChecker`` (which uses the padding proxy
    ``num_utt * max_frames``), this criterion tracks the true cumulative length so a
    packed row's total tokens stay bounded by ``batch_size``. This is required for the
    forward-tokenize path where each sample's length already includes the answer-codec
    expansion (``est_answer_positions``): packing by sum keeps rows filled close to the
    budget yet guaranteed <= budget, avoiding both the under-packing and the >8192
    O(L^2) blow-up that the ``max_frames`` proxy causes.
    """

    def __init__(self, batch_size):
        self.sum_frames = 0
        self.num_utt = 0
        self.batch_size = batch_size

    def reset(self):
        """Resets all counts."""
        self.sum_frames = 0
        self.num_utt = 0

    def check_constraint(self):
        """Returns if constraint is satisfied."""
        return self.sum_frames > self.batch_size

    def remove_utt(self, utt_idx, sorted_idx_len_pair):
        """Removes an utterance from the current batch."""
        self.sum_frames -= sorted_idx_len_pair[utt_idx][1]
        self.num_utt -= 1

    def add_utt(self, utt_idx, sorted_idx_len_pair):
        """Adds an utterance to the current batch."""
        self.sum_frames += sorted_idx_len_pair[utt_idx][1]
        self.num_utt += 1


class BlobChunkALMDataset(ChunkSpeechTextSeq2SeqDataset):
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
        batcher_criterion: float = "frame",
        filter_options: dict = None,
        text_normalization: dict = None,
        conv_template: str = "phi_3",
        retain_first_n_turns_in_conversation: Optional[int] = None,
        allow_pure_text_return: bool = False,
        chunk_processing_pipe: Optional[List] = None,
        prefetch_queue_size: int = 64,
        use_storage_api: bool = False,
    ):

        super().__init__(
            dataset_spec,
            num_chunks_for_shuffling,
            transforms,
            tokenize,
            context_length,
            num_replicas,
            rank,
            drop_last,
            seed,
            multi_threaded_download,
            use_num_examples_as_durations,
            tsv_chunk_size,
            debug,
            None,
            False,
            tokenize_padding=tokenize_padding,
            allow_pure_text_return=allow_pure_text_return,
            chunk_processing_pipe=chunk_processing_pipe,
            path_prefix_in_orange=path_prefix_in_orange,
            use_storage_api=use_storage_api,
        )

        self.label_offset = 1 if shift_labels else 0
        self.compression_rate = compression_rate
        self.streaming_compression = streaming_compression
        self.qformer_compression_rate = qformer_compression_rate

        if isinstance(audio_processor, dict) and audio_processor.get("name", None) == "cascades":
            if "feature" not in dataset_spec["data_sources"][0]["chunk_type"]:
                self.audio_processor = LogFbankProcessor()
            else:
                self.audio_processor = None
            self.feat_stride = 1
        elif isinstance(audio_processor, dict) and audio_processor.get("name", None) == "whisper":
            pad_features_inside = audio_processor.get("pad_features_inside", False)
            pad_features_outside = audio_processor.get("pad_features_outside", False)
            self.audio_processor = WhisperProcessor(pad_features_inside=pad_features_inside, pad_features_outside=pad_features_outside)
            self.feat_stride = 1
        elif isinstance(audio_processor, dict) and audio_processor.get("name", None) == "w2vbert2":
            self.audio_processor = W2vBERTProcessor()
            self.feat_stride = 2
        else:
            raise NotImplementedError

        # Create a shared dict so that we can get information from another
        # process (created by the dataloader) and save the state of the dataset
        self.dataset_spec_str = str(dataset_spec)
        self.shared_state = Manager().dict()
        self.shared_state["dataset_path"] = self.dataset_spec_str
        self.shared_state["epoch"] = 0
        self.shared_state["generator_state"] = 0

        # needed for calculating batch statics
        self.seq_len = seq_len

        # for dynamic batcher
        self._batch_size_object = batcher_batch_size  # maximum is context length

        if batcher_criterion == "frame":
            self._condition_checker_class = FrameBatchConditionChecker  # audio + annotation lengths
        elif batcher_criterion == "frame_sum":
            self._condition_checker_class = SumFrameBatchConditionChecker  # sum of packed token lengths (<= batch_size)
        elif batcher_criterion == "utterance":
            self._condition_checker_class = UttBatchConditionChecker  # number of utterances in a batch
        else:
            raise NotImplementedError

        self._input_key = batcher_input_key if self.audio_processor else "feature"
        self._final_audio_key = final_audio_key if self.audio_processor else self._input_key
        self._buffer_size = batcher_buffer_size
        self._padding_constraint = padding_constraint
        self._shuffle = True
        self._example_buffer = []
        self._batch_buffer = []
        filter_options = filter_options or {}
        self._default_filter_options = copy.deepcopy(filter_options)
        self._default_text_normalization = copy.deepcopy(text_normalization)
        self._default_sp_transcription = sp_transcription
        self._sample_processor = BlobChunkALMSampleProcessor(
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
        )

        self.sp_transcription = sp_transcription
        assert conv_template in ["phi_3", "phi_3_200k", "phi_o", "phi_o_long", "qwen3_audio_instruct", "qwen3_audio_nothink_instruct", "mistral3_audio_instruct", "gemma3_audio_instruct"], "conv template has to be either phi_3, phi_3_200k, phi_o, phi_o_long, qwen3_audio_instruct, qwen3_audio_nothink_instruct, mistral3_audio_instruct, or gemma3_audio_instruct"
        self.conv_template = conv_template
        assert retain_first_n_turns_in_conversation is None or retain_first_n_turns_in_conversation > 0, retain_first_n_turns_in_conversation
        self.retain_first_n_turns_in_conversation = retain_first_n_turns_in_conversation
        self.allow_pure_text_return = allow_pure_text_return
        
        self._prefetch_queue: Optional[Queue] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_exception: Optional[BaseException] = None
        self._prefetch_stop = False
        self._prefetch_queue_size = prefetch_queue_size

        #For phi, the max token length is 8192 (8k), for ministral, it's 131072 (128k), we save some space for other tokens like <|user|>, <|end|> and so on. so we set the max token length to 8000 for phi and 131000 for ministral. 
        if self.conv_template in ("phi_o_long", "phi_o_long_s2s"):
            self.max_token_len = 8000
        elif self.conv_template == "mistral3_audio_instruct":
            self.max_token_len = 131000
        else:
            self.max_token_len = 32000

    def get_max_prompt_len(self):
        """get the max prompt length for the current dataset"""
        if self.conv_template in ["phi_3", "phi_o"]:
            prompt_text_enriched = [f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n<|end|>" for prompt in ASR_PROMPTS]
        elif self.conv_template == "phi_3_200k":
            prompt_text_enriched = [f"user:{prompt}<|endofprompt|>assistant:<|endoftext|>" for prompt in ASR_PROMPTS]
        elif self.conv_template == "phi_o_long":
            prompt_text_enriched = [f"<|user|>{prompt}<|end|><|assistant|><|end|>" for prompt in ASR_PROMPTS]
        elif self.conv_template == "qwen3_audio_instruct":
            prompt_text_enriched = [f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<|im_end|>" for prompt in ASR_PROMPTS]
        elif self.conv_template == "qwen3_audio_nothink_instruct":
            prompt_text_enriched = [f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>" for prompt in ASR_PROMPTS]
        elif self.conv_template == "mistral3_audio_instruct":
            prompt_text_enriched = [f"<s>[INST]{prompt}[/INST]</s>" for prompt in ASR_PROMPTS]
        elif self.conv_template == "gemma3_audio_instruct":
            prompt_text_enriched = [f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n<end_of_turn>" for prompt in ASR_PROMPTS]

        prompt_ids = [self.tokenize.encode(p_text, add_special_tokens=False) for p_text in prompt_text_enriched]
        prompt_ids_len = [len(ids) for ids in prompt_ids]
        return max(prompt_ids_len)

    @property
    def original_size(self) -> int:
        # the original size is used to compute the weight for each datasource
        if self.use_num_examples_as_durations:
            # in this case, we compute the step size to consume all examples in each data source assuming the micro-batch size is 1
            # the final dataset weight will be computed based on the step size
            logger.info("Number of examples for training is {:,}".format(self.total_durations))
            assert self._batch_size_object > 0
            step_size = self.total_durations // self._batch_size_object
            logger.info("Number of steps to consume all training examples with one gpu is {:,}".format(step_size))
            return step_size
        if self.audio_processor is not None:
            num_frames = self.audio_processor.duration_to_frames(self.total_durations)
        else:
            num_frames = self.total_durations * 100

        num_audio_tokens = self._compute_audio_embed_size(num_frames)
        logger.info("Estimated number of audio tokens for training is {:,}".format(num_audio_tokens))
        return num_audio_tokens

    @property
    def dataset_path(self) -> str:
        return self.dataset_spec_str

    def get_state(self) -> Dict[str, Any]:
        return dict(self.shared_state)

    def _producer_loop(self):
        """Background thread: fills the prefetch queue with ready batches."""
        _profile = os.environ.get("DATALOADER_PROFILE", "0") == "1"
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

            if _profile:
                _prof_transform = 0.0
                _prof_transform_n = 0

            while not self._prefetch_stop:
                if self._first_iteration:
                    self._first_iteration = False
                    self._fill_buffer_by_length(self._example_buffer, self._buffer_size)

                # both buffer are empty, stop
                if not self._batch_buffer and not self._example_buffer:
                    break

                if not self._batch_buffer:  # time to buffer switch
                    # create_batch_list might leave some examples un-batched
                    self._batch_buffer, self._example_buffer = self._create_batch_list()
                    if not self._batch_buffer:
                        consecutive_empty_batch_creations += 1
                        logger.warning(
                            "No batch could be formed from %d buffered examples. "
                            "Discarding buffered examples and refilling. "
                            "This usually means every example was filtered out, e.g. by max_token_len=%d.",
                            len(self._example_buffer),
                            self.max_token_len,
                        )
                        if consecutive_empty_batch_creations >= 3:
                            logger.error(
                                "Stopping dataset producer after %d consecutive empty batch formations.",
                                consecutive_empty_batch_creations,
                            )
                            break
                        self._example_buffer = []
                        self._fill_buffer_by_length(self._example_buffer, self._buffer_size)
                        continue

                    consecutive_empty_batch_creations = 0

                # pop the last batch from the batch buffer. O(1),
                # pop the fist element is O(N)
                single_batch = self._batch_buffer.pop()

                # fill the same number of examples in the example buffer
                if len(self._example_buffer) < self._buffer_size:
                    self._fill_buffer_by_length(self._example_buffer, len(single_batch))

                audios, texts, user_prompts, languages, spkid, contexts = [], [], [], [], [], []
                audio_embed_sizes, audio_frames = [], []

                for example in single_batch:
                    audios.extend(example[self._final_audio_key])
                    # audio_embed_sizes: use append to support multi-turn messages and concat_to_batch
                    audio_embed_sizes.append(example["audio_embed_sizes"])
                    audio_frames.extend(example["audio_frames"])
                    texts.append(example["annotations"])
                    user_prompts.append(example["user_prompt"])
                    languages.append(example["language"])
                    spkid.append(example.get("spkid"))
                    contexts.append(example.get("context", None))

                # TODO: when user prompts are in audio format
                raw_sample = {
                    "audios": audios,
                    "texts": texts,
                    "audio_embed_sizes": audio_embed_sizes,
                    "audio_frames": audio_frames,
                    "tokenizer": self.tokenize,
                    "user_prompts": user_prompts,
                    "languages": languages,
                    "spkid": spkid,
                    "contexts": contexts,
                }

                if self.transforms.prompt_formatter == "asr-lid-codeswitch":
                    raw_sample = self.sythetic_codeswitch(raw_sample)
            
                try:
                    if _profile:
                        _t_xform = time.time()
                    sample = self.transforms(raw_sample)
                    if _profile:
                        _prof_transform += time.time() - _t_xform
                        _prof_transform_n += 1
                except Exception as e:
                    logger.info(f"Transform failed, error: {e}")
                    logger.info(f"Transform failed, traceback: {traceback.format_exc()}")
                    continue

                # convert to Phi-2 model input
                input_embeds = sample["audios"]
                inputs = sample["inputs"]
                labels = sample["labels"].input_ids
                input_ids = inputs["input_ids"]
                audio_embed_sizes = sample["audio_embed_sizes"]
                audio_frames = sample["audio_frames"]
                is_text_sample = sample["is_text_sample"]
                assert is_text_sample == self.allow_pure_text_return, "please make sure this dataloader is with pure text data"
                spkid = sample.get("spkid", [None] * len(input_embeds))

                IGNORE_IDX = -100

                # check if not all labels is IGNORE_IDX, labels is tensor
                if all(labels.eq(IGNORE_IDX).view(-1)):
                    logger.info(f"All labels are IGNORE_IDX, skip")
                    continue

                if 'ctc_labels' in sample:
                    ctc_labels = sample['ctc_labels'].input_ids
                    ctc_label_lens = sample['ctc_label_lens']
                else:
                    ctc_labels = None

                sample = {
                    "input_ids": input_ids,
                    "input_embeds": input_embeds,
                    "labels": labels,
                    "audio_embed_sizes": audio_embed_sizes,
                    "audio_frames": audio_frames,
                    "is_text_sample": is_text_sample,
                    "spkid": spkid,
                }

                if ctc_labels is not None:
                    sample.update({
                        'ctc_labels': ctc_labels,
                        'ctc_label_lens': ctc_label_lens,
                    })

                # Attach resume metadata to the batch so that shared_state
                # is only updated on the *consumer* side (in __iter__),
                # preventing data loss from the prefetch queue on resume.
                if self.worker_id == 0:
                    sample["_resume_epoch"] = self.audio_iter.epoch
                    sample["_resume_step"] = self.audio_iter.worker_step_in_epoch

                if _profile and self.worker_id == 0 and _prof_transform_n > 0 and _prof_transform_n % 10 == 0:
                    print(f"[DL Profile] transforms: avg={_prof_transform/_prof_transform_n*1000:.1f}ms/batch over {_prof_transform_n} batches, total={_prof_transform:.2f}s", flush=True)

                if not self._queue_put_with_stop(sample):
                    break
            
            self._queue_put_with_stop(None)
        except BaseException as e:
            self._prefetch_exception = e
            self._queue_put_with_stop(None)

    def _queue_put_with_stop(self, item) -> bool:
        while not self._prefetch_stop:
            try:
                self._prefetch_queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False
                
    def __iter__(self):
        """Iterator that consumes batches from a background prefetch thread."""
        # Lazily initialize queue and thread per worker process
        self._prefetch_stop = False
        self._prefetch_queue = Queue(maxsize=self._prefetch_queue_size)
        self._prefetch_exception = None

        self._prefetch_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,
        )
        self._prefetch_thread.start()
        try:
            while True:
                batch = self._prefetch_queue.get()  # blocking
                if batch is None:
                    # Producer signaled end-of-data or error
                    if self._prefetch_exception is not None:
                        raise self._prefetch_exception
                    break
                # Update resume state on the consumer side so that
                # checkpoint reflects the last *consumed* batch,
                # not the last *produced* batch.
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

    def sythetic_codeswitch(self, sample):
        cs_samples = copy.deepcopy(sample)
        for i in range(len(sample["audios"])):
            first_language = sample["languages"][i]
            utt_pools = []
            for j in range(len(sample["audios"])):
                if i == j:
                    continue
                if sample["languages"][j] == first_language:
                    continue
                utt_pools.append(j)

            # randomly sample one utterance from different language as code-switching example
            if len(utt_pools) > 0:
                sampled_utt_idx = random.choice(utt_pools)
                cs_samples["texts"][i] = [sample["texts"][i][0] + "\n\n" + sample["texts"][sampled_utt_idx][0]]
                assert sample["user_prompts"][i] == None, "user prompt should be None code-switching example."
                cs_samples["languages"][i] = {"audio_0": sample["languages"][i], "audio_1": sample["languages"][sampled_utt_idx]}
                cs_samples["audios"][i] = np.concatenate([sample["audios"][i], sample["audios"][sampled_utt_idx]], axis=0)
                cs_samples["audio_embed_sizes"][i] = [self._compute_audio_embed_size(len(cs_samples["audios"][i]))]
                cs_samples["audio_frames"][i] = [len(cs_samples["audios"][i])]
                if sample["contexts"][i] is not None and sample["contexts"][sampled_utt_idx] is not None:
                    cs_samples["contexts"][i] = sample["contexts"][i] + ", " + sample["contexts"][sampled_utt_idx]
                
        del sample
        return cs_samples
                    
    def _create_batch_list(self):
        """Will form the batch buffer list."""
        # create batches using bucketing. the batch_list is a
        # list of list of utterances
        # [ [utt1, utt2, ...], [utt1, utt2, ...], ...]

        sorted_idx_len_pair, leftover_buffer = self._build_sorted_idx(self._example_buffer)
        batch_list, leftover_buffer2 = _group_utterance_to_batches(
            self._example_buffer,
            sorted_idx_len_pair,
            self._padding_constraint,
            self._condition_checker_class(self._batch_size_object),
        )

        leftover_buffer.extend(leftover_buffer2)
        if self._shuffle:
            np.random.shuffle(batch_list)  # random the group index.
        else:  # reversing buffer as we are going to pop from the end
            batch_list = batch_list[::-1]

        return batch_list, leftover_buffer

    def _parse_long_form_segments(self, parsed_data):
        """Parse long_form podcast info with pre-defined segments.

        The info dict has a 'segments' list where each segment has:
          - start/end: float in seconds
          - transcription: str
        Returns segments in the same format as _parse_info_for_podcast.
        """
        raw_segments = parsed_data.get("segments", [])
        if not raw_segments:
            return []

        segments = []
        for seg in raw_segments:
            start_sec = seg.get("start", seg.get("original_start", 0.0))
            end_sec = seg.get("end", seg.get("original_end", 0.0))
            text = seg.get("transcription", "")
            if not text or end_sec <= start_sec:
                continue
            segments.append({
                "segment_index": len(segments),
                "start_ms": int(start_sec * 1000),
                "end_ms": int(end_sec * 1000),
                "duration_ms": int((end_sec - start_sec) * 1000),
                "duration_sec": round(end_sec - start_sec, 1),
                "text": text,
                "phrase_count": 1,
            })
        return segments

    def _parse_info_for_podcast(self, parsed_data, target_duration_ms=30000, min_confidence=0.0):
        """Parse podcast info to create time-based segments from machine_transcription.

        Supports two formats:
          - azure_fast: phrases with offsetMilliseconds/durationMilliseconds/text
          - whisper:    recognizedPhrases with offsetMilliseconds/durationMilliseconds/nBest[0].display

        Handles missing millisecond fields by falling back to durationInTicks / 10000.
        """
        mt = parsed_data.get("machine_transcription", {})

        # Try azure_fast first (existing format)
        azure_fast = mt.get("azure_fast", {})
        phrases = azure_fast.get("phrases")
        total_duration = azure_fast.get("durationMilliseconds")
        if not total_duration:
            total_duration_ticks = azure_fast.get("durationInTicks", 0)
            if total_duration_ticks:
                total_duration = int(total_duration_ticks / 10000)

        # Fall back to whisper format
        if not phrases:
            whisper = mt.get("whisper", {})
            raw_phrases = whisper.get("recognizedPhrases", [])
            total_duration = whisper.get("durationMilliseconds")
            if not total_duration:
                total_duration_ticks = whisper.get("durationInTicks", 0)
                if total_duration_ticks:
                    total_duration = int(total_duration_ticks / 10000)
            if raw_phrases and total_duration:
                # Normalize whisper phrases to the same format as azure_fast
                phrases = []
                for rp in raw_phrases:
                    text = ""
                    confidence = 0.0
                    n_best = rp.get("nBest", [])
                    if n_best:
                        text = n_best[0].get("display", "")
                        confidence = n_best[0].get("confidence", 0.0)
                    if not text:
                        continue
                    if min_confidence > 0 and confidence < min_confidence:
                        continue
                    offset_ms = rp.get("offsetMilliseconds", 0)
                    duration_ms = rp.get("durationMilliseconds", 0)
                    if not offset_ms:
                        offset_ms = int(rp.get("offsetInTicks", 0) / 10000)
                    if not duration_ms:
                        duration_ms = int(rp.get("durationInTicks", 0) / 10000)
                    phrases.append({
                        "offsetMilliseconds": offset_ms,
                        "durationMilliseconds": duration_ms,
                        "text": text,
                    })

        if not phrases or not total_duration:
            return []

        segments = []
        current_segment_phrases = []
        current_duration = 0
        segment_start = 0

        for phrase in phrases:
            offset = phrase["offsetMilliseconds"]
            duration = phrase["durationMilliseconds"]

            current_segment_phrases.append(phrase)
            current_duration = (offset + duration) - segment_start

            if current_duration >= target_duration_ms:
                end_ms = offset + duration
                segments.append({
                    "segment_index": len(segments),
                    "start_ms": segment_start,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - segment_start,
                    "duration_sec": round((end_ms - segment_start) / 1000, 1),
                    "text": " ".join(p["text"] for p in current_segment_phrases),
                    "phrase_count": len(current_segment_phrases),
                })
                current_segment_phrases = []
                current_duration = 0
                segment_start = end_ms

        # Remaining phrases
        if current_segment_phrases:
            segments.append({
                "segment_index": len(segments),
                "start_ms": segment_start,
                "end_ms": total_duration,
                "duration_ms": total_duration - segment_start,
                "duration_sec": round((total_duration - segment_start) / 1000, 1),
                "text": " ".join(p["text"] for p in current_segment_phrases),
                "phrase_count": len(current_segment_phrases),
            })
        elif segments:
            segments[-1]["end_ms"] = total_duration
            segments[-1]["duration_ms"] = total_duration - segments[-1]["start_ms"]
            segments[-1]["duration_sec"] = round(segments[-1]["duration_ms"] / 1000, 1)

        return segments

    def _segment_feature_by_time(self, feature, segments, frame_duration_ms=10):
        """Split feature array into segments based on time boundaries."""
        segmented_features = []
        for seg in segments:
            start_frame = seg["start_ms"] // frame_duration_ms
            end_frame = math.ceil(seg["end_ms"] / frame_duration_ms)
            end_frame = min(end_frame, len(feature))

            if start_frame >= end_frame:
                logger.warning(f"Segment start frame {start_frame} is greater than or equal to end frame {end_frame}. Skipping this segment.")
                logger.warning(f"Segment info: {seg}")
                segmented_features.append(feature[0:0])  # add an empty segment to keep indexing consistent
            else:
                segmented_features.append(feature[start_frame:end_frame])
        return segmented_features

    def _segment_audio_by_time(self, audio_tuple, segments):
        """Split audio array into segments based on time boundaries."""
        audio_data, sample_rate = audio_tuple
        segmented_audios = []
        for seg in segments:
            start_sample = int(seg["start_ms"] * sample_rate / 1000)
            end_sample = int(math.ceil(seg["end_ms"] * sample_rate / 1000))
            end_sample = min(end_sample, len(audio_data))
            if start_sample >= end_sample:
                logger.warning(f"Segment start sample {start_sample} is greater than or equal to end sample {end_sample}. Skipping this segment.")
                logger.warning(f"Segment info: {seg}")
                segmented_audios.append((audio_data[0:0], sample_rate))  # add an empty segment to keep indexing consistent
            else:
                segmented_audios.append((audio_data[start_sample:end_sample], sample_rate))
        return segmented_audios

    def _maybe_segment_podcast(self, sample):
        """Expand a podcast sample into per-segment entries if it has segmentation info.

        Supports two info formats:
          1. 'machine_transcription' key (azure_fast/whisper phrases) — existing format
          2. 'segments' key with start/end in seconds and transcription — long_form format

        Splits the corresponding 'feature' or 'audio' by time boundaries.
        Returns a list of samples (one per segment, or the original sample in a list).
        """
        info = sample.get("info")
        if not isinstance(info, dict):
            return [sample]

        if "machine_transcription" in info:
            segments = self._parse_info_for_podcast(info)
        elif "segments" in info and isinstance(info["segments"], list):
            segments = self._parse_long_form_segments(info)
        else:
            return [sample]

        if not segments:
            return [sample]

        feature = sample.get("feature")
        audio = sample.get("audio")

        seg_features = None
        seg_audios = None
        if feature is not None:
            seg_features = self._segment_feature_by_time(feature, segments)
        elif audio is not None:
            if self.audio_processor is not None:
                # Extract features for full audio at once, then segment features
                audio_data, sample_rate = audio
                full_features = self.audio_processor.extract_features(audio_data, sample_rate)
                seg_features = self._segment_feature_by_time(full_features, segments)
            else:
                seg_audios = self._segment_audio_by_time(audio, segments)

        segmented = []
        for seg_idx, seg in enumerate(segments):
            new_sample = {}
            invalid_seg = False
            for key, value in sample.items():
                if key == "feature" and seg_features is not None:
                    if seg_features[seg_idx].shape[0] == 0:
                        invalid_seg = True
                    new_sample["feature"] = seg_features[seg_idx]
                elif key == "audio" and seg_audios is not None:
                    # Need at least 400 samples (25ms at 16kHz) for spectrogram extraction
                    if seg_audios[seg_idx][0].shape[0] < 400:
                        invalid_seg = True
                    new_sample["audio"] = seg_audios[seg_idx]
                elif key == "audio" and seg_features is not None:
                    # Audio was converted to features above; skip audio key
                    continue
                elif key == "info":
                    new_sample["transcription"] = seg["text"]
                    new_sample["info"] = seg
                else:
                    new_sample[key] = value
            # Assign pre-extracted segment features even if original sample had no "feature" key
            if "feature" not in new_sample and seg_features is not None:
                if seg_features[seg_idx].shape[0] == 0:
                    invalid_seg = True
                new_sample["feature"] = seg_features[seg_idx]
            elif "feature" not in new_sample:
                new_sample["feature"] = None
            if invalid_seg:
                continue
            segmented.append(new_sample)
        return segmented

    def _fill_buffer_by_length(self, example_buffer, length):
        """Fills the buffer of utterances."""
        _profile = os.environ.get("DATALOADER_PROFILE", "0") == "1"
        try:
            buffer = []
            if _profile:
                _prof = {"io": 0.0, "seg": 0.0, "trans": 0.0, "filter": 0.0, "feat": 0.0, "other": 0.0}
                _prof_n_raw, _prof_n_seg = 0, 0
                _prof_start = time.time()

            while len(buffer) < length:
                if _profile:
                    _t0 = time.time()
                raw_sample = next(self.audio_iter)
                if _profile:
                    _prof["io"] += time.time() - _t0

                if len(raw_sample) == 0:
                    continue

                if _profile:
                    _prof_n_raw += 1
                    _t0 = time.time()

                # Expand podcast samples into per-segment entries
                samples_to_process = self._maybe_segment_podcast(raw_sample)
                if _profile:
                    _prof["seg"] += time.time() - _t0
                    _prof_n_seg += len(samples_to_process)

                for sample in samples_to_process:
                    if _profile:
                        _t0 = time.time()
                    sample_processing_state = self._sample_processor.resolve_sample_processing_state(sample)

                    sample = self._sample_processor.prepare_multiaudio_sample(sample, sample_processing_state)
                    if sample is None:
                        if _profile:
                            _prof["other"] += time.time() - _t0
                        continue

                    sample_rate = None
                    if self._input_key == "audio" and "audio" in sample:
                        sample[self._input_key], sample_rate = sample[self._input_key]

                    sample = self._sample_processor.apply_transcription_processing(sample, sample_processing_state)
                    if _profile:
                        _prof["trans"] += time.time() - _t0
                    if sample is None:
                        continue

                    if "single_qa" in sample:
                        scores = sample["single_qa"]["score"]
                        if not isinstance(scores, list):
                            scores = [scores]

                        # if no qualified qa, skip the example
                        if not any(score > 8 for score in scores):
                            continue

                    if _profile:
                        _t0 = time.time()

                    # Determine filter_key: use "feature" when features are pre-extracted
                    _filter_key = "feature" if ("audio" not in sample and sample.get("feature") is not None) else None
                    if not self._sample_processor.passes_sample_filters(sample, sample_processing_state, sample_rate, filter_key=_filter_key):
                        if _profile:
                            _prof["filter"] += time.time() - _t0
                        continue
                    if _profile:
                        _prof["filter"] += time.time() - _t0

                    if _profile:
                        _t0 = time.time()
                    audio_frames, audio_embed_sizes = self._sample_processor.extract_audio_features(sample, sample_rate)
                    if _profile:
                        _prof["feat"] += time.time() - _t0

                    new_examples = self._sample_processor.materialize_examples(sample, audio_frames, audio_embed_sizes, sample_processing_state)
                    if not self.allow_pure_text_return:
                        # Keep only examples whose EVERY audio is usable by the conformer encoder:
                        #  (a) audio_embed_size > 0 -- else the encoder time reduction gives length 0
                        #      ("sequence length after time reduction is invalid: 0") or the sample
                        #      trips the is_text_sample assertion (embed formula == encoder formula
                        #      (frames-7)//8, so <15 frames -> 0).
                        #  (b) feature length <= max_utterance_frames -- the encoder self-attention is
                        #      O(L^2) (manual path, used whenever relative_k adds pos_k/pos_v), so a very
                        #      long audio OOMs the softmax. passes_sample_filters only length-checks the
                        #      FIRST audio; a long audio in a later turn would slip through, so enforce
                        #      the cap on EVERY audio here.
                        max_feat = sample_processing_state["max_utterance_frames"]
                        fs = self.feat_stride
                        kept = [
                            ex for ex in new_examples
                            if ex["audio_embed_sizes"]
                            and all(sz > 0 for sz in ex["audio_embed_sizes"])
                            and all((f // fs) <= max_feat for f in ex["audio_frames"])
                        ]
                        n_dropped = len(new_examples) - len(kept)
                        if n_dropped:
                            logger.info(f"Dropped {n_dropped}/{len(new_examples)} example(s) with zero-length or over-long audio in non-text dataloader.")
                        new_examples = kept
                    buffer.extend(new_examples)

            if _profile and self.worker_id == 0:
                _total = time.time() - _prof_start
                print(f"[DL Profile] fill_buffer: {_total:.2f}s | raw={_prof_n_raw} segs={_prof_n_seg} buf={len(buffer)} | "
                      f"data_io={_prof['io']:.2f}s({_prof['io']/_total*100:.0f}%) "
                      f"segmentation={_prof['seg']:.2f}s({_prof['seg']/_total*100:.0f}%) "
                      f"transcription={_prof['trans']:.2f}s({_prof['trans']/_total*100:.0f}%) "
                      f"filter={_prof['filter']:.2f}s({_prof['filter']/_total*100:.0f}%) "
                      f"feature_extract={_prof['feat']:.2f}s({_prof['feat']/_total*100:.0f}%) "
                      f"other={_prof['other']:.2f}s({_prof['other']/_total*100:.0f}%)", flush=True)

            example_buffer.extend(buffer)
        except StopIteration:
            # self.audio_iter is a repeating dataset and will have no stop iteration
            # just in case when self.audio_iter is not a reapting dataset in the future
            if len(buffer) > 0:
                example_buffer.extend(buffer)
            pass

    def _compute_audio_embed_size(self, audio_frames):
        if self.streaming_compression:
            result = (audio_frames - 7) // self.compression_rate
        else:
            integer = audio_frames // self.compression_rate
            remainder = audio_frames % self.compression_rate

            result = integer if remainder == 0 else integer + 1

        integer = result // self.qformer_compression_rate
        remainder = result % self.qformer_compression_rate
        result = integer if remainder == 0 else integer + 1  # qformer compression

        return result

    def _build_sorted_idx(self, example_list):
        """In format as [(idx, uttlen), (idx, uttlen), ...]"""
        min_sweep = min(example.get("sweep", 0) for example in example_list)
        leftover_buffer = []
        idx_len_pair = []
        for idx, example in enumerate(example_list):

            if example.get("sweep", 0) != min_sweep:
                leftover_buffer.append(example)
                continue

            if hasattr(self.audio_processor, "pad_to_thirty_seconds_encoder_only") and self.audio_processor.pad_to_thirty_seconds_encoder_only:
                uttlen = len(example["audio_embed_sizes"]) * self._compute_audio_embed_size(3000)
            else:
                uttlen = sum(example["audio_embed_sizes"])
            contextlen = 0
            if example.get("context", None) is not None:
                context_ids = self.tokenize.encode(example["context"], add_special_tokens=False)
                contextlen = len(context_ids)
                uttlen += contextlen

            lablen = 0

            if example["label"] is not None:
                # for stage 1, only ASR data is used
                lablen = self._max_prompt_len + len(example["label"])
            else:
                # this is for stage2 finetuning, there are multiple situations:
                # 1. when user prompt is not provided
                # 2. when user prompt is provided
                # 3. SFT data, with the user prompt in role user.
                user_prompt, annotations = example["user_prompt"], example["annotations"]
                if user_prompt is None:
                    if isinstance(annotations[0], str):
                        lablen = self._max_prompt_len + len(self.tokenize.encode(annotations[0], add_special_tokens=False))
                    else:
                        if self.conv_template in ["phi_3", "phi_o"]:
                            lablen = len(self.tokenize.encode("".join([f"<|{m['role']}|>\n{m['content']}<|end|>" for m in annotations]), add_special_tokens=False))
                        elif self.conv_template == "phi_3_200k":
                            lablen = len(self.tokenize.encode("".join([f"{m['role']}:{m['content']}<|endoftext|>" for m in annotations]), add_special_tokens=False))
                        elif self.conv_template == "phi_o_long":
                            lablen = len(self.tokenize.encode("".join([f"<|{m['role']}|>{m['content']}<|end|>" for m in annotations]), add_special_tokens=False))
                        elif self.conv_template.startswith("qwen3"):
                            lablen = len(self.tokenize.encode("".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in annotations]), add_special_tokens=False))
                        elif self.conv_template == "mistral3_audio_instruct":
                            lablen = len(self.tokenize.encode("".join([f"<s>[INST]{m['content']}[/INST]</s>" for m in annotations]), add_special_tokens=False))
                        elif self.conv_template == "gemma3_audio_instruct":
                            lablen = len(self.tokenize.encode("".join([f"<bos><start_of_turn>{m['role']}\n{m['content']}<end_of_turn>" for m in annotations]), add_special_tokens=False))
                else:
                    if self.conv_template in ["phi_3", "phi_o"]:
                        lablen = len(self.tokenize.encode(f"<|user|>\n{user_prompt}<|end|><|assistant|>\n{annotations[0]}<|end|>", add_special_tokens=False))
                    elif self.conv_template == "phi_3_200k":
                        lablen = len(self.tokenize.encode(f"user:{user_prompt}<|endofprompt|>assistant:{annotations[0]}<|endoftext|>", add_special_tokens=False))
                    elif self.conv_template == "phi_o_long":
                        lablen = len(self.tokenize.encode(f"<|user|>{user_prompt}<|end|><|assistant|>{annotations[0]}<|end|>", add_special_tokens=False))
                    elif self.conv_template.startswith("qwen3"):
                        lablen = len(self.tokenize.encode(f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{annotations[0]}<|im_end|>", add_special_tokens=False))
                    elif self.conv_template == "mistral3_audio_instruct":
                        lablen = len(self.tokenize.encode(f"<s>[INST]{user_prompt}[/INST]{annotations[0]}</s>", add_special_tokens=False))
                    elif self.conv_template == "gemma3_audio_instruct":
                        lablen = len(self.tokenize.encode(f"<bos><start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{annotations[0]}<end_of_turn>", add_special_tokens=False))
            est_answer_positions = int(example.get("est_answer_positions") or 0)
            # simul mode: answer text and answer codec are parallel channels (see
            # s2s_utils.py `max_codec_len = max(len(message_input_ids), codec_len)`),
            # so the built row length is uttlen + max(text, codec), NOT their sum.
            # Summing over-estimated ~2.5x -> systematic under-packing. max() also keeps
            # non-forward datasets (est_answer_positions=0) at lablen, i.e. unchanged.
            total_len = uttlen + max(lablen, est_answer_positions)
            if total_len > self.max_token_len:
                # if the utterance length is longer than the max sequence length, discard it
                continue
            # (C) forward samples_to_sequence (frame_sum): drop a single sample whose estimated
            # built row alone would exceed context_length (rare giant singleton). The real fix for
            # placeholder/audio_embed_sizes divergence is in the SEQ builder (split-not-truncate);
            # this only avoids feeding an oversized lone sample into that path.
            if self._condition_checker_class is SumFrameBatchConditionChecker and total_len > int(self.seq_len * 0.94):
                continue
            idx_len_pair.append((idx, total_len))

        # sort the data sort according to utterance length
        sorted_idx_len_pair = sorted(idx_len_pair, key=lambda var: var[1], reverse=True)
        return sorted_idx_len_pair, leftover_buffer


class InfiniteBlobChunkALMDataset(BlobChunkALMDataset):
    def __iter__(self):
        while True:
            yield from super().__iter__()
            logger.info(f"Epoch {self.audio_dataset.epoch} finished, starting new epoch")
            self.set_epoch(self.audio_dataset.epoch + 1)
