# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-MTP Cascades adapter for the native Qwen4Exp vLLM runtime."""

from collections.abc import Mapping, Sequence

import torch
from transformers import BatchFeature, PretrainedConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.qwen4_exp import Qwen4ExpForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing.processor import PromptReplacement, PromptUpdate

from .qwen3_5_audio import (
    Qwen3_5AudioDummyInputsBuilder,
    Qwen3_5AudioEmbedding,
    Qwen3_5AudioForCausalLM,
    Qwen3_5AudioMultiModalProcessor,
    Qwen3_5AudioProcessingInfo,
    _extract_fbank,
)


def audio_embed_size(frame_count: int, config: PretrainedConfig) -> int:
    """Match the training collator's streaming truncation and post-encoder downsampling."""
    rate = int(config.embd_layer["compression_rate"])
    count = (frame_count - 7) // rate if config.speech_streaming_compression else (frame_count + rate - 1) // rate
    downsample = int(config.embd_layer["downsample_rate"])
    count = (count + downsample - 1) // downsample
    if count <= 0:
        raise ValueError("Audio is too short for the trained speech compression contract")
    return count


class Qwen4ExpAudioProcessingInfo(Qwen3_5AudioProcessingInfo):
    """Use exactly the audio token count used by the training data manifest."""

    def get_audio_embed_size(self, sample_count: int) -> int:
        """Return the number of retained projected frames for a 16 kHz waveform."""
        frames = max(1, (sample_count - 400) // 160 + 1)
        return audio_embed_size(frames, self.get_hf_config())


class Qwen4ExpAudioProcessor(Qwen3_5AudioMultiModalProcessor):
    """Keep the training-time EOS-valued audio IDs in the PLE n-gram history."""

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_items.get_count("audio", strict=False):
            return BatchFeature({})
        audios = mm_items.get_items("audio", AudioProcessorItems)
        features = []
        for index in range(audios.get_count()):
            audio = audios.get(index)
            if audio is None:
                raise ValueError(f"Cannot extract features from a missing audio item at {index}")
            features.append(_extract_fbank(audio, 16_000))
        return BatchFeature({"input_audio_embeds": features})

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        if not mm_items.get_count("audio", strict=False):
            return []
        audios = mm_items.get_items("audio", AudioProcessorItems)
        eos = self.info.get_hf_config().text_config.eos_token_id
        if type(eos) is not int or eos < 0:
            raise ValueError("Qwen4Exp speech requires a single EOS token ID for PLE audio positions")
        return [
            PromptReplacement(
                modality="audio",
                target=self.info.get_tokenizer().encode("<audio>", add_special_tokens=False),
                replacement=lambda index: [eos] * self.info.get_audio_embed_size(audios.get_audio_length(index)),
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen4ExpAudioProcessor,
    info=Qwen4ExpAudioProcessingInfo,
    dummy_inputs=Qwen3_5AudioDummyInputsBuilder,
)
class Qwen4ExpAudioForCausalLM(Qwen4ExpForCausalLM, SupportsMultiModal):
    """Reuse native QSA/GDN/PLE and the existing Cascades tower without speculative decoding."""

    hf_to_vllm_mapper = Qwen4ExpForCausalLM.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={"model.embed_tokens_extend.": "embed_tokens_extend.", "model.visual.": None}
    )
    get_placeholder_str = Qwen3_5AudioForCausalLM.get_placeholder_str
    embed_input_ids = Qwen3_5AudioForCausalLM.embed_input_ids

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.speculative_config is not None:
            raise ValueError("Qwen4Exp speech evaluation requires standard decoding without MTP")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        if config.audio_processor["name"] != "cascades":
            raise ValueError("Expected the Cascades audio encoder")
        with self._mark_tower_model(vllm_config, "audio"):
            self.embed_tokens_extend = Qwen3_5AudioEmbedding(config, **dict(config.embd_layer))

    def embed_multimodal(self, **kwargs: torch.Tensor | list[torch.Tensor]) -> MultiModalEmbeddings:
        """Trim the encoder output to the same valid-frame prefix as the trained model."""
        features = kwargs.pop("input_audio_embeds", None)
        if features is None:
            return []
        dtype = next(self.embed_tokens_extend.parameters()).dtype
        embeddings = []
        for feature in features:
            count = audio_embed_size(feature.shape[0], self.model_config.hf_config)
            output = self.embed_tokens_extend(feature.to(dtype))
            if output.shape[0] < count:
                raise ValueError("Encoder returned fewer frames than the training audio-token contract")
            embeddings.append(output[:count])
        return tuple(embeddings)

    def get_mm_mapping(self) -> MultiModelKeys:
        """Identify the audio tower and connector independently of the Qwen4Exp language model."""
        return MultiModelKeys.from_string_field(
            language_model="model.",
            connector="embed_tokens_extend.audio_projection",
            tower_model="embed_tokens_extend.encoder",
        )
