"""vLLM speech-input adapter for the Qwen3.5 text backbone."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import scipy.fft
import scipy.signal
import torch
from torch import nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    _require_is_multimodal,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.phi4mm import Phi4MMAudioFeatureInputs
from vllm.model_executor.models.phi4mm_audio import AudioEmbedding
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase, Qwen3_5MoeForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)


_AUDIO_MAX_SAMPLES = 640_000
_PLACEHOLDER_TOKEN_ID = 0


def _speechlib_mel() -> np.ndarray:
    sample_rate = 16_000
    n_fft = 512
    n_mels = 80
    fmax = 7_690
    mel = lambda frequency: 1127.0 * np.log(1.0 + frequency / 700.0)
    bin_to_mel = lambda index: 1127.0 * np.log(1.0 + index * sample_rate / (n_fft * 700.0))
    first_bin = 1
    last_bin = int(fmax * n_fft / sample_rate + 0.5)
    centers = np.linspace(mel(0), mel(fmax), n_mels + 2)
    spacing = (centers[-1] - centers[0]) / (n_mels + 1)
    matrix = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for mel_index in range(n_mels):
        left, center, right = centers[mel_index : mel_index + 3]
        for fft_bin in range(first_bin, last_bin):
            value = bin_to_mel(fft_bin)
            if left < value < right:
                matrix[mel_index, fft_bin] = 1.0 - abs(center - value) / spacing
    return matrix.T


_MEL_FILTER = _speechlib_mel()
_HAMMING = np.hamming(400)


def _extract_fbank(waveform: np.ndarray, sample_rate: int) -> torch.Tensor:
    waveform = np.asarray(waveform)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    if sample_rate != 16_000:
        if sample_rate < 8_000:
            raise ValueError(f"Unsupported audio sample rate: {sample_rate}")
        waveform = scipy.signal.resample_poly(waveform, 16_000, sample_rate)

    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if waveform.size < 400:
        waveform = np.pad(waveform, (0, 400 - waveform.size))
    frame_count = (waveform.size - 400) // 160 + 1
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(frame_count, 400),
        strides=(waveform.strides[0] * 160, waveform.strides[0]),
    )
    previous = np.empty((frame_count, 400), dtype=np.float32)
    previous[:, 0] = frames[:, 0]
    previous[:, 1:] = frames[:, :-1]
    emphasized = (frames - 0.97 * previous) * 32_768
    spectrum = scipy.fft.rfft(_HAMMING * emphasized, n=512, axis=1).astype(np.complex64)
    power = np.abs(spectrum).astype(np.float32) ** 2
    features = np.log(np.clip(power.dot(_MEL_FILTER), 1.0, None)).astype(np.float32)
    return torch.from_numpy(features)


class _ProjectorRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        values = hidden_states.float()
        values *= torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return (values * self.weight.float()).to(dtype)


class _SquaredReLU(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.relu(hidden_states).square()


class _GatedProj(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.norm = _ProjectorRMSNorm(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
        self.activation = _SquaredReLU()
        self.linear2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(self.norm(hidden_states))))


class _SwiGLU(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values, gate = hidden_states.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * values


class _UltravoxProj(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.ln_pre = _ProjectorRMSNorm(input_size)
        self.linear_1 = nn.Linear(input_size, hidden_size, bias=False)
        self.act = _SwiGLU()
        self.ln_mid = _ProjectorRMSNorm(hidden_size // 2)
        self.linear_2 = nn.Linear(hidden_size // 2, output_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.ln_mid(self.act(self.linear_1(self.ln_pre(hidden_states)))))


class Qwen3_5AudioEmbedding(AudioEmbedding):
    def __init__(self, config: Any, **kwargs: Any):
        projection_cls = kwargs.get("projection_cls", "linear")
        if projection_cls in ("linear", "mlp"):
            super().__init__(config, **kwargs)
            if hasattr(self, "audio_projection_for_vision"):
                del self.audio_projection_for_vision
            return

        base_kwargs = dict(kwargs)
        base_kwargs["projection_cls"] = "linear"
        super().__init__(config, **base_kwargs)
        self.linear_downsample_rate = kwargs.get("downsample_rate", 1)
        input_size = self.audio_dim_out * self.linear_downsample_rate
        output_size = config.hidden_size
        if projection_cls == "lfg3":
            self.audio_projection = _GatedProj(input_size, output_size, kwargs.get("projector_hidden", 4096))
        elif projection_cls == "ultravox":
            hidden_size = kwargs.get("projector_hidden")
            if hidden_size is None:
                target = kwargs.get("projector_target_params", 50_000_000)
                hidden_size = int(round(target / (input_size + output_size / 2)))
            hidden_size = max(2, hidden_size - hidden_size % 2)
            self.audio_projection = _UltravoxProj(input_size, output_size, hidden_size)
        else:
            raise ValueError(f"Unsupported speech projector: {projection_cls}")


class Qwen3_5AudioProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self):
        return MultiModalDataParser(target_sr=16_000, audio_resample_method="scipy")

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_audio_embed_size(self, sample_count: int) -> int:
        frame_count = max(1, (sample_count - 400) // 160 + 1)
        config = self.get_hf_config()
        encoder_config = config.audio_processor["config"]
        time_reduction = encoder_config.get("time_reduction", 1)
        encoder_tokens = (frame_count + time_reduction - 1) // time_reduction
        projector_rate = config.embd_layer.get("downsample_rate", 1)
        return (encoder_tokens + projector_rate - 1) // projector_rate


class Qwen3_5AudioDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3_5AudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<audio>" * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        return {
            "audio": self._get_dummy_audios(
                length=_AUDIO_MAX_SAMPLES,
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class Qwen3_5AudioMultiModalProcessor(BaseMultiModalProcessor[Qwen3_5AudioProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, **tok_kwargs)
        if not mm_data:
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature({"input_ids": torch.tensor([prompt_ids], dtype=torch.long)})

        audio_data = mm_data.get("audios", [])
        features = [_extract_fbank(audio, 16_000) for audio in audio_data]
        return BatchFeature(
            {
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
                "input_audio_embeds": features,
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {"input_audio_embeds": MultiModalFieldConfig.batched("audio")}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        audios = mm_items.get_items("audio", AudioProcessorItems)
        tokenizer = self.info.get_tokenizer()
        placeholder = "<|audio_pad|>"
        placeholder_ids = tokenizer.encode(placeholder, add_special_tokens=False)
        if (
            len(placeholder_ids) != 1
            or tokenizer.decode(placeholder_ids) != placeholder
            or placeholder_ids[0] not in tokenizer.all_special_ids
        ):
            raise ValueError("Speech serving requires the existing <|audio_pad|> special token")
        return [
            PromptReplacement(
                modality="audio",
                target="<audio>",
                replacement=lambda index: placeholder_ids
                * self.info.get_audio_embed_size(audios.get_audio_length(index)),
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3_5AudioMultiModalProcessor,
    info=Qwen3_5AudioProcessingInfo,
    dummy_inputs=Qwen3_5AudioDummyInputsBuilder,
)
class Qwen3_5AudioForCausalLM(Qwen3_5ForCausalLMBase, SupportsMultiModal, SupportsMRoPE):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={"base_layer.": ""},
        orig_to_new_prefix={
            "language_model.": "",
            "model.embed_tokens_extend.": "embed_tokens_extend.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, index: int) -> str | None:
        if modality == "audio":
            return "<audio>"
        raise ValueError("Qwen3.5 speech-in supports only audio inputs")

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[Any],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.expand(3, -1).contiguous(), 0

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        if config.model_type == "qwen3_5_moe_text":
            Qwen3_5MoeForCausalLM.set_moe_parameters(self)
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None, "multimodal_config is required"
        embedding_config = dict(config.embd_layer)
        with self._mark_tower_model(vllm_config, "audio"):
            self.embed_tokens_extend = Qwen3_5AudioEmbedding(config, **embedding_config)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        features = kwargs.pop("input_audio_embeds", None)
        if features is None:
            return []
        audio_input = Phi4MMAudioFeatureInputs(type="audio_features", audio_features=features)
        dtype = next(self.embed_tokens_extend.parameters()).dtype
        return tuple(self.embed_tokens_extend(item.to(dtype)) for item in audio_input["audio_features"])

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(input_ids, self.model.embed_input_ids, is_multimodal=is_multimodal)
        if not multimodal_embeddings:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."], skip_substrs=["lora_"])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="model.",
            connector="embed_tokens_extend.audio_projection",
            tower_model="embed_tokens_extend.encoder",
        )
