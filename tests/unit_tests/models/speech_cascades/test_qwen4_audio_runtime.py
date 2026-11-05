import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig


pytest.importorskip("vllm.models.qwen4_exp")
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples/models/speech_cascades"))

from speech_eval.runtime.qwen4_audio import (  # noqa: E402
    Qwen4ExpAudioForCausalLM,
    Qwen4ExpAudioProcessor,
    audio_embed_size,
)
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems


pytestmark = pytest.mark.unit


def audio_config(*, streaming=True, downsample=1):
    return PretrainedConfig(
        embd_layer={"compression_rate": 8, "downsample_rate": downsample},
        speech_streaming_compression=streaming,
        text_config=SimpleNamespace(eos_token_id=248044),
    )


@pytest.mark.parametrize(
    "frames,streaming,downsample,expected",
    [(100, True, 1, 11), (100, False, 1, 13), (100, True, 2, 6), (15, True, 1, 1)],
)
def test_audio_count_matches_training(frames, streaming, downsample, expected):
    assert audio_embed_size(frames, audio_config(streaming=streaming, downsample=downsample)) == expected


def test_streaming_audio_rejects_empty_prefix():
    with pytest.raises(ValueError, match="too short"):
        audio_embed_size(14, audio_config())


def test_audio_placeholders_are_eos_for_native_ple():
    processor = SimpleNamespace(
        info=SimpleNamespace(
            get_hf_config=audio_config,
            get_audio_embed_size=lambda samples: 11,
            get_tokenizer=lambda: SimpleNamespace(encode=lambda text, **kwargs: [101, 102]),
        )
    )
    items = SimpleNamespace(
        get_count=lambda *args, **kwargs: 1,
        get_items=lambda *args: SimpleNamespace(get_audio_length=lambda index: 16000),
    )
    updates = Qwen4ExpAudioProcessor._get_prompt_updates(processor, items, {}, {})
    assert updates[0].target == [101, 102]
    assert updates[0].replacement(0) == [248044] * 11
    assert Qwen4ExpAudioForCausalLM.requires_raw_input_tokens is True


def test_native_split_preprocessing_extracts_fbank_without_image_processor():
    items = MultiModalDataItems({"audio": AudioProcessorItems([np.zeros(16_000, dtype=np.float32)])})
    processed = Qwen4ExpAudioProcessor._apply_hf_processor_main(object(), items, {})
    assert set(processed) == {"input_audio_embeds"}
    assert len(processed["input_audio_embeds"]) == 1
    assert processed["input_audio_embeds"][0].shape == (98, 80)
    assert torch.isfinite(processed["input_audio_embeds"][0]).all()


def test_native_split_preprocessing_accepts_text_only():
    processed = Qwen4ExpAudioProcessor._apply_hf_processor_main(object(), MultiModalDataItems({}), {})
    assert dict(processed) == {}
    assert Qwen4ExpAudioProcessor._get_prompt_updates(object(), MultiModalDataItems({}), {}, {}) == []


def test_native_split_preprocessing_rejects_missing_audio():
    items = MultiModalDataItems({"audio": AudioProcessorItems([None])})
    with pytest.raises(ValueError, match="missing audio"):
        Qwen4ExpAudioProcessor._apply_hf_processor_main(object(), items, {})


def test_speculative_configuration_is_rejected_before_model_allocation():
    with pytest.raises(ValueError, match="without MTP"):
        Qwen4ExpAudioForCausalLM(vllm_config=SimpleNamespace(speculative_config=object()))


def test_native_mapper_skips_vision_weights_in_shared_backbone_shards():
    tensor = torch.zeros(1)
    weights = [
        ("model.visual.blocks.0.weight", tensor),
        ("model.language_model.layers.0.weight", tensor),
        ("model.embed_tokens_extend.encoder.weight", tensor),
    ]
    mapped = list(Qwen4ExpAudioForCausalLM.hf_to_vllm_mapper.apply(weights))
    assert [name for name, _ in mapped] == ["model.layers.0.weight", "embed_tokens_extend.encoder.weight"]


def test_audio_embedding_trims_encoder_to_training_prefix():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, feature):
            return torch.arange(13 * 4, dtype=feature.dtype).reshape(13, 4)

    model = SimpleNamespace(embed_tokens_extend=Encoder(), model_config=SimpleNamespace(hf_config=audio_config()))
    embeddings = Qwen4ExpAudioForCausalLM.embed_multimodal(model, input_audio_embeds=[torch.zeros(100, 80)])
    assert len(embeddings) == 1
    assert embeddings[0].shape == (11, 4)
    assert torch.equal(embeddings[0], torch.arange(11 * 4, dtype=torch.float32).reshape(11, 4))
