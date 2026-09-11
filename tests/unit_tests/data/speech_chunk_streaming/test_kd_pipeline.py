"""Streaming source transcript and teacher-field propagation for S2TT."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.alm_sample_processor import (
    BlobChunkALMSampleProcessor,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.blob_chunk_alm_dataset import BlobChunkALMDataset
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.audio_text_conversation import (
    AudioTextConversationTransform,
)
from megatron.bridge.data.speech_chunk_streaming.data_collator import ALMDataDynamicCollator


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]

    def decode(self, tokens):
        return "".join(chr(int(token) - 1) for token in tokens if token > 0)


def processor():
    return BlobChunkALMSampleProcessor(
        tokenizer=CharacterTokenizer(), audio_processor=None, feat_stride=1,
        input_key="feature", final_audio_key="features", allow_pure_text_return=False,
        retain_first_n_turns_in_conversation=None, default_filter_options={},
        default_text_normalization=None, default_sp_transcription=False,
        compute_audio_embed_size=lambda frames: (frames - 7) // 8,
    )


@pytest.mark.unit
@pytest.mark.parametrize("answer", [None, "Paris."])
def test_processor_separates_rewrite_answer_from_transcript(answer):
    instance = processor()
    sample = {"transcription": "Where is the Eiffel Tower?", "language": "en", "features": [torch.zeros(80, 80)]}
    if answer is not None:
        sample["answer"] = answer
    state = instance.resolve_sample_processing_state(sample)
    processed = instance.apply_transcription_processing(sample, state)
    example = instance.materialize_examples(processed, [80], [9], state)[0]
    assert example["annotations"] == [answer or sample["transcription"]]
    assert example["label"] == instance.tokenizer.encode(answer or sample["transcription"])
    assert example["audio_transcripts"] == ([sample["transcription"]] if answer else None)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["audio-only", "audio-question", "text-question"])
def test_sft_answer_overlay_retains_original_transcript(mode):
    instance = processor()
    user = "Context: <|audio_0|>" if mode != "text-question" else "Where is Paris?"
    sample = {"sft": [{"answer_overlay_mode": mode, "messages": [
        {"role": "user", "content": user}, {"role": "assistant", "content": "Where is Paris?"},
    ]}], "answer": "In France."}
    original = deepcopy(sample)
    result = instance.prepare_multiaudio_sample(sample, instance.resolve_sample_processing_state(sample))
    assert result["sft"][0]["messages"][-1]["content"] == "In France."
    assert original["sft"][0]["messages"][-1]["content"] == "Where is Paris?"
    if mode != "text-question":
        assert result["audio_transcripts"] == ["Where is Paris?"]
    assert result["sft"][0]["messages"][0]["content"] == ("<|audio_0|>" if mode == "audio-only" else user)


@pytest.mark.unit
def test_dataset_kd_preparation_rejects_missing_rewrite_transcript():
    dataset = object.__new__(BlobChunkALMDataset)
    dataset.transforms = AudioTextConversationTransform(
        {"pad_to_context_length": False}, conv_template="qwen3_audio_instruct",
        prompt_formatter="audio-response", knowledge_distillation=True,
    )
    example = {"annotations": ["Paris."], "audio_embed_sizes": [3], "language": "en"}
    assert not dataset._prepare_example_for_distillation(example)
    example["audio_transcripts"] = ["Where is the Eiffel Tower?"]
    assert dataset._prepare_example_for_distillation(example)


@pytest.mark.unit
def test_dataset_producer_preserves_teacher_fields():
    dataset = object.__new__(BlobChunkALMDataset)
    dataset.transforms = AudioTextConversationTransform(
        {"pad_to_context_length": False}, conv_template="qwen3_audio_instruct",
        prompt_formatter="audio-response", knowledge_distillation=True,
    )
    dataset.tokenize = CharacterTokenizer()
    dataset._prefetch_stop = False
    dataset._buffer_size = 1
    dataset._final_audio_key = "features"
    dataset.allow_pure_text_return = False
    dataset.get_max_prompt_len = lambda: 0
    class AudioIterator:
        epoch = 0
        worker_step_in_epoch = 0

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    dataset.audio_dataset = AudioIterator()
    example = {
        "features": [torch.zeros(80, 80)], "annotations": ["Paris."],
        "audio_transcripts": ["Where is the Eiffel Tower?"], "audio_embed_sizes": [3],
        "audio_frames": [80], "user_prompt": None, "language": "en",
    }
    dataset._fill_buffer_by_length = lambda buffer, length: buffer.append(deepcopy(example))
    dataset._create_batch_list = lambda: ([[deepcopy(example)]], [deepcopy(example)])
    records = []

    def put(sample):
        if sample is not None:
            records.append(sample)
        return False

    dataset._queue_put_with_stop = put
    dataset._producer_loop()
    assert len(records) == 1, getattr(dataset, "_prefetch_exception", None)
    assert "Where is the Eiffel Tower?" in dataset.tokenize.decode(records[0]["teacher_input_ids"])
    assert torch.equal(
        records[0]["labels"][records[0]["labels"] != -100],
        records[0]["teacher_labels"][records[0]["teacher_labels"] != -100],
    )
    batch = collator()(deepcopy(records))
    assert torch.equal(batch["teacher_input_ids"][0], records[0]["teacher_input_ids"])


def collator():
    instance = object.__new__(ALMDataDynamicCollator)
    instance.tokenizer = CharacterTokenizer()
    instance.knowledge_distillation = True
    instance.audio_key = "inputs_embeds"
    instance.padding_mask = True
    return instance


def teacher_example():
    return {
        "input_ids": torch.tensor([10, -1, -1, 30, 31]),
        "labels": torch.tensor([-100, -100, -100, 30, 31]),
        "teacher_input_ids": torch.tensor([10, 20, 30, 31]),
        "teacher_labels": torch.tensor([-100, -100, 30, 31]),
        "teacher_attention_mask": torch.tensor([True, True, True, True]),
    }


@pytest.mark.unit
def test_collator_prefers_template_teacher_and_preserves_padding():
    first = teacher_example()
    first["transcript_ids"] = [torch.tensor([999])]
    second = {key: value.unsqueeze(0).repeat(2, 1) for key, value in teacher_example().items()}
    for key, pad in (("teacher_input_ids", 0), ("teacher_labels", -100), ("teacher_attention_mask", False)):
        second[key] = torch.nn.functional.pad(second[key], (0, 2), value=pad)
    batch = {}
    collator()._add_teacher_batch([first, second], batch)
    assert batch["teacher_input_ids"].shape == (3, 6)
    assert batch["teacher_input_ids"][0].tolist() == [10, 20, 30, 31, 0, 0]
    assert batch["teacher_labels"][0].tolist() == [-100, -100, 30, 31, -100, -100]
    assert batch["teacher_attention_mask"].sum(dim=1).tolist() == [4, 4, 4]


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["teacher_input_ids", "teacher_labels", "teacher_attention_mask", "all"])
def test_collator_rejects_partial_or_mixed_teacher_fields(missing):
    first, second = teacher_example(), teacher_example()
    for key in list(second):
        if key == missing or (missing == "all" and key.startswith("teacher_")):
            del second[key]
    with pytest.raises(ValueError, match="all three teacher fields"):
        collator()._add_teacher_batch([first, second], {})


@pytest.mark.unit
def test_collator_never_infers_rewrite_transcript_from_answer():
    example = {key: value for key, value in teacher_example().items() if not key.startswith("teacher_")}
    with pytest.raises(ValueError, match="response labels are not a source transcript"):
        collator()._add_teacher_batch([example], {})
    example["transcript_ids"] = [torch.tensor([20])]
    batch = {}
    collator()._add_teacher_batch([example], batch)
    assert batch["teacher_input_ids"].tolist() == [[10, 20, 30, 31]]


@pytest.mark.unit
@pytest.mark.parametrize("kd_enabled", [False, True])
def test_streaming_provider_forwards_kd_to_transform(monkeypatch, tmp_path, kd_enabled):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    import megatron.bridge.data.speech_chunk_streaming as streaming
    from megatron.bridge.data.speech_chunk.streaming_provider import SpeechChunkStreamingDatasetProvider

    entries = [
        {"tokenizer": {"pretrained_model_name_or_path": "/unused"}, "transform_cls": {
            "name": "AudioTextConversationTransform", "prompt_formatter": formatter,
        }} for formatter in ("asr", "audio-response", "audio-continuation", "ast-cot")
    ]
    path = tmp_path / "data.yaml"
    OmegaConf.save({"dataset": entries, "dataset_collator": {"knowledge_distillation": kd_enabled}}, path)
    seen = []

    def build(dataset_entries, **kwargs):
        seen.extend(dataset_entries)
        return [{"sentinel": True}]

    monkeypatch.setattr(streaming, "build_speech_chunk_dataset", build)
    monkeypatch.setattr(streaming, "build_speech_chunk_collator", lambda *args: lambda records: records[0])
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: CharacterTokenizer())
    monkeypatch.setattr(SpeechChunkStreamingDatasetProvider, "_data_parallel_rank_and_size", staticmethod(lambda _: (0, 1)))
    provider = SpeechChunkStreamingDatasetProvider(seq_length=8192, speech_config_yaml=str(path), num_workers=0)
    train, _, _ = provider.build_datasets(SimpleNamespace())
    assert next(train) == {"sentinel": True}
    assert len(seen) == 4
    for entry in seen:
        assert entry["transform_cls"].get("knowledge_distillation", False) == kd_enabled


@pytest.mark.unit
@pytest.mark.parametrize("formatter", ["asr", "ast-cot", "audio-response", "audio-continuation"])
def test_raw_annotation_to_teacher_batch(formatter):
    dataset = object.__new__(BlobChunkALMDataset)
    dataset.transforms = AudioTextConversationTransform(
        {"pad_to_context_length": False}, conv_template="qwen3_audio_instruct",
        prompt_formatter=formatter, knowledge_distillation=True,
    )
    dataset.tokenize = CharacterTokenizer()
    dataset._sample_processor = processor()
    dataset._prefetch_stop = False
    dataset._buffer_size = 1
    dataset._input_key = "feature"
    dataset._final_audio_key = "features"
    dataset.feat_stride = 1
    dataset.allow_pure_text_return = False
    dataset.get_max_prompt_len = lambda: 0
    dataset._maybe_segment_podcast = lambda sample: [sample]
    transcript = "Where is Paris?"
    sample = {"transcription": transcript, "feature": torch.zeros(80, 80), "language": "en"}
    if formatter in {"audio-response", "audio-continuation"}:
        sample["answer"] = "In France."
    elif formatter == "ast-cot":
        sample["transcription"] = transcript + "<sep>Translation."
        sample["language"] = ["fr", "en"]

    class AudioIterator:
        epoch = 0
        worker_step_in_epoch = 0

        def __init__(self):
            self.samples = iter([sample])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.samples)

    dataset.audio_dataset = AudioIterator()
    dataset._create_batch_list = lambda: ([list(dataset._example_buffer)], [])
    records = []

    def put(record):
        if record is not None:
            records.append(record)
        return False

    dataset._queue_put_with_stop = put
    dataset._producer_loop()
    assert len(records) == 1, getattr(dataset, "_prefetch_exception", None)
    result = collator()(deepcopy(records))
    assert transcript in dataset.tokenize.decode(result["teacher_input_ids"][0])
    assert not (result["teacher_input_ids"] < 0).any()
    assert torch.equal(
        result["labels"][result["labels"] != -100],
        result["teacher_labels"][result["teacher_labels"] != -100],
    )
    if "answer" in sample:
        supervised = dataset.tokenize.decode(result["labels"][result["labels"] != -100])
        assert "In France." in supervised
        assert transcript not in supervised


@pytest.mark.unit
def test_plain_non_kd_batch_keeps_existing_fields():
    instance = collator()
    instance.knowledge_distillation = False
    record = {key: value for key, value in teacher_example().items() if not key.startswith("teacher_")}
    batch = instance([deepcopy(record)])
    assert not any(key.startswith("teacher_") for key in batch)
    assert torch.equal(batch["input_ids"][0], record["input_ids"])
    assert torch.equal(batch["labels"][0], record["labels"])
    record["teacher_labels"] = torch.tensor([-100, 30, 31])
    with pytest.raises(ValueError, match="all three teacher fields"):
        instance([record])