"""Unit tests for the WP-A1 teacher text-forward data path (KD)."""

import pytest


torch = pytest.importorskip("torch")

from megatron.bridge.data.speech_chunk import ALMDataCollator  # noqa: E402
from megatron.bridge.data.speech_chunk.teacher_text import (  # noqa: E402
    build_teacher_text_example,
    derive_transcript_ids,
)
from megatron.bridge.models.speech_cascades.step import get_alm_batch  # noqa: E402


IGNORE_INDEX = -100


def _iter_once(batch):
    yield batch


def _response_labels(labels: torch.Tensor) -> torch.Tensor:
    """Causal-shift then mask, mirroring the student/teacher response selection."""
    labels = labels.reshape(-1)
    shifted = labels[1:]
    return shifted[shifted != IGNORE_INDEX]


def _make_kd_record(audio_size=3, transcript=(300, 301), answer=(200, 201, 202), feat_dim=8):
    """Synthetic ASR-style example: prompt, one audio span (negative ids), then a teacher-forced answer."""
    input_ids = [100, 101] + [-1] * audio_size + [102] + list(answer)
    labels = [IGNORE_INDEX] * (3 + audio_size) + list(answer)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "input_embeds": [torch.ones(audio_size * 2, feat_dim)],
        "audio_embed_sizes": torch.tensor([audio_size], dtype=torch.long),
        "audio_frames": torch.tensor([audio_size * 2], dtype=torch.long),
        "is_text_sample": False,
        "transcript_ids": [torch.tensor(list(transcript), dtype=torch.long)],
    }


def test_build_teacher_text_example_replaces_span_and_keeps_answer():
    answer = (200, 201, 202)
    transcript = (300, 301)
    record = _make_kd_record(audio_size=3, transcript=transcript, answer=answer)

    teacher_input_ids, teacher_labels = build_teacher_text_example(
        record["input_ids"], record["labels"], record["transcript_ids"]
    )

    # K=3 placeholders replaced by the T=2 transcript tokens, prompt/answer text kept verbatim.
    assert teacher_input_ids.tolist() == [100, 101, 300, 301, 102, 200, 201, 202]
    # Answer segment is byte-identical to the student input tail.
    assert teacher_input_ids[-len(answer) :].tolist() == list(answer)
    assert teacher_labels.tolist() == [IGNORE_INDEX] * 5 + list(answer)
    assert teacher_input_ids.dtype == torch.long
    assert teacher_labels.dtype == torch.long


def test_teacher_response_labels_align_with_student():
    record = _make_kd_record()
    _, teacher_labels = build_teacher_text_example(
        record["input_ids"], record["labels"], record["transcript_ids"]
    )

    assert torch.equal(_response_labels(record["labels"]), _response_labels(teacher_labels))


def test_build_teacher_text_example_raises_on_missing_transcript():
    input_ids = torch.tensor([-1, -2, 5], dtype=torch.long)
    labels = torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 5], dtype=torch.long)
    with pytest.raises(ValueError, match="no transcript|only 1 transcript"):
        build_teacher_text_example(input_ids, labels, [torch.tensor([9], dtype=torch.long)])


def test_derive_transcript_ids_single_segment():
    """ASR response labels become the segment transcript when transcript_ids is absent."""
    record = _make_kd_record(audio_size=3, answer=(200, 201, 202))
    transcripts = derive_transcript_ids(record["input_ids"], record["labels"])
    assert len(transcripts) == 1
    # The transcript is the supervised response run following the audio placeholder span.
    assert transcripts[0].tolist() == [200, 201, 202]
    assert transcripts[0].dtype == torch.long


def test_derive_transcript_ids_multi_segment_packed():
    """A packed multi-utterance sequence yields one transcript per audio segment, in order."""
    # [prompt, audio0(-1 x2), ans0, prompt, audio1(-2 x3), ans1]
    input_ids = torch.tensor([100, -1, -1, 200, 201, 101, -2, -2, -2, 300], dtype=torch.long)
    labels = torch.tensor(
        [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 200, 201, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 300],
        dtype=torch.long,
    )
    transcripts = derive_transcript_ids(input_ids, labels)
    assert [t.tolist() for t in transcripts] == [[200, 201], [300]]


def test_derive_transcript_ids_raises_on_missing_response():
    input_ids = torch.tensor([100, -1, -1, 102], dtype=torch.long)
    labels = torch.tensor([IGNORE_INDEX] * 4, dtype=torch.long)
    with pytest.raises(ValueError):
        derive_transcript_ids(input_ids, labels)


def test_collator_emits_teacher_batch_when_kd_enabled():
    collator = ALMDataCollator(pad_token_id=0, knowledge_distillation=True)
    records = [
        _make_kd_record(audio_size=3, transcript=(300, 301), answer=(200, 201, 202)),
        _make_kd_record(audio_size=2, transcript=(400, 401, 402), answer=(210, 211)),
    ]
    batch = collator(records)

    assert set(batch) >= {"teacher_input_ids", "teacher_labels", "teacher_attention_mask"}
    assert batch["teacher_input_ids"].dtype == torch.long
    assert batch["teacher_labels"].dtype == torch.long
    assert batch["teacher_attention_mask"].dtype == torch.bool
    b, t = batch["teacher_input_ids"].shape
    assert b == 2
    assert batch["teacher_labels"].shape == (b, t)
    assert batch["teacher_attention_mask"].shape == (b, t)
    # First example: 2 prompt + 2 transcript + 1 prompt + 3 answer = 8 real tokens.
    assert batch["teacher_attention_mask"][0].sum().item() == 8
    assert batch["teacher_input_ids"][0, :8].tolist() == [100, 101, 300, 301, 102, 200, 201, 202]
    # Per-example teacher response labels stay aligned with the student labels.
    for row in range(b):
        assert torch.equal(
            _response_labels(batch["labels"][row]), _response_labels(batch["teacher_labels"][row])
        )


def test_collator_kd_disabled_is_byte_identical():
    records = [_make_kd_record(), _make_kd_record(audio_size=2)]
    baseline = ALMDataCollator(pad_token_id=0)(records)
    kd_off = ALMDataCollator(pad_token_id=0, knowledge_distillation=True)(
        [{k: v for k, v in r.items() if k != "transcript_ids"} for r in records]
    )

    assert "teacher_input_ids" not in baseline
    assert "teacher_input_ids" not in kd_off
    assert baseline.keys() == kd_off.keys()
    assert torch.equal(baseline["input_ids"], kd_off["input_ids"])
    assert torch.equal(baseline["labels"], kd_off["labels"])


def test_collator_rejects_mixed_teacher_examples():
    collator = ALMDataCollator(pad_token_id=0, knowledge_distillation=True)
    with_teacher = _make_kd_record()
    without_teacher = {k: v for k, v in _make_kd_record().items() if k != "transcript_ids"}
    with pytest.raises(ValueError, match="cannot mix"):
        collator([with_teacher, without_teacher])


def test_get_alm_batch_passes_teacher_tensors():
    collator = ALMDataCollator(pad_token_id=0, knowledge_distillation=True)
    batch = collator([_make_kd_record(), _make_kd_record(audio_size=2)])

    out = get_alm_batch(_iter_once(batch))

    assert out["teacher_input_ids"].dtype == torch.long
    assert out["teacher_labels"].dtype == torch.long
    assert out["teacher_attention_mask"].dtype == torch.bool
    assert out["teacher_input_ids"].shape == batch["teacher_input_ids"].shape


def test_get_alm_batch_omits_teacher_tensors_when_absent():
    collator = ALMDataCollator(pad_token_id=0)
    batch = collator([_make_kd_record(), _make_kd_record(audio_size=2)])

    out = get_alm_batch(_iter_once(batch))

    assert "teacher_input_ids" not in out
    assert "teacher_labels" not in out
    assert "teacher_attention_mask" not in out
