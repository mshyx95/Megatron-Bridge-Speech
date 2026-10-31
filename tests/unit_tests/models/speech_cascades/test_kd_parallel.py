from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from megatron.bridge.data.speech_chunk.streaming_provider import SpeechChunkStreamingDatasetProvider
from megatron.bridge.models.speech_cascades.distillation import chunked_kl_divergence
from megatron.bridge.models.speech_cascades.model import SpeechCascadesModel
from megatron.bridge.models.speech_cascades.step import _make_logits_fn, forward_step, get_alm_batch
from megatron.bridge.training.config import DatasetBuildContext


class _ParallelHead(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, *args, **kwargs):
        raise AssertionError("KD must not invoke the SP head on already gathered response tokens")


class _ParallelLanguageModel(nn.Module):
    def __init__(self, weight, bias, tied):
        super().__init__()
        self.output_layer = _ParallelHead(weight, bias)
        self.pg_collection = SimpleNamespace(tp=dist.group.WORLD)
        self.tied = tied
        self.share_embeddings_and_output_weights = tied

    def shared_embedding_or_output_weight(self):
        assert self.tied, "An untied LM must not use its input embedding for the KD output head"
        return self.output_layer.weight

    def _scale_logits(self, logits):
        return logits * 0.75


def _parallel_kl_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        for tied in (False, True):
            for sequence_parallel in (False, True):
                torch.manual_seed(73)
                weight = torch.randn(14, 8)
                bias = torch.randn(14)
                student = torch.randn(7, 2, 8, requires_grad=True)
                teacher = torch.randn(9, 2, 8)
                padded = F.pad(student.detach(), (0, 0, 0, 0, 0, 1))
                local = (padded.chunk(2)[rank] if sequence_parallel else student.detach()).clone().requires_grad_()
                teacher_padded = F.pad(teacher, (0, 0, 0, 0, 0, 1))
                teacher_local = teacher_padded.chunk(2)[rank] if sequence_parallel else teacher
                model = SimpleNamespace(
                    config=SimpleNamespace(tensor_model_parallel_size=2),
                    language_model=_ParallelLanguageModel(weight.chunk(2)[rank], bias.chunk(2)[rank], tied),
                )
                holder = {}
                teacher_holder = {}
                # Gloo exercises the real MCore collectives; its allocation helper otherwise assumes CUDA.
                with patch("torch.cuda.current_device", return_value=torch.device("cpu")):
                    actual_hidden = SpeechCascadesModel._capture_hidden_states_only(
                        holder, sequence_parallel=sequence_parallel, tp_group=dist.group.WORLD, sequence_length=7
                    )(hidden_states=local)
                    with torch.no_grad():
                        actual_teacher = SpeechCascadesModel._capture_hidden_states_only(
                            teacher_holder,
                            sequence_parallel=sequence_parallel,
                            tp_group=dist.group.WORLD,
                            sequence_length=9,
                        )(hidden_states=teacher_local)
                    torch.testing.assert_close(actual_hidden, student)
                    torch.testing.assert_close(actual_teacher, teacher)
                    student_selected = actual_hidden[2:6].reshape(-1, 8)
                    teacher_selected = actual_teacher[4:8].reshape(-1, 8)
                    actual = chunked_kl_divergence(student_selected, teacher_selected, _make_logits_fn(model), 2.0, 3)
                    expected = chunked_kl_divergence(
                        student[2:6].reshape(-1, 8),
                        teacher[4:8].reshape(-1, 8),
                        lambda hidden: F.linear(hidden, weight, bias) * 0.75,
                        2.0,
                        3,
                    )
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                    actual.backward()
                    expected.backward()
                expected_gradient = (
                    F.pad(student.grad, (0, 0, 0, 0, 0, 1)).chunk(2)[rank] if sequence_parallel else student.grad
                )
                torch.testing.assert_close(local.grad, expected_gradient, rtol=2e-5, atol=2e-6)
                assert local.grad.abs().sum() > 0
                assert all(parameter.grad is None for parameter in model.language_model.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_two_rank_kd_matches_full_vocabulary_loss_and_gradients(tmp_path):
    mp.spawn(_parallel_kl_worker, args=(f"file://{tmp_path}/init",), nprocs=2, join=True)


def _speech_batch_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:

        def source_batches():
            for length in (5, 9, 4):
                batch = {
                    "input_ids": torch.arange(length).unsqueeze(0),
                    "labels": torch.tensor([[-100, -100, *range(2, length)]]),
                    "audio_frames": torch.tensor([length * 8]),
                }
                if length != 4:
                    batch.update(
                        {
                            "inputs_embeds": torch.ones(1, length, 3),
                            "audio_attention_mask": torch.ones(1, length, dtype=torch.long),
                            "audio_embed_sizes": torch.tensor([length]),
                            "teacher_input_ids": torch.arange(length + 1).unsqueeze(0),
                            "teacher_labels": torch.tensor([[-100, -100, -100, *range(3, length + 1)]]),
                            "teacher_attention_mask": torch.ones(1, length + 1, dtype=torch.bool),
                        }
                    )
                yield batch

        with patch("megatron.bridge.models.speech_cascades.step._to_cuda", side_effect=lambda value: value):
            if rank == 0:
                iterator = source_batches()
            else:
                provider = SpeechChunkStreamingDatasetProvider(seq_length=9, speech_config_yaml="/missing-data.yaml")
                context = DatasetBuildContext(
                    train_samples=1,
                    valid_samples=0,
                    test_samples=0,
                    pg_collection=SimpleNamespace(tp=dist.group.WORLD),
                )
                iterator, _, _ = provider.build_datasets(context)
                assert list(iterator) == []
            for length in (5, 9, 4):
                batch = get_alm_batch(iterator, tp_group=dist.group.WORLD)
                assert batch["input_ids"].shape == (1, length)
                assert batch["labels"].tolist() == [[0, *range(2, length), 0]]
                assert batch["loss_mask"].tolist() == [[0.0, *([1.0] * (length - 2)), 0.0]]
                if length == 4:
                    assert batch["audio_input_features"] is None
                    assert "teacher_input_ids" not in batch
                else:
                    assert batch["audio_input_features"].shape == (1, length, 3)
                    assert batch["audio_input_features"].dtype == torch.bfloat16
                    assert batch["audio_embed_sizes"].tolist() == [length]
                    assert batch["teacher_input_ids"].shape == (1, length + 1)
                    assert batch["teacher_attention_mask"].dtype == torch.bool

            class Timer:
                def start(self):
                    pass

                def stop(self):
                    pass

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.language_model = SimpleNamespace(pg_collection=SimpleNamespace(tp=dist.group.WORLD))

                def forward(self, *, input_ids, labels, **kwargs):
                    return labels.float()

            state = SimpleNamespace(
                timers=lambda *args, **kwargs: Timer(),
                cfg=SimpleNamespace(
                    model=SimpleNamespace(tensor_model_parallel_size=2),
                    rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=False, check_for_spiky_loss=False),
                ),
            )
            asr_batch = source_batches()
            next(asr_batch)
            next(asr_batch)
            forward_iterator = asr_batch if rank == 0 else iter(())
            output, _ = forward_step(state, forward_iterator, Model())
            assert output.shape == (1, 4)
            assert output.tolist() == [[0.0, 2.0, 3.0, 0.0]]
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_two_rank_dynamic_speech_batch_is_identical_without_non_source_iterator(tmp_path):
    mp.spawn(_speech_batch_worker, args=(f"file://{tmp_path}/init-batch",), nprocs=2, join=True)


@pytest.mark.unit
def test_single_rank_untied_kd_uses_output_head_not_input_embedding():
    head = nn.Linear(8, 14, bias=False).requires_grad_(False)
    inputs = torch.randn(5, 8)

    def output_layer(hidden, *, weight):
        assert weight is None
        return head(hidden), None

    model = SimpleNamespace(
        config=SimpleNamespace(tensor_model_parallel_size=1),
        language_model=SimpleNamespace(
            share_embeddings_and_output_weights=False,
            output_layer=output_layer,
            _scale_logits=lambda logits: logits * 0.75,
        ),
    )
    torch.testing.assert_close(_make_logits_fn(model)(inputs), head(inputs) * 0.75)
