"""Unit tests for the WP-A2 dual-forward knowledge-distillation loss in the training step.

CPU-only, tiny dims, with a stubbed language model standing in for the Megatron-Core GPT model. The
stub exposes exactly the surface the KD step uses: ``captured_hidden_states`` set by the student
forward, ``text_forward_hidden_states`` for the frozen teacher forward, and
``language_model.{shared_embedding_or_output_weight, output_layer}`` for the logits projection.
"""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from megatron.bridge.models.speech_cascades import step as step_mod  # noqa: E402
from megatron.bridge.models.speech_cascades.step import (  # noqa: E402
    _compute_kd_kl,
    _kd_loss_function,
    forward_step,
)
from megatron.bridge.training.losses import masked_next_token_loss  # noqa: E402


IGNORE_INDEX = -100
# get_alm_batch moves batch tensors to cuda when available; keep the fake model on the same device.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _FakeLanguageModel(nn.Module):
    """Frozen tied output head surface used by the KD logits projection."""

    def __init__(self, hidden: int, vocab: int) -> None:
        super().__init__()
        self.head = nn.Parameter(torch.randn(vocab, hidden) * 0.02)
        self.share_embeddings_and_output_weights = True

    def _scale_logits(self, logits):
        return logits

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        return self.head

    def output_layer(self, hidden_sbhd: torch.Tensor, weight: torch.Tensor | None = None):
        weight = self.head if weight is None else weight
        return hidden_sbhd @ weight.t(), None


class _FakeSpeechModel(nn.Module):
    """Minimal stand-in for :class:`SpeechCascadesModel` exercising only the KD step surface."""

    def __init__(
        self,
        hidden: int,
        vocab: int,
        *,
        distillation_weight: float = 0.75,
        temperature: float = 2.0,
        chunk_size: int = 8,
    ) -> None:
        super().__init__()
        self.language_model = _FakeLanguageModel(hidden, vocab)
        self.embed = nn.Embedding(vocab, hidden)  # frozen
        self.student_proj = nn.Linear(hidden, hidden, bias=False)  # trainable student
        self.teacher_body = nn.Linear(hidden, hidden, bias=False)  # frozen teacher LM body proxy
        self.config = SimpleNamespace(
            distillation_weight=distillation_weight,
            distillation_temperature=temperature,
            distillation_kl_chunk_size=chunk_size,
        )
        self.captured_hidden_states: torch.Tensor | None = None
        self.last_forward_kwargs: dict | None = None
        # Freeze everything except the student projector.
        for module in (self.language_model, self.embed, self.teacher_body):
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, input_ids, *, labels=None, return_hidden_states=False, **kwargs):
        self.last_forward_kwargs = {"return_hidden_states": return_hidden_states, **kwargs}
        hidden = self.student_proj(self.embed(input_ids.clamp_min(0)))  # [B, S, H]
        self.captured_hidden_states = hidden if (return_hidden_states and labels is not None) else None
        logits = hidden @ self.language_model.head.t()  # [B, S, V]
        b, s, v = logits.shape
        ce = F.cross_entropy(logits.reshape(b * s, v), labels.reshape(b * s), reduction="none")
        return ce.reshape(b, s)

    def text_forward_hidden_states(self, input_ids, attention_mask=None):
        return self.teacher_body(self.embed(input_ids.clamp_min(0)))  # [B, T, H]


def _kd_raw_batch(prompt=(4, 5, 6), response=(7, 8)):
    """One-example KD batch: prompt then a teacher-forced response, teacher = same text view."""
    input_ids = list(prompt) + list(response)
    labels = [IGNORE_INDEX] * len(prompt) + list(response)
    teacher_input_ids = list(input_ids)
    teacher_labels = list(labels)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "teacher_input_ids": torch.tensor([teacher_input_ids], dtype=torch.long),
        "teacher_labels": torch.tensor([teacher_labels], dtype=torch.long),
        "teacher_attention_mask": torch.ones(1, len(teacher_input_ids), dtype=torch.bool),
    }


def _prepare(model, raw_batch):
    """Run the student forward and return (batch, loss_mask, output_tensor, kl)."""
    from megatron.bridge.models.speech_cascades.step import get_alm_batch

    batch = get_alm_batch(iter([raw_batch]))
    loss_mask = batch.pop("loss_mask")
    forward_args = {"input_ids": batch["input_ids"], "labels": batch["labels"], "return_hidden_states": True}
    output_tensor = model(**forward_args)
    kl = _compute_kd_kl(model, batch, loss_mask)
    return batch, loss_mask, output_tensor, kl


def test_dual_forward_returns_finite_scalar_and_student_only_grad():
    torch.manual_seed(0)
    model = _FakeSpeechModel(hidden=16, vocab=48).to(DEVICE)
    batch, loss_mask, output_tensor, kl = _prepare(model, _kd_raw_batch())

    total_loss, num_tokens, _ = _kd_loss_function(loss_mask, kl, 0.75, False, False, output_tensor)

    assert total_loss.ndim == 0 and torch.isfinite(total_loss)
    assert int(num_tokens) == 2

    total_loss.backward()
    # Student trainable receives gradient; the frozen teacher/output-head/embeddings do not.
    assert model.student_proj.weight.grad is not None
    assert torch.any(model.student_proj.weight.grad != 0)
    assert model.teacher_body.weight.grad is None
    assert model.language_model.head.grad is None
    assert model.embed.weight.grad is None


def test_teacher_path_is_detached():
    torch.manual_seed(0)
    model = _FakeSpeechModel(hidden=16, vocab=48).to(DEVICE)
    raw = {k: v.to(DEVICE) for k, v in _kd_raw_batch().items()}

    model(input_ids=raw["input_ids"], labels=raw["labels"], return_hidden_states=True)
    with torch.no_grad():
        teacher_hidden = model.text_forward_hidden_states(raw["teacher_input_ids"])
    assert teacher_hidden.requires_grad is False


def test_mixing_reduces_to_ce_and_pure_kl_at_limits():
    torch.manual_seed(0)
    model = _FakeSpeechModel(hidden=16, vocab=48).to(DEVICE)
    _, loss_mask, output_tensor, kl = _prepare(model, _kd_raw_batch())

    ce_sum, num_tokens, _ = masked_next_token_loss(loss_mask, output_tensor, False, False)

    total_ce, _, _ = _kd_loss_function(loss_mask, kl, 0.0, False, False, output_tensor)
    total_kl, _, _ = _kd_loss_function(loss_mask, kl, 1.0, False, False, output_tensor)

    # w -> 0 is the existing masked CE; w -> 1 is the per-token-mean KL scaled by num_tokens.
    assert torch.isclose(total_ce, ce_sum, atol=1e-5)
    assert torch.isclose(total_kl, kl * num_tokens, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("weight", [0.0, 0.75, 1.0])
def test_kd_reports_actual_optimized_loss(weight):
    mask = torch.tensor([[1.0, 1.0, 0.0]], device=DEVICE)
    ce = torch.tensor([[4.0, 6.0, 99.0]], device=DEVICE, requires_grad=True)
    kl = torch.tensor(0.25, device=DEVICE, requires_grad=True)
    total, tokens, reporting = _kd_loss_function(mask, kl, weight, False, False, ce)
    assert tokens.item() == 2
    assert torch.isclose(reporting["lm loss"][0], total.detach())
    assert reporting["ce loss"].tolist() == [10.0, 2.0]
    assert reporting["kl loss"].tolist() == [0.5, 2.0]
    assert all(not value.requires_grad for value in reporting.values())


def test_misaligned_response_labels_raise():
    torch.manual_seed(0)
    model = _FakeSpeechModel(hidden=16, vocab=48).to(DEVICE)
    raw = _kd_raw_batch()
    # Corrupt the teacher response token (valid id, differs from the student) so alignment fails.
    raw["teacher_labels"][0, -1] = 9
    raw["teacher_input_ids"][0, -1] = 9

    batch = step_mod.get_alm_batch(iter([raw]))
    loss_mask = batch.pop("loss_mask")
    model(input_ids=batch["input_ids"], labels=batch["labels"], return_hidden_states=True)
    with pytest.raises(ValueError, match="not token-aligned"):
        _compute_kd_kl(model, batch, loss_mask)


def _fake_state():
    class _Timer:
        def start(self, *args, **kwargs):
            return self

        def stop(self):
            return None

    return SimpleNamespace(
        timers=lambda *args, **kwargs: _Timer(),
        cfg=SimpleNamespace(
            rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=False, check_for_spiky_loss=False)
        ),
    )


def test_forward_step_kd_selects_kd_loss_and_requests_hidden_states():
    torch.manual_seed(0)
    model = _FakeSpeechModel(hidden=16, vocab=48).to(DEVICE)
    output_tensor, loss_function = forward_step(_fake_state(), iter([_kd_raw_batch()]), model)

    assert model.last_forward_kwargs["return_hidden_states"] is True
    assert loss_function.func is _kd_loss_function
    total_loss, _, _ = loss_function(output_tensor)
    assert torch.isfinite(total_loss)


def test_forward_step_non_kd_is_unchanged():
    torch.manual_seed(0)
    model = _FakeSpeechModel(hidden=16, vocab=48).to(DEVICE)
    raw = {
        "input_ids": torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long),
        "labels": torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 7, 8]], dtype=torch.long),
    }
    _, loss_function = forward_step(_fake_state(), iter([raw]), model)

    # The ASR path never asks for hidden states and keeps the masked next-token CE loss function.
    assert model.last_forward_kwargs["return_hidden_states"] is False
    assert loss_function.func is masked_next_token_loss
