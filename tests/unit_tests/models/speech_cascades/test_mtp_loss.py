"""Unit tests for the WP-C2 MTP per-depth KL+CE loss (CPU, tiny dims, no model parallel).

Two layers are exercised:

* :func:`mtp_depth_loss` directly, asserting the ported reference formula on a tiny example -- the
  ``T^2`` factor on each depth KL, the depth label shift ``labels[..., k + 1:]``, the depth weighting
  by ``mtp_loss_depth_weights`` and the ``mtp_loss_weight`` scale.
* The training-step wiring (``_compute_mtp_loss`` + ``_kd_loss_function``) with a stubbed model, so the
  total main-KD + MTP loss is finite, gradients reach only the trainable student/MTP heads (the
  teacher target is detached), and the MTP path is a no-op when disabled.
"""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from megatron.bridge.models.speech_cascades.distillation import (  # noqa: E402
    chunked_kl_divergence,
    mtp_depth_loss,
)
from megatron.bridge.models.speech_cascades.step import (  # noqa: E402
    _compute_kd_kl,
    _compute_mtp_loss,
    _kd_loss_function,
    get_alm_batch,
)
from megatron.bridge.training.losses import masked_next_token_loss  # noqa: E402


IGNORE_INDEX = -100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPERATURE = 2.0
CHUNK = 8
MTP_W = 0.75
MTP_LOSS_WEIGHT = 1.0
DEPTH_WEIGHTS = [0.50, 0.25, 0.15, 0.10]


def _fixture(hidden=3, vocab=5, seq=6, depths=2, seed=0):
    """Deterministic per-depth hidden states, main hidden, labels, mask and a linear output head."""
    torch.manual_seed(seed)
    head = nn.Linear(hidden, vocab, bias=False)

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        return head(x)

    mtp_hidden_states = [torch.randn(seq - k, 1, hidden) for k in range(1, depths + 1)]
    main_hidden = torch.randn(1, seq, hidden)
    labels = torch.randint(0, vocab, (1, seq))
    loss_mask = torch.ones(1, seq)
    loss_mask[:, -1] = 0.0  # last shifted position is always ignored
    return mtp_hidden_states, main_hidden, labels, loss_mask, logits_fn


def _reference_mtp(mtp_hidden_states, main_hidden, labels, loss_mask, logits_fn, mtp_w, depth_weights):
    """Independent re-derivation of the reference per-depth MTP loss (self-distillation teacher)."""
    teacher_full = main_hidden.detach()
    depth_losses = []
    for depth_idx, depth_hidden in enumerate(mtp_hidden_states):
        depth = depth_idx + 1
        # Reference depth shift: labels[..., depth + 1:] == shifted_labels[..., depth:-1].
        label_ids = labels[:, depth:-1]
        if label_ids.shape[1] == 0:
            break
        mask = loss_mask[:, depth:-1].bool()
        student_seq = depth_hidden.transpose(0, 1)[:, :-1, :]
        teacher_seq = teacher_full[:, depth:-1, :]
        student_sel = student_seq[mask][1:]
        labels_sel = label_ids[mask].reshape(-1)[1:]
        teacher_sel = teacher_seq[mask][1:]
        if student_sel.numel() == 0:
            continue
        ce = F.cross_entropy(logits_fn(student_sel).float(), labels_sel)
        student_log_probs = F.log_softmax(logits_fn(student_sel).float() / TEMPERATURE, dim=-1)
        teacher_probs = F.softmax(logits_fn(teacher_sel).float() / TEMPERATURE, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="sum") / student_sel.shape[0]
        kl = kl * TEMPERATURE**2  # the T^2 factor unique to the MTP KL
        depth_losses.append((1.0 - mtp_w) * ce + mtp_w * kl)
    stacked = torch.stack(depth_losses)
    weights = torch.tensor(depth_weights[: len(depth_losses)])
    weights = weights / weights.sum()
    return MTP_LOSS_WEIGHT * (stacked * weights).sum()


def test_mtp_depth_loss_matches_reference_formula():
    mtp_hidden_states, main_hidden, labels, loss_mask, logits_fn = _fixture()

    agg, comps = mtp_depth_loss(
        mtp_hidden_states,
        main_hidden,
        labels,
        loss_mask,
        logits_fn,
        temperature=TEMPERATURE,
        chunk_size=CHUNK,
        mtp_distillation_weight=MTP_W,
        mtp_loss_weight=MTP_LOSS_WEIGHT,
        mtp_loss_depth_weights=DEPTH_WEIGHTS,
    )

    expected = _reference_mtp(mtp_hidden_states, main_hidden, labels, loss_mask, logits_fn, MTP_W, DEPTH_WEIGHTS)
    assert torch.isclose(agg, expected, atol=1e-6)

    # (c)+(d): the aggregate is the mtp_loss_weight-scaled normalized depth-weighted sum of the
    # per-depth losses the helper itself reported.
    stacked = torch.stack(comps["depth_losses"])
    weights = torch.tensor(DEPTH_WEIGHTS[: len(stacked)])
    weights = weights / weights.sum()
    assert torch.isclose(agg, MTP_LOSS_WEIGHT * (stacked * weights).sum(), atol=1e-6)
    assert len(comps["depth_losses"]) == len(mtp_hidden_states)


def test_mtp_depth_kl_has_temperature_squared_factor():
    # Pure-KL, single depth, unit weight: the aggregate must equal chunked_kl * T^2 exactly.
    mtp_hidden_states, main_hidden, labels, loss_mask, logits_fn = _fixture(depths=1)
    agg, comps = mtp_depth_loss(
        mtp_hidden_states,
        main_hidden,
        labels,
        loss_mask,
        logits_fn,
        temperature=TEMPERATURE,
        chunk_size=CHUNK,
        mtp_distillation_weight=1.0,
        mtp_loss_weight=1.0,
        mtp_loss_depth_weights=None,
    )

    # Rebuild the depth-1 selection to feed chunked_kl_divergence directly.
    mask = loss_mask[:, 1:-1].bool()
    student_sel = mtp_hidden_states[0].transpose(0, 1)[:, :-1, :][mask][1:].contiguous()
    teacher_sel = main_hidden.detach()[:, 1:-1, :][mask][1:].contiguous()
    raw_kl = chunked_kl_divergence(student_sel, teacher_sel, logits_fn, TEMPERATURE, CHUNK)

    assert torch.isclose(agg, raw_kl * TEMPERATURE**2, atol=1e-6)
    assert torch.isclose(comps["depth_kl"][0], raw_kl * TEMPERATURE**2, atol=1e-6)


def test_mtp_depth_loss_grad_reaches_student_not_teacher():
    hidden, vocab, seq = 4, 6, 6
    torch.manual_seed(1)
    head = nn.Linear(hidden, vocab, bias=False)
    for p in head.parameters():
        p.requires_grad = False

    def logits_fn(x):
        return head(x)

    student_head = nn.Linear(hidden, hidden, bias=False)  # trainable MTP/student proxy
    teacher_source = nn.Linear(hidden, hidden, bias=False)  # feeds only the (detached) teacher target
    base = torch.randn(seq, 1, hidden)

    mtp_hidden_states = [student_head(base[:-k]) for k in range(1, 3)]
    main_hidden = teacher_source(torch.randn(1, seq, hidden))
    labels = torch.randint(0, vocab, (1, seq))
    loss_mask = torch.ones(1, seq)
    loss_mask[:, -1] = 0.0

    agg, _ = mtp_depth_loss(
        mtp_hidden_states,
        main_hidden,
        labels,
        loss_mask,
        logits_fn,
        temperature=TEMPERATURE,
        chunk_size=CHUNK,
        mtp_distillation_weight=MTP_W,
        mtp_loss_weight=MTP_LOSS_WEIGHT,
        mtp_loss_depth_weights=DEPTH_WEIGHTS,
    )
    assert torch.isfinite(agg)
    agg.backward()

    assert student_head.weight.grad is not None and torch.any(student_head.weight.grad != 0)
    assert teacher_source.weight.grad is None  # teacher target detached inside the helper
    assert head.weight.grad is None


# --------------------------------------------------------------------------------------------- #
# Training-step wiring (_compute_mtp_loss + _kd_loss_function)                                   #
# --------------------------------------------------------------------------------------------- #


class _FakeLanguageModel(nn.Module):
    def __init__(self, hidden: int, vocab: int) -> None:
        super().__init__()
        self.head = nn.Parameter(torch.randn(vocab, hidden) * 0.02)

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        return self.head

    def output_layer(self, hidden_sbhd: torch.Tensor, weight: torch.Tensor | None = None):
        weight = self.head if weight is None else weight
        return hidden_sbhd @ weight.t(), None


class _FakeKdMtpModel(nn.Module):
    """Stub exposing the KD + MTP surface the training step reads."""

    def __init__(self, hidden: int, vocab: int, *, mtp_depths: int) -> None:
        super().__init__()
        self.language_model = _FakeLanguageModel(hidden, vocab)
        self.embed = nn.Embedding(vocab, hidden)  # frozen
        self.student_proj = nn.Linear(hidden, hidden, bias=False)  # trainable student
        self.mtp_depth = nn.Linear(hidden, hidden, bias=False)  # trainable MTP head proxy
        self.teacher_body = nn.Linear(hidden, hidden, bias=False)  # frozen teacher LM body proxy
        self.config = SimpleNamespace(
            distillation_weight=0.75,
            distillation_temperature=TEMPERATURE,
            distillation_kl_chunk_size=CHUNK,
            mtp_num_hidden_layers=mtp_depths,
            mtp_loss_weight=MTP_LOSS_WEIGHT,
            mtp_distillation_weight=MTP_W,
            mtp_loss_depth_weights=DEPTH_WEIGHTS,
        )
        self.captured_hidden_states: torch.Tensor | None = None
        self.mtp_hidden_states: list[torch.Tensor] | None = None
        for module in (self.language_model, self.embed, self.teacher_body):
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, input_ids, *, labels=None, return_hidden_states=False, **kwargs):
        hidden = self.student_proj(self.embed(input_ids.clamp_min(0)))  # [B, S, H]
        capture = return_hidden_states and labels is not None
        self.captured_hidden_states = hidden if capture else None
        if capture and self.config.mtp_num_hidden_layers > 0:
            prev = hidden.transpose(0, 1)  # [S, B, H]
            outs: list[torch.Tensor] = []
            for _ in range(self.config.mtp_num_hidden_layers):
                prev = self.mtp_depth(prev[:-1])  # [S - k, B, H]
                outs.append(prev)
            self.mtp_hidden_states = outs
        else:
            self.mtp_hidden_states = None
        logits = hidden @ self.language_model.head.t()
        b, s, v = logits.shape
        ce = F.cross_entropy(logits.reshape(b * s, v), labels.reshape(b * s), reduction="none")
        return ce.reshape(b, s)

    def text_forward_hidden_states(self, input_ids, attention_mask=None):
        return self.teacher_body(self.embed(input_ids.clamp_min(0)))  # [B, T, H]


def _kd_raw_batch(prompt=(4, 5, 6), response=(1, 2, 3)):
    input_ids = list(prompt) + list(response)
    labels = [IGNORE_INDEX] * len(prompt) + list(response)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "teacher_input_ids": torch.tensor([input_ids], dtype=torch.long),
        "teacher_labels": torch.tensor([labels], dtype=torch.long),
        "teacher_attention_mask": torch.ones(1, len(input_ids), dtype=torch.bool),
    }


def _prepare(model, raw_batch):
    batch = get_alm_batch(iter([raw_batch]))
    loss_mask = batch.pop("loss_mask")
    output_tensor = model(input_ids=batch["input_ids"], labels=batch["labels"], return_hidden_states=True)
    kl = _compute_kd_kl(model, batch, loss_mask)
    mtp = _compute_mtp_loss(model, batch, loss_mask)
    return batch, loss_mask, output_tensor, kl, mtp


def test_total_loss_finite_and_grad_to_student_and_mtp_only():
    torch.manual_seed(0)
    model = _FakeKdMtpModel(hidden=8, vocab=16, mtp_depths=2).to(DEVICE)
    _, loss_mask, output_tensor, kl, mtp = _prepare(model, _kd_raw_batch())

    assert mtp is not None and torch.isfinite(mtp)
    total_loss, num_tokens, _ = _kd_loss_function(loss_mask, kl, 0.75, False, False, output_tensor, mtp_loss=mtp)
    assert total_loss.ndim == 0 and torch.isfinite(total_loss)

    total_loss.backward()
    assert model.student_proj.weight.grad is not None and torch.any(model.student_proj.weight.grad != 0)
    assert model.mtp_depth.weight.grad is not None and torch.any(model.mtp_depth.weight.grad != 0)
    assert model.teacher_body.weight.grad is None
    assert model.language_model.head.grad is None
    assert model.embed.weight.grad is None


def test_total_loss_adds_mtp_scaled_by_num_tokens():
    torch.manual_seed(0)
    model = _FakeKdMtpModel(hidden=8, vocab=16, mtp_depths=2).to(DEVICE)
    _, loss_mask, output_tensor, kl, mtp = _prepare(model, _kd_raw_batch())

    _, num_tokens, _ = masked_next_token_loss(loss_mask, output_tensor, False, False)
    with_mtp, _, _ = _kd_loss_function(loss_mask, kl, 0.75, False, False, output_tensor, mtp_loss=mtp)
    without_mtp, _, _ = _kd_loss_function(loss_mask, kl, 0.75, False, False, output_tensor, mtp_loss=None)

    assert torch.isclose(with_mtp - without_mtp, mtp * num_tokens.clamp(min=1), atol=1e-5)


def test_mtp_disabled_is_noop():
    torch.manual_seed(0)
    model = _FakeKdMtpModel(hidden=8, vocab=16, mtp_depths=0).to(DEVICE)
    _, loss_mask, output_tensor, kl, mtp = _prepare(model, _kd_raw_batch())

    # No MTP heads: _compute_mtp_loss is a no-op and the KD loss is byte-identical to WP-A2.
    assert mtp is None
    assert model.mtp_hidden_states is None
    with_none, _, _ = _kd_loss_function(loss_mask, kl, 0.75, False, False, output_tensor, mtp_loss=None)
    baseline, _, _ = _kd_loss_function(loss_mask, kl, 0.75, False, False, output_tensor)
    assert torch.equal(with_none, baseline)


def test_mtp_noop_when_hidden_states_missing():
    torch.manual_seed(0)
    model = _FakeKdMtpModel(hidden=8, vocab=16, mtp_depths=2).to(DEVICE)
    batch = get_alm_batch(iter([_kd_raw_batch()]))
    loss_mask = batch.pop("loss_mask")
    # MTP configured on, but the model never captured per-depth hidden states this step.
    model.mtp_hidden_states = None
    model.captured_hidden_states = torch.randn(1, 6, 8)
    assert _compute_mtp_loss(model, batch, loss_mask) is None
