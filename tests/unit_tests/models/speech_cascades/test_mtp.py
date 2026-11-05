"""Unit tests for the speech-cascades MTP depth heads (WP-C1)."""

from dataclasses import fields
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from megatron.bridge.models.speech_cascades.mtp import SpeechCascadesMTP  # noqa: E402


class _StubLayer(nn.Module):
    """Minimal stand-in for a Megatron ``TransformerLayer`` with the same call/return contract."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, *, attention_mask=None, rotary_pos_emb=None):
        return self.linear(hidden_states), None


def _make_mtp(hidden_size: int, num_depth_layers: int, *, init_from_first: bool = False):
    source_state_dict = None
    if init_from_first:
        torch.manual_seed(123)
        source_state_dict = _StubLayer(hidden_size).state_dict()

    def _builder():
        return _StubLayer(hidden_size)

    return SpeechCascadesMTP(
        hidden_size=hidden_size,
        num_depth_layers=num_depth_layers,
        layer_builder=_builder,
        source_state_dict=source_state_dict,
    ), source_state_dict


def test_provider_mtp_defaults_are_off():
    from megatron.bridge.models.speech_cascades.provider import SpeechCascadesModelProvider

    defaults = {f.name: f.default for f in fields(SpeechCascadesModelProvider)}
    assert defaults["mtp_num_hidden_layers"] == 0
    assert defaults["mtp_loss_weight"] == 0.0
    assert defaults["mtp_loss_depth_weights"] is None
    assert defaults["mtp_distillation_weight"] == 0.0
    assert defaults["mtp_init_extra_layers_from_first"] is False


def test_requires_at_least_one_depth():
    with pytest.raises(ValueError):
        _make_mtp(hidden_size=8, num_depth_layers=0)


def test_param_count_is_base_plus_depth_blocks():
    hidden_size, depth = 16, 4
    mtp, _ = _make_mtp(hidden_size, depth)

    assert len(mtp.layers) == depth

    # Shared projection + three RMSNorms, plus one stub decoder layer per depth.
    per_layer = sum(p.numel() for p in _StubLayer(hidden_size).parameters())
    fc_params = hidden_size * 2 * hidden_size
    norm_params = 3 * hidden_size
    expected = fc_params + norm_params + depth * per_layer
    assert sum(p.numel() for p in mtp.parameters()) == expected


def test_init_from_first_copies_source_into_every_depth_block():
    hidden_size, depth = 16, 4
    mtp, source_state_dict = _make_mtp(hidden_size, depth, init_from_first=True)

    for layer in mtp.layers:
        layer_state = layer.state_dict()
        assert layer_state.keys() == source_state_dict.keys()
        for key, source_tensor in source_state_dict.items():
            torch.testing.assert_close(layer_state[key], source_tensor)


def test_forward_returns_per_depth_hidden_states_with_expected_shapes():
    hidden_size, depth = 16, 4
    seq_len, batch = 6, 2
    mtp, _ = _make_mtp(hidden_size, depth)

    full_hidden = torch.randn(seq_len, batch, hidden_size)
    inputs_embeds = torch.randn(seq_len, batch, hidden_size)

    outputs = mtp(full_hidden, inputs_embeds)

    assert len(outputs) == depth
    for depth_idx, hidden in enumerate(outputs):
        k = depth_idx + 1
        assert hidden.shape == (seq_len - k, batch, hidden_size)


def test_forward_stops_when_sequence_exhausted():
    # With seq_len == depth the deepest head has zero-length input and is skipped.
    hidden_size, depth = 8, 4
    seq_len, batch = 4, 1
    mtp, _ = _make_mtp(hidden_size, depth)

    outputs = mtp(
        torch.randn(seq_len, batch, hidden_size),
        torch.randn(seq_len, batch, hidden_size),
    )

    # depths 1,2,3 -> lengths 3,2,1; depth 4 -> length 0 (stopped).
    assert [h.shape[0] for h in outputs] == [3, 2, 1]


def test_rotary_fn_is_invoked_with_depth_sequence_length():
    hidden_size, depth = 8, 3
    seq_len, batch = 5, 1
    mtp, _ = _make_mtp(hidden_size, depth)

    seen: list[int] = []

    def _rotary(length: int):
        seen.append(length)
        return None

    mtp(
        torch.randn(seq_len, batch, hidden_size),
        torch.randn(seq_len, batch, hidden_size),
        rotary_pos_emb_fn=_rotary,
    )
    # depth k consumes a length S-k sequence.
    assert seen == [seq_len - 1, seq_len - 2, seq_len - 3]


def test_capture_hidden_states_reproduces_standard_output_and_stashes_hidden():
    from megatron.bridge.models.speech_cascades.model import SpeechCascadesModel

    holder: dict[str, torch.Tensor] = {}
    process = SpeechCascadesModel._capture_hidden_states(holder)

    hidden = torch.randn(3, 2, 8)
    logits = torch.randn(3, 2, 5)
    sentinel_loss = torch.tensor(1.234)

    def _output_layer(h, *, weight=None, runtime_gather_output=None):
        return logits, None

    # labels present -> returns the language-model loss and captures hidden states.
    out = process(
        hidden_states=hidden,
        output_layer=_output_layer,
        output_weight=None,
        labels=torch.zeros(2, 3, dtype=torch.long),
        runtime_gather_output=None,
        compute_language_model_loss=lambda labels, lg: sentinel_loss,
        scale_logits=lambda lg: lg,
    )
    assert out is sentinel_loss
    assert holder["hidden_states"] is hidden

    # labels absent -> returns [b, s, h] logits (matches GPTModel's inference return).
    out_infer = process(
        hidden_states=hidden,
        output_layer=_output_layer,
        output_weight=None,
        labels=None,
        runtime_gather_output=None,
        compute_language_model_loss=lambda labels, lg: sentinel_loss,
        scale_logits=lambda lg: lg,
    )
    assert out_infer.shape == (2, 3, 5)


@pytest.mark.unit
@pytest.mark.parametrize("ce_requires_grad", [False, True])
@pytest.mark.parametrize("with_labels", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_monitoring_only_ce_preserves_values_hidden_gradients_and_inference(
    ce_requires_grad, with_labels, grad_enabled
):
    from megatron.bridge.models.speech_cascades.model import SpeechCascadesModel

    torch.manual_seed(29)
    hidden = torch.randn(5, 2, 8, requires_grad=True)
    weight = torch.randn(11, 8, requires_grad=True)
    labels = torch.randint(0, 11, (2, 5)) if with_labels else None
    expected_grad = grad_enabled and (ce_requires_grad or labels is None)
    observed_grad = []

    def output_layer(states, *, weight=None, runtime_gather_output=None):
        observed_grad.append(torch.is_grad_enabled())
        return torch.nn.functional.linear(states, projection), None

    def compute_loss(targets, logits):
        return torch.nn.functional.cross_entropy(
            logits.transpose(0, 1).reshape(-1, 11), targets.reshape(-1), reduction="none"
        ).reshape_as(targets)

    projection = weight
    holder = {}
    with torch.set_grad_enabled(grad_enabled):
        output = SpeechCascadesModel._capture_hidden_states(
            holder, sequence_length=3, ce_requires_grad=ce_requires_grad
        )(
            hidden_states=hidden,
            output_layer=output_layer,
            output_weight=None,
            labels=labels,
            runtime_gather_output=None,
            compute_language_model_loss=compute_loss,
            scale_logits=lambda logits: logits * 0.75,
        )
    with torch.no_grad():
        logits = torch.nn.functional.linear(hidden, weight) * 0.75
        expected = logits.transpose(0, 1).contiguous() if labels is None else compute_loss(labels, logits)
    assert torch.equal(output, expected[:, :3])
    assert output.is_contiguous()
    assert output.requires_grad is expected_grad
    head_calls = 2 if not ce_requires_grad and labels is not None else 1
    assert observed_grad == [expected_grad] * head_calls
    assert torch.equal(holder["hidden_states"], hidden[:3])
    if grad_enabled:
        # This stands in for KL/MTP, which must retain their route into the student.
        holder["hidden_states"].square().sum().backward()
        torch.testing.assert_close(hidden.grad[:3], 2 * hidden.detach()[:3], rtol=0, atol=0)
        assert torch.count_nonzero(hidden.grad[3:]) == 0
        assert weight.grad is None


@pytest.mark.unit
@pytest.mark.parametrize("distillation_weight", [0.0, 0.75, 1.0, 1.25])
@pytest.mark.parametrize("return_hidden_states", [False, True])
@pytest.mark.parametrize("mtp_enabled", [False, True])
def test_student_forward_only_disables_ce_graph_for_pure_kd(distillation_weight, return_hidden_states, mtp_enabled):
    from megatron.bridge.models.speech_cascades.model import SpeechCascadesModel

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Parameter(torch.randn(5, 2, 8))
            self.head = nn.Linear(8, 11)

        def forward(self, *, labels, output_processor=None, **kwargs):
            def project(hidden_states, **kwargs):
                return self.head(hidden_states), None

            def loss(targets, logits):
                return torch.nn.functional.cross_entropy(
                    logits.transpose(0, 1).reshape(-1, 11), targets.reshape(-1), reduction="none"
                ).reshape_as(targets)

            if output_processor is None:
                return loss(labels, self.head(self.hidden))
            return output_processor(
                hidden_states=self.hidden,
                output_layer=project,
                output_weight=None,
                labels=labels,
                runtime_gather_output=None,
                compute_language_model_loss=loss,
                scale_logits=lambda logits: logits,
            )

    class MTP(nn.Module):
        def forward(self, *args, **kwargs):
            return []

    model = SpeechCascadesModel.__new__(SpeechCascadesModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=False, distillation_weight=distillation_weight)
    model.pre_process = False
    model.language_model = LanguageModel()
    model.mtp = MTP() if mtp_enabled else None
    output = model(
        torch.ones(2, 5, dtype=torch.long),
        labels=torch.zeros(2, 5, dtype=torch.long),
        return_hidden_states=return_hidden_states,
    )
    pure_kd = return_hidden_states and distillation_weight >= 1.0
    assert output.requires_grad is not pure_kd
    if return_hidden_states or mtp_enabled:
        assert model.captured_hidden_states.requires_grad
        model.captured_hidden_states.square().sum().backward()
        assert model.language_model.hidden.grad is not None
    else:
        assert model.captured_hidden_states is None
        output.sum().backward()
        assert model.language_model.hidden.grad is not None


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("tied_weight", [False, True])
@pytest.mark.parametrize(
    ("sequence_length", "chunk_size"), [(7, 3), (7, 1024), (1024, 1024), (1025, 1024), (2051, 1024)]
)
def test_monitoring_ce_bounds_logits_to_one_physical_row(batch_size, tied_weight, sequence_length, chunk_size):
    import weakref
    from unittest.mock import patch

    from megatron.bridge.models.speech_cascades.model import SpeechCascadesModel
    from megatron.bridge.training.losses import masked_next_token_loss

    torch.manual_seed(47)
    hidden = torch.randn(sequence_length, batch_size, 8, requires_grad=True)
    weight = torch.randn(13, 8, requires_grad=True)
    bias = torch.randn(13, requires_grad=True)
    labels = torch.randint(0, 13, (batch_size, sequence_length))
    trimmed_length = sequence_length - 2
    expected_weight = weight if tied_weight else None
    projection_shapes, live_logits = [], []

    def project(states, *, weight=None, runtime_gather_output=None):
        assert not torch.is_grad_enabled()
        assert weight is expected_weight
        assert runtime_gather_output is False
        assert all(reference() is None for reference in live_logits)
        projection_shapes.append(tuple(states.shape))
        logits = torch.nn.functional.linear(states, projection_weight, bias)
        live_logits.append(weakref.ref(logits))
        return logits, None

    def loss(targets, logits):
        return torch.nn.functional.cross_entropy(
            logits.transpose(0, 1).reshape(-1, 13), targets.reshape(-1), reduction="none"
        ).reshape_as(targets)

    def scale_logits(logits):
        scaled = logits * 0.5
        live_logits.append(weakref.ref(scaled))
        return scaled

    projection_weight = weight
    holder = {}
    with patch("megatron.bridge.models.speech_cascades.model._MONITORING_CE_LOCAL_CHUNK_SIZE", chunk_size):
        output = SpeechCascadesModel._capture_hidden_states(
            holder, ce_requires_grad=False, sequence_length=trimmed_length
        )(
            hidden_states=hidden,
            output_layer=project,
            output_weight=expected_weight,
            labels=labels,
            runtime_gather_output=False,
            compute_language_model_loss=loss,
            scale_logits=scale_logits,
        )
    with torch.no_grad():
        expected = loss(labels, torch.nn.functional.linear(hidden, weight, bias) * 0.5)[:, :trimmed_length]
    torch.testing.assert_close(output, expected)
    total, tokens, _ = masked_next_token_loss(torch.ones_like(output), output, False, False)
    torch.testing.assert_close(total, expected.sum())
    assert tokens.item() == expected.numel()
    lengths = (
        [sequence_length]
        if batch_size == 1 and sequence_length <= chunk_size
        else [min(chunk_size, sequence_length - start) for start in range(0, sequence_length, chunk_size)]
    )
    assert projection_shapes == [(length, 1, 8) for _ in range(batch_size) for length in lengths]
    assert not output.requires_grad
    assert holder["hidden_states"].requires_grad
    holder["hidden_states"].square().sum().backward()
    torch.testing.assert_close(hidden.grad[:trimmed_length], 2 * hidden.detach()[:trimmed_length], rtol=0, atol=0)
    assert torch.count_nonzero(hidden.grad[trimmed_length:]) == 0
    assert weight.grad is None and bias.grad is None
