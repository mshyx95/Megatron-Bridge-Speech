"""Unit tests for the speech-cascades MTP depth heads (WP-C1)."""

from dataclasses import fields

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
