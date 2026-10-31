from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.models.qwen4_exp.modeling_qwen4_exp.hyper_connection import Qwen4ExpGatedResidual


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        hc_count=2,
        hc_lowrank=3,
        hidden_size=4,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
    )


@pytest.mark.unit
def test_gated_residual_matches_reference_equations() -> None:
    module = Qwen4ExpGatedResidual(_config())
    hidden_states = torch.randn(3, 2, 8)
    block_output = torch.randn(3, 2, 4)

    mixed, residual, injection = module(hidden_states)

    grouped = hidden_states.float().unflatten(-1, (2, 4))
    normalized = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + 1e-6)
    normalized = normalized.flatten(-2) * (1.0 + module.hc_norm.weight.float())
    low_rank_gate = F.silu(F.linear(normalized, module.input_mix_weight_down.weight) / 2)
    gate = torch.sigmoid(F.linear(low_rank_gate, module.input_mix_weight_up.weight)).unflatten(-1, (2, 4))
    expected_mixed = (gate * normalized.unflatten(-1, (2, 4))).mean(dim=-2)
    expected_injection = 2.0 * torch.sigmoid(F.linear(normalized, module.block_inject_weight.weight) / 2)
    expected_output = hidden_states.unflatten(-1, (2, 4)) + block_output.unsqueeze(-2) * expected_injection.unsqueeze(
        -1
    )

    torch.testing.assert_close(mixed, expected_mixed)
    torch.testing.assert_close(injection, expected_injection)
    torch.testing.assert_close(module.combine(residual, block_output, injection), expected_output.flatten(-2))


@pytest.mark.unit
def test_final_gated_residual_has_no_injection_weights() -> None:
    module = Qwen4ExpGatedResidual(_config(), use_combine=False)
    output = module(torch.randn(3, 2, 8))

    assert isinstance(output, torch.Tensor)
    assert module.block_inject_weight is None


@pytest.mark.unit
def test_layer_hooks_preserve_gr_residual_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from megatron.bridge.models.qwen4_exp.modeling_qwen4_exp.transformer_layer import Qwen4ExpTransformerLayer

    layer = Qwen4ExpTransformerLayer.__new__(Qwen4ExpTransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.attn_hyper_connection = Qwen4ExpGatedResidual(_config())
    layer.mlp_hyper_connection = Qwen4ExpGatedResidual(_config())
    layer.hidden_dropout = 0.0
    layer.attn_norm_manager = SimpleNamespace(group_offload=lambda output: output)
    layer.mlp_norm_manager = SimpleNamespace(group_offload=lambda output: output)
    monkeypatch.setattr(TransformerLayer, "_run_input_layernorm", lambda self, hidden: (hidden, hidden, ()))
    monkeypatch.setattr(TransformerLayer, "_pre_mlp_layernorm_and_residual", lambda self, hidden: (hidden, hidden, ()))
    hidden = torch.randn(3, 2, 8, requires_grad=True)

    branch, residual, state = layer._run_input_layernorm(hidden)
    after_attention = layer._apply_self_attn_bda_step((branch.square(), None), residual, state)
    branch, residual, state = layer._pre_mlp_layernorm_and_residual(after_attention)
    output = layer._apply_mlp_bda_step((branch.sin(), None), residual, state)

    expected_branch, expected_residual, injection = layer.attn_hyper_connection(hidden)
    expected = layer.attn_hyper_connection.combine(expected_residual, expected_branch.square(), injection)
    expected_branch, expected_residual, injection = layer.mlp_hyper_connection(expected)
    expected = layer.mlp_hyper_connection.combine(expected_residual, expected_branch.sin(), injection)
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(parameter.grad is not None for parameter in layer.parameters())
