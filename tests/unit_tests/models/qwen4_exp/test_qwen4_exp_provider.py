from types import SimpleNamespace

import pytest

from megatron.bridge.models.qwen4_exp.qwen4_exp_provider import Qwen4ExpModelProvider, apply_qwen4_exp_config


def _make_provider(**overrides) -> Qwen4ExpModelProvider:
    kwargs = {
        "num_layers": 48,
        "hidden_size": 2560,
        "ffn_hidden_size": 640,
        "num_attention_heads": 24,
        "num_query_groups": 2,
        "kv_channels": 256,
        "vocab_size": 248320,
        "seq_length": 262144,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "mtp_num_layers": 1,
    }
    kwargs.update(overrides)
    return Qwen4ExpModelProvider(**kwargs)


@pytest.mark.unit
def test_qwen4_exp_provider_matches_flash_next_geometry() -> None:
    provider = _make_provider()

    assert provider.hc_count == 4
    assert provider.hc_lowrank == 320
    assert provider.indexer_budget == 2048
    assert provider.indexer_compress_ratio == 4
    assert provider.ple_layer_ids == [2]
    assert provider.ple_embed_dim == 2560
    assert provider.mtp_num_layers == 1


@pytest.mark.unit
def test_qwen4_exp_provider_rejects_invalid_ple_layer() -> None:
    provider = _make_provider(ple_layer_ids=[49])

    with pytest.raises(ValueError, match="1-based"):
        provider.finalize()


@pytest.mark.unit
def test_apply_qwen4_exp_config_preserves_checkpoint_architecture() -> None:
    provider = _make_provider(mtp_num_layers=None)
    layer_types = ["linear_attention"] * 48
    for layer_index in range(3, 48, 4):
        layer_types[layer_index] = "full_attention"
    config = SimpleNamespace(
        attention_bias=False,
        attention_dropout=0.0,
        layer_types=layer_types,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        moe_intermediate_size=640,
        num_experts=512,
        num_experts_per_tok=10,
        shared_expert_intermediate_size=640,
        router_aux_loss_coef=0.001,
        hc_count=4,
        hc_lowrank=320,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        indexer_head_dim=128,
        indexer_kv_heads=1,
        indexer_n_heads=4,
        ple_layer_ids=[2],
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        mtp_num_hidden_layers=1,
    )

    apply_qwen4_exp_config(provider, config)

    assert provider.linear_attention_freq == [1, 1, 1, 0] * 12
    assert provider.ffn_hidden_size == 640
    assert provider.moe_ffn_hidden_size == 640
    assert provider.num_moe_experts == 512
    assert provider.moe_router_topk == 10
    assert provider.hc_count == 4
    assert provider.ple_layer_ids == [2]
    assert provider.mtp_num_layers == 1
