from transformers import Qwen3_5TextConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider


def test_qwen35_4b_speech_baseline_provider_settings() -> None:
    """Keep the native language provider aligned with the current speech model."""
    layer_types = ["full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(32)]
    text_config = Qwen3_5TextConfig(
        architectures=["Qwen3_5ForCausalLM"],
        attention_bias=False,
        linear_num_value_heads=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        layer_types=layer_types,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
    )

    provider = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)

    assert isinstance(provider, GPTModelProvider)
    assert provider.normalization == "RMSNorm"
    assert provider.gated_linear_unit is True
    assert provider.add_qkv_bias is False
    assert provider.add_bias_linear is False
    assert provider.qk_layernorm is True
    assert provider.hidden_dropout == 0.0
    assert provider.layernorm_zero_centered_gamma is True
    assert provider.attention_output_gate is True
    assert provider.experimental_attention_variant == "gated_delta_net"
    assert provider.linear_attention_freq == 4
    assert provider.linear_num_value_heads == 32
    assert provider.rotary_percent == 0.25
    assert provider.linear_conv_kernel_dim == 4
    assert provider.linear_key_head_dim == 128
    assert provider.linear_value_head_dim == 128
    assert provider.linear_num_key_heads == 16