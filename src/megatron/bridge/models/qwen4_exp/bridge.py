from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
from megatron.bridge.models.conversion.utils import moe_experts_stored_packed
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35MoEBridge

from .modeling_qwen4_exp.model import Qwen4ExpGPTModel
from .qwen4_exp_provider import Qwen4ExpModelProvider, apply_qwen4_exp_config


@MegatronModelBridge.register_bridge(
    source="Qwen4ExpForCausalLM", target=Qwen4ExpGPTModel, model_type="qwen4_exp_text"
)
class Qwen4ExpTextBridge(MegatronModelBridge):
    """Convert the Qwen4Exp text backbone without auxiliary MTP or vision."""

    PROVIDER_CLASS = Qwen4ExpModelProvider

    def provider_bridge(self, hf_pretrained):
        """Retain HF hash configuration and configure native text layers."""
        hf_config = hf_pretrained.config
        text = getattr(hf_config, "text_config", hf_config)
        kwargs = self.hf_config_to_provider_kwargs(text)
        provider = self.PROVIDER_CLASS(**kwargs)
        from types import SimpleNamespace

        values = text.to_dict()
        values.setdefault("mtp_num_hidden_layers", 0)
        apply_qwen4_exp_config(provider, SimpleNamespace(**values))
        provider.hf_text_config = text.to_dict()
        provider.share_embeddings_and_output_weights = hf_config.tie_word_embeddings
        provider.rotary_percent = text.rope_parameters["partial_rotary_factor"]
        provider.rotary_base = text.rope_parameters["rope_theta"]
        provider.position_embedding_type = "rope"
        provider.apply_rope_fusion = False
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Reuse GDN/MoE conversion and map model-specific residual parameters."""
        AutoMapping.register_module_type("Qwen4ExpGatedDeltaNet", "column")
        prefix = "model.language_model." if hasattr(self.hf_config, "text_config") else "model."
        mappings = Qwen35MoEBridge._get_moe_lm_mappings(
            hf_prefix=prefix,
            experts_packed=moe_experts_stored_packed(self.hf_pretrained, f"{prefix}layers."),
        )
        excluded = ("decoder.final_layernorm.weight", "pre_mlp_layernorm.weight", "layer_norm_weight")
        mappings = [mapping for mapping in mappings if not any(name in mapping.megatron_param for name in excluded)]
        for branch in ("attn_hyper_connection", "mlp_hyper_connection"):
            for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
                mappings.append(
                    ReplicatedMapping(
                        f"decoder.layers.*.{branch}.{name}.weight", f"{prefix}layers.*.{branch}.{name}.weight"
                    )
                )
        mappings.append(ReplicatedMapping("decoder.final_layernorm.**", f"{prefix}hyper_connection_mixer.**"))
        for name in ("index_qk_proj", "q_layernorm", "k_layernorm"):
            mappings.append(
                ReplicatedMapping(
                    f"decoder.layers.*.self_attention.indexer.{name}.weight",
                    f"{prefix}layers.*.self_attn.indexer.{name}.weight",
                )
            )
        for name in ("key_proj", "value_proj", "norm_key", "norm_query", "norm_conv", "conv1d"):
            mappings.append(
                ReplicatedMapping(f"decoder.layers.*.ple.{name}.weight", f"{prefix}layers.*.ple.{name}.weight")
            )
        for name in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
            mappings.append(
                ReplicatedMapping(
                    f"decoder.layers.*.ple.ple_embedding.{name}", f"{prefix}layers.*.ple.ple_embedding.{name}"
                )
            )
        mappings.append(
            AutoMapping(
                "decoder.layers.*.ple.ple_embedding.ngram_embedding.shard_*.weight",
                f"{prefix}layers.*.ple.ple_embedding.ngram_embedding.shard_*.weight",
            )
        )
        return MegatronMappingRegistry(*mappings)
