from dataclasses import dataclass

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.speech_cascades.model import SpeechCascadesModel
from megatron.bridge.models.speech_cascades.provider import SpeechCascadesModelProvider

from .bridge import Qwen4ExpTextBridge
from .qwen4_exp_provider import Qwen4ExpModelProvider


class Qwen4ExpSpeechModel(SpeechCascadesModel):
    """Cascades speech encoder with the native Qwen4Exp text backbone."""

    def _build_language_model(self, config, pre_process, post_process, vp_stage):
        return Qwen4ExpModelProvider.provide(config, pre_process, post_process, vp_stage)

    def forward(self, input_ids, **kwargs):
        """Retain lexical token IDs for PLE alongside audio-conditioned embeddings."""
        return super().forward(input_ids, ple_input_ids=input_ids, **kwargs)


@dataclass
class Qwen4ExpSpeechProvider(SpeechCascadesModelProvider, Qwen4ExpModelProvider):
    """Configure the existing speech stack and Qwen4Exp-specific text blocks."""

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Build and optionally freeze the speech components."""
        model = Qwen4ExpSpeechModel(
            self,
            audio_token_id=self.audio_token_id,
            audio_encoder_config=self.audio_encoder_config,
            audio_projection_cls=self.audio_projection_cls,
            audio_compression_rate=self.audio_compression_rate,
            audio_downsample_rate=self.audio_downsample_rate,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            vp_stage=vp_stage,
        )
        model.freeze(
            freeze_language_model=self.freeze_language_model,
            freeze_audio_model=self.freeze_audio_model,
            freeze_audio_encoder=self.freeze_audio_encoder,
        )
        return model


@MegatronModelBridge.register_bridge(
    source="Qwen4ExpSpeechCascadesForConditionalGeneration", target=Qwen4ExpSpeechModel
)
class Qwen4ExpSpeechBridge(Qwen4ExpTextBridge):
    """Load a speech manifest referencing the original Qwen4Exp checkpoint shards."""

    PROVIDER_CLASS = Qwen4ExpSpeechProvider

    def provider_bridge(self, hf_pretrained):
        """Add the encoder configuration to the native text provider."""
        provider = super().provider_bridge(hf_pretrained)
        config = hf_pretrained.config
        provider.audio_encoder_config = {**config.audio_processor["config"], "activation_checkpointing": ""}
        provider.audio_projection_cls = config.embd_layer["projection_cls"]
        provider.audio_compression_rate = config.embd_layer["compression_rate"]
        provider.autocast_dtype = provider.params_dtype
        provider.mtp_num_layers = 0
        provider.mtp_num_hidden_layers = 0
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Combine native language mappings with the existing audio parameter contract."""
        registry = super().mapping_registry()
        mappings = list(registry.mappings)
        for mapping in mappings:
            mapping.megatron_param = "language_model." + mapping.megatron_param
        mappings.extend(
            [
                ReplicatedMapping("audio_model.encoder.**", "model.embed_tokens_extend.encoder.**"),
                ReplicatedMapping("audio_model.audio_projection.**", "model.embed_tokens_extend.audio_projection.**"),
            ]
        )
        return MegatronMappingRegistry(*mappings)
