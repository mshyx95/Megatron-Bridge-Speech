import copy
from contextlib import contextmanager
from functools import partial

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from torch.utils.checkpoint import checkpoint

from .gdn import Qwen4ExpGatedDeltaNet
from .hyper_connection import Qwen4ExpGatedResidual
from .qsa import Qwen4ExpDotProductAttention, Qwen4ExpQSASelfAttention
from .transformer_layer import Qwen4ExpTransformerLayer


def qwen4_exp_layer_spec(config):
    """Build native GDN/QSA/MoE specs without extra branch input norms."""
    block = get_transformer_block_with_experimental_attention_variant_spec(config)
    block = copy.deepcopy(block)
    for index, spec in enumerate(block.layer_specs):
        spec.module = Qwen4ExpTransformerLayer
        spec.submodules.input_layernorm = IdentityOp
        spec.submodules.pre_mlp_layernorm = IdentityOp
        if config.linear_attention_freq[index]:
            spec.submodules.self_attention.module = Qwen4ExpGatedDeltaNet
            spec.submodules.self_attention.submodules.in_proj = TEColumnParallelLinear
        else:
            spec.submodules.self_attention.module = Qwen4ExpQSASelfAttention
            spec.submodules.self_attention.submodules.core_attention = Qwen4ExpDotProductAttention
            spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
    block.layer_norm = IdentityOp
    return block


class Qwen4ExpGPTModel(GPTModel):
    """Native text backbone with GR streams and vocabulary-parallel PLE."""

    def __init__(self, config, hf_text_config, **kwargs) -> None:
        from .ple import Qwen4ExpParallelPLE

        if config.pipeline_model_parallel_size != 1 or config.context_parallel_size != 1:
            raise NotImplementedError("Qwen4Exp currently requires PP=CP=1.")
        if config.mtp_num_layers:
            raise NotImplementedError(
                "Qwen4Exp native MTP is not implemented; set mtp_num_layers=0 for backbone smoke."
            )
        if config.fp8 or config.fp4 or config.cuda_graph_impl != "none":
            raise NotImplementedError("Qwen4Exp supports eager BF16/FP32 execution only.")
        super().__init__(
            config,
            transformer_layer_spec=qwen4_exp_layer_spec(config),
            scatter_embedding_sequence_parallel=False,
            **kwargs,
        )
        self.decoder.final_layernorm = Qwen4ExpGatedResidual(config, use_combine=False)
        self.eos_token_id = hf_text_config.eos_token_id
        if isinstance(self.eos_token_id, list):
            self.eos_token_id = self.eos_token_id[0]
        for layer in self.decoder.layers:
            if layer.layer_number in config.ple_layer_ids:
                layer.ple = Qwen4ExpParallelPLE(
                    hf_text_config,
                    config,
                    layer.layer_number - 1,
                    config.ple_layer_ids.index(layer.layer_number),
                    self.decoder.pg_collection.tp,
                )

    @staticmethod
    def _checkpoint_contexts(layer):
        router = layer.mlp.router
        statistics = {}

        @contextmanager
        def original_forward():
            for name in ("global_tokens_per_expert", "ga_steps"):
                value = getattr(router, name, None)
                if value is not None:
                    statistics[name] = value.clone()
            yield

        @contextmanager
        def recompute_forward():
            current = {name: getattr(router, name).clone() for name in statistics}
            tracker = get_moe_metrics_tracker()
            metrics = {name: entry.values.clone() for name, entry in tracker.metrics.items()}
            try:
                for name, value in statistics.items():
                    getattr(router, name).copy_(value)
                yield
            finally:
                for name, value in current.items():
                    getattr(router, name).copy_(value)
                for name, value in metrics.items():
                    tracker.metrics[name].values.copy_(value)

        return original_forward(), recompute_forward()

    @staticmethod
    def _run_layer(layer, hidden, token_ids, mask, rope, padding=None):
        if hasattr(layer, "ple"):
            hidden = hidden + layer.ple.forward_sequence(hidden, token_ids)
        return layer(hidden, attention_mask=mask, rotary_pos_emb=rope, padding_mask=padding)[0]

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        loss_mask=None,
        runtime_gather_output=None,
        ple_input_ids=None,
        output_processor=None,
        output_processor_context=None,
        **kwargs,
    ):
        unsupported = {name: value for name, value in kwargs.items() if value is not None}
        if unsupported:
            raise NotImplementedError(f"Unsupported Qwen4Exp forward options: {sorted(unsupported)}")
        token_ids = input_ids if ple_input_ids is None else ple_input_ids
        if self.config.ple_layer_ids and token_ids is None:
            raise ValueError("Qwen4Exp requires token IDs for PLE even when decoder_input is supplied.")
        if token_ids is not None:
            token_ids = torch.where(token_ids < 0, self.eos_token_id, token_ids)
        if attention_mask is not None and attention_mask.ndim != 4:
            raise NotImplementedError("Qwen4Exp expects a four-dimensional exclusion mask.")
        if decoder_input is None:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        original_length = decoder_input.shape[0]
        extra = (-original_length) % self.config.tensor_model_parallel_size if self.config.sequence_parallel else 0
        padding = None
        if extra:
            if output_processor is not None:
                raise NotImplementedError("Custom output processing with SP padding is not yet supported.")
            decoder_input = F.pad(decoder_input, (0, 0, 0, 0, 0, extra))
            if token_ids is not None:
                token_ids = F.pad(token_ids, (0, extra), value=self.eos_token_id)
            if labels is not None:
                labels = F.pad(labels, (0, extra))
            if loss_mask is not None:
                loss_mask = F.pad(loss_mask, (0, extra))
            padding = torch.arange(original_length + extra, device=decoder_input.device) >= original_length
            padding = padding.expand(decoder_input.shape[1], -1).contiguous()
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, extra, 0, extra), value=True)
                attention_mask[..., original_length:, :original_length] = False
        if attention_mask is None:
            length = decoder_input.shape[0]
            attention_mask = ~torch.ones(1, 1, length, length, device=decoder_input.device, dtype=torch.bool).tril()
            if extra:
                attention_mask[..., original_length:] = True
        if self.config.sequence_parallel:
            decoder_input = scatter_to_sequence_parallel_region(decoder_input, group=self.decoder.pg_collection.tp)
            if padding is not None:
                padding = (
                    scatter_to_sequence_parallel_region(
                        padding.transpose(0, 1).contiguous(), group=self.decoder.pg_collection.tp
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
        preprocessed = self._preprocess(input_ids, position_ids, decoder_input=decoder_input, padding_mask=padding)
        hidden, rope, rope_cos, rope_sin, offset, padding = preprocessed[:6]
        hidden = hidden.repeat(1, 1, self.config.hc_count)
        for layer in self.decoder.layers:
            forward = partial(self._run_layer, layer)
            if self.training and self.config.recompute_granularity == "full":
                hidden = checkpoint(
                    forward,
                    hidden,
                    token_ids,
                    attention_mask,
                    rope,
                    padding,
                    use_reentrant=False,
                    context_fn=partial(self._checkpoint_contexts, layer),
                )
            else:
                hidden = forward(hidden, token_ids, attention_mask, rope, padding)
        hidden = self.decoder.final_layernorm(hidden)
        output = self._postprocess(
            hidden,
            input_ids,
            position_ids,
            labels,
            rope,
            rope_cos,
            rope_sin,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            padding_mask=padding,
            sequence_len_offset=offset,
            runtime_gather_output=runtime_gather_output,
            output_processor=output_processor,
            output_processor_context=output_processor_context,
        )
        return output[:, :original_length] if extra else output
