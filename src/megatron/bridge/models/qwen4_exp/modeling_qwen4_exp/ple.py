import torch
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextNGramEmbedding,
    Qwen4ExpTextPLELayer,
    _build_layer_multipliers,
)


class Qwen4ExpSplitVocabEmbedding(nn.Module):
    """Keep HF storage shards while reducing vocabulary-parallel outputs once."""

    def __init__(self, vocabulary: int, hidden_size: int, parts: int, config, tp_group) -> None:
        super().__init__()
        if vocabulary % parts or (vocabulary // parts) % config.tensor_model_parallel_size:
            raise ValueError("PLE storage shards must divide the vocabulary and TP partitions exactly.")
        self.rows_per_shard = vocabulary // parts
        self.tp_group = tp_group
        for index in range(parts):
            self.add_module(
                f"shard_{index}",
                VocabParallelEmbedding(
                    self.rows_per_shard, hidden_size, init_method=config.init_method, config=config, tp_group=tp_group
                ),
            )

    @property
    def weight(self):
        """Expose a local shard only for the HF hash lookup's device selection."""
        return self.shard_0.weight

    def forward(self, token_ids):
        output = None
        for index, shard in enumerate(self.children()):
            relative = token_ids - index * self.rows_per_shard - shard.vocab_start_index
            valid = (relative >= 0) & (relative < shard.weight.shape[0])
            local = F.embedding(relative.clamp(0, shard.weight.shape[0] - 1), shard.weight)
            local = local * valid.unsqueeze(-1)
            output = local if output is None else output + local
        return reduce_from_tensor_model_parallel_region(output, group=self.tp_group)


class Qwen4ExpParallelNGramEmbedding(Qwen4ExpTextNGramEmbedding):
    """Preserve HF n-gram hashing with a tensor-parallel vocabulary table."""

    def __init__(self, hf_config, config, layer_idx: int, ple_layer_index: int, tp_group) -> None:
        with torch.device("meta"):
            super().__init__(hf_config, config.ple_embed_dim, layer_idx, ple_layer_index)
        self.layer_multipliers = _build_layer_multipliers(
            self.unigram_vocab_size, self.ngram_size, self.ple_layer_index, self.seed
        )
        self.ngram_heads_vocab_sizes = torch.tensor(self.head_vocab_sizes, dtype=torch.long)
        self.ngram_heads_offsets = torch.tensor(self.head_offsets, dtype=torch.long)
        vocabulary = self.ngram_embedding.num_embeddings
        if vocabulary % config.tensor_model_parallel_size:
            raise ValueError("PLE padded vocabulary must be divisible by tensor parallel size.")
        self.ngram_embedding = Qwen4ExpSplitVocabEmbedding(
            vocabulary, config.ple_embed_dim // self.ngram_heads, hf_config.split_ngram_parts, config, tp_group
        )


class Qwen4ExpParallelPLE(Qwen4ExpTextPLELayer):
    """Use the reference PLE computation with MCore vocabulary sharding."""

    def __init__(self, hf_config, config, layer_idx: int, ple_layer_index: int, tp_group) -> None:
        nn.Module.__init__(self)
        self.sequence_parallel = config.sequence_parallel
        self.tp_group = tp_group
        with torch.device("meta"):
            reference = Qwen4ExpTextPLELayer(hf_config, layer_idx, ple_layer_index)
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.short_conv_state_len = reference.short_conv_state_len
        for name in ("key_proj", "value_proj", "norm_key", "norm_query", "norm_conv", "conv1d"):
            module = getattr(reference, name).to_empty(device="cpu")
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
            else:
                nn.init.zeros_(module.weight)
            self.add_module(name, module)
        self.ple_embedding = Qwen4ExpParallelNGramEmbedding(hf_config, config, layer_idx, ple_layer_index, tp_group)

    def forward_sequence(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute full-context PLE and return the local sequence partition."""
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
            )
        output = self(hidden_states.transpose(0, 1), input_ids, None).transpose(0, 1).contiguous()
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output
