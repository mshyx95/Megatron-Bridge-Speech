# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/env python3
"""ConformerEncoder Module"""

from typing import Optional, Tuple, List, Literal
import abc
import math
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from .activation_checkpointing import (
    encoder_checkpoint_wrapper,
    attn_checkpointing,
    embedding_checkpoint_wrapper,
)

from .utils import (
    adaptive_enc_mask,
    MeanVarianceNormLayer,
    NemoConvSubsampling,
    Conv2dSubsampling,
    RelativePositionalEncoding,
    AbsolutePositionalEncoding,
    T5RelativeAttentionLogitBias,
    ConvModule,
    FeedForward,
    MultiHeadedAttention,
    unfold_tensor,
)


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential"""

    @torch.jit.ignore
    def forward(self, *args):
        """Forward method implementation."""
        for m in self:
            args = m(*args)
        return args

def repeat(repeat_num, module_gen_fn):
    """repeat module N times

    :param int repeat_num: repeat time
    :param function module_gen_fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[module_gen_fn(i) for i in range(repeat_num)])

class ConformerEncoderLayer(nn.Module):
    """ConformerEncoder Layer module.
    for more details see conformer paper:
        https://arxiv.org/abs/2005.08100
    This module implement the Conformer block layer.

    Args:
        d_model: int
            attention dim.
        ext_pw_out_channel: int
            if > 0, ext_pw_out_channel is a dim channel size
             for the last pointwise conv after swish activation.
        depthwise_seperable_out_channel: int
            if set different to 0, the number of depthwise_seperable_out_channel
             will be used as a channel_out of the second conv1d layer.
             otherwise, it equal to 0, the second conv1d layer is skipped.
        depthwise_multiplier: int
            number of input_dim channels duplication. this value
             will be used to compute the hidden channels of the Conv1D.
        n_head: int
            the number of heads for multihead attention module.
        d_ffn: int
            output size of the feed_forward blocks.
        ext_pw_kernel_size: int
            kernel size of the conv pointwise of the conformer.
        kernel_size: int
            kernel size.
        dropout_rate: float
            dropout rate.
        causal: bool, optional
            if set to True, convolution have no access
             to future frames. default False.
        batch_norm: bool, optional
            if set to True, apply batchnorm before activation
            in ConvModule layer of the conformer.
            default False
        activation: str, optional
            activation function name,
            one of ["relu", "swish", "sigmoid"],
            sigmoid activation is only used with "glu_in_fnn=True",
            default "relu".
        chunk_se: int, optional
            0 for offline SE.
            1 for streaming SE, where mean is computed
             by accumulated history until current chunk_se.
            2 for streaming SE, where mean is computed
             by only the current chunk.
            default 0.
        chunk_size: int, optional
            chunk_size for cnn. default 18
        conv_activation: str, optional
            activation function used in ConvModule part
            of the conformer, default "relu".
        conv_glu_type: str, optional
            activation function used for the glu inside
            the ConvModule part of the conformer.
            default: "sigmoid".
        bias_in_glu: bool, optional
            if set to True, use additive bias in the weight module
             before GLU.
        linear_glu_in_convm: bool, optional
            if set to True, use GLULinear module,
             otherwise, used GLUPointWiseConv module.
              default to False.
        attention_innner_dim: int, otional
            if equal to -1, attention dim for linears k/q/v is
            equal to d_model. otherwise attention_innner_dim is used.
            default -1.
        attention_glu_type: str, optional
            activation function for glu used in the multihead attention,
             default "swish".
        activation_checkpointing: str, optional
            a dictionarry of {"module","interval","offload"}, where
                "module": str
                    accept ["transformer", "attention"] to select
                    which module should do activation checkpointing.
                "interval": int, default 1,
                    interval of applying activation checkpointing,
                    interval = 1 means that we apply checkpointing
                    on every layer (if activation), otherwise,
                    we apply it every x interval.
                "offload": bool, default False,
                    if set to True, we offload activation to cpu and
                    reload it during backward, otherwise,
                    we recalculate activation in backward.
            default "".
        export: bool, optional
            if set to True, it remove the padding from convolutional layers
             and allow the onnx conversion for inference.
              default False.
        use_pt_scaled_dot_product_attention: bool, optional
            if set to True, use pytorch's scaled dot product attention implementation in training.
        attn_group_sizes: int, optional
            the number of groups to use for attention, default 1 (Multi-Head Attention),
            1 = typical Multi-Head Attention,
            1 < attn_group_sizes < attention_heads = Grouped-Query Attention
            attn_group_sizes = attenion_heads = Multi-Query Attention
    """

    def __init__(
        self,
        d_model=512,
        ext_pw_out_channel=0,
        depthwise_seperable_out_channel=256,
        depthwise_multiplier=1,
        n_head=4,
        d_ffn=2048,
        ext_pw_kernel_size=1,
        kernel_size=3,
        dropout_rate=0.1,
        causal=False,
        batch_norm=False,
        activation="relu",
        chunk_se=0,
        chunk_size=18,
        conv_activation="relu",
        conv_glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        attention_innner_dim=-1,
        attention_glu_type="swish",
        activation_checkpointing="",
        export=False,
        use_pt_scaled_dot_product_attention=False,
        attn_group_sizes=1,
        conv_threshold=8,
        th=0,
        use_flashattention_t5bias=False,
    ):
        super().__init__()

        self.feed_forward_in = FeedForward(
            d_model=d_model,
            d_inner=d_ffn,
            dropout_rate=dropout_rate,
            activation=activation,
            bias_in_glu=bias_in_glu,
        )

        self.self_attn = encoder_checkpoint_wrapper(
            activation_checkpointing,
            MultiHeadedAttention,
        )(
            MultiHeadedAttention(
                n_head,
                d_model,
                dropout_rate,
                attention_innner_dim,
                attention_glu_type,
                bias_in_glu,
                use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention,
                group_size=attn_group_sizes,
                use_flashattention_t5bias=use_flashattention_t5bias,
            )
        )
        
        self.th = th
        self.use_conv = (self.th < conv_threshold)
        if self.use_conv:
            self.conv = ConvModule(
                d_model,
                ext_pw_out_channel,
                depthwise_seperable_out_channel,
                ext_pw_kernel_size,
                kernel_size,
                depthwise_multiplier,
                dropout_rate,
                causal,
                batch_norm,
                chunk_se,
                chunk_size,
                conv_activation,
                conv_glu_type,
                bias_in_glu,
                linear_glu_in_convm,
                export=export,
            )
        else:
            self.conv = None

        self.feed_forward_out = FeedForward(
            d_model=d_model,
            d_inner=d_ffn,
            dropout_rate=dropout_rate,
            activation=activation,
            bias_in_glu=bias_in_glu,
        )

        self.layer_norm_att = nn.LayerNorm(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x,
        pos_k,
        pos_v,
        mask,
        mask_seqlens: Optional[Tuple[Tensor, Tensor, bool]] = None,
        relative_attention_bias: Optional[Tensor] = None,
    ):
        """ConformerEncoder forward.

        Args:
            x: torch.Tensor
                input feature of shape (batch, max_time_in, size)
            pos_k: torch.Tensor
                positional key embedding.
            mask: torch.Tensor
                mask for x (batch, max_time_in)
            relative_attention_bias: Optional[torch.Tensor]
                bias added to attention logits w.r.t. relative positions (1, n_head, time1, time2)
        """
        x = x + 0.5 * self.feed_forward_in(x)
        norm_x = self.layer_norm_att(x)

        x = x + self.self_attn(
            norm_x,
            norm_x,
            norm_x,
            pos_k,
            pos_v,
            mask,
            mask_seqlens=mask_seqlens,
            relative_attention_bias=relative_attention_bias,
        )
        if self.use_conv and self.conv is not None:
            x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)

        out = self.layer_norm(x)

        return out, pos_k, pos_v, mask
        
class TransformerEncoderBase(abc.ABC, nn.Module):
    """The Base class for Transformer based encoders

    Please set causal = True in streaming model
    Args:
        input_size: int
            input feature dimension.
        chunk_size: int, list(int)
            Number of frames for each chunk
            This variable can take 2 forms:
            int:  Used for inference, or single chunk size training
            list(int) : Used only for variable chunk size training
            Some examples for the 2 cases:
            chunk_size = 12
            chunk_size = [6, 8, 12, 24]
        left_chunk: int, list(int)
            Number of chunks used for masking in streaming mode.
            This variable can take 2 forms:
            int:  Used for inference, or single chunk size training
            list(int) : Used only for variable chunk size training. When
            chunk_size is a list, left_chunk must be a list with same length.
            Some examples for the 2 cases:
            left_chunk = 6
            left_chunk = [12, 9, 6, 3]
        attention_dim: int, optional
            attention dimension. default 256.
        attention_heads: int, optional
            the number of heads. default 4
        input_layer: str, optional
            input layer type before Conformer,
            one of ["linear", "conv2d", "custom", "vgg2l", "embed"],
            default "conv2d"
        cnn_out: int, optional
            the number of CNN channels before Conformer.
            default -1.
        cnn_layer_norm: bool, optional
            layer norm between Conformer and the first CNN.
            default False.
        time_reduction: int, optional
            time reduction factor
            default 4
        dropout_rate: float, optional
            dropout rate. default 0.1
        padding_idx: int, optional
            padding index for input_layer=embed
            default -1
        relative_attention_bias_args: dict, optional
            use more efficient scalar bias-based relative multihead attention (Q*K^T + B)
            implemented in cmb.basics.embedding.[T5/ALiBi]RelativeAttentionLogitBias
            usage: relative_attention_bias_args={"type": t5/alibi}
            additional method-specific arguments can be provided (see transformer_base.py)
        positional_dropout_rate: float, optional
            dropout rate after positional encoding. default 0.0
        nemo_conv_settings: dict, optional
            A dictionary of settings for NeMo Subsampling.
            default None
        conv2d_extra_padding: str, optional
            Add extra padding in conv2d subsampling layers. Choices are
            (feat, feat_time, none, True).
            if True or feat_time, the extra padding is added into non full
            supraframe utts in batch.
            Default: none
        attention_group_size: int, optional
            the number of groups to use for attention, default 1 (Multi-Head Attention),
            1 = typical Multi-Head Attention,
            1 < attention_group_size < attention_heads = Grouped-Query Attention
            attention_group_size = attenion_heads = Multi-Query Attention
    """

    def __init__(
        self,
        input_size,
        chunk_size,
        left_chunk,
        attention_dim=256,
        attention_heads=4,
        input_layer="nemo_conv",
        cnn_out=-1,
        cnn_layer_norm=False,
        time_reduction=4,
        dropout_rate=0.0,
        padding_idx=-1,
        relative_k=False,
        relative_v=False,
        relative_attention_bias_args=None,
        positional_dropout_rate=0.0,
        nemo_conv_settings=None,
        conv2d_extra_padding: Literal["feat", "feat_time", "none", True] = "none",
        attention_group_size=1,
        encoder_embedding_config=None,
        use_flashattention_t5bias: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_layer = input_layer
        self.chunk_size = chunk_size
        self.left_chunk = left_chunk
        self.attention_dim = attention_dim
        self.num_heads = attention_heads
        self.attention_group_size = attention_group_size
        self.time_reduction = time_reduction
        self.nemo_conv_settings = nemo_conv_settings
        self.encoder_embedding_config = encoder_embedding_config
        self.use_flashattention_t5bias = use_flashattention_t5bias

        if self.input_layer == "nemo_conv":
            default_nemo_conv_settings = {
                "subsampling": "dw_striding",
                "subsampling_factor": self.time_reduction,
                "feat_in": input_size,
                "feat_out": attention_dim,
                "conv_channels": 256,
                "subsampling_conv_chunking_factor": 1,
                "activation": nn.ReLU(),
                "is_causal": False,
            }
            # Override any of the defaults with the incoming, user settings
            if nemo_conv_settings:
                default_nemo_conv_settings.update(nemo_conv_settings)
                for i in ["subsampling_factor", "feat_in", "feat_out"]:
                    assert (
                        i not in nemo_conv_settings
                    ), "{i} should be specified outside of the NeMo dictionary"

            self.embed = NemoConvSubsampling(
                **default_nemo_conv_settings,
            )
        elif self.input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                attention_dim,
                cnn_out=cnn_out,
                layer_norm=cnn_layer_norm,
                time_reduction=self.time_reduction,
                extra_padding=conv2d_extra_padding,
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.relative_k = relative_k
        if relative_k:
            self.pos_emb = RelativePositionalEncoding(
                attention_dim // attention_heads, 1000, relative_v
            )
            self.dropout_layer = torch.nn.Dropout(p=positional_dropout_rate)
        else:
            # There is a dropout applied inside it PosENC
            # where not applied in rel pos enc.
            # TODO
            self.pos_emb = AbsolutePositionalEncoding(attention_dim, positional_dropout_rate)

        self.relative_attention_bias_type = (
            relative_attention_bias_args.get("type") if relative_attention_bias_args else None
        )
        if self.relative_attention_bias_type == "t5":
            assert (
                self.num_heads % self.attention_group_size == 0
            ), "attention_group_size must divide n_head"
            self.relative_attention_bias_layer = T5RelativeAttentionLogitBias(
                self.num_heads // self.attention_group_size,
                max_distance=relative_attention_bias_args.get("t5_bias_max_distance", 1000),
                symmetric=relative_attention_bias_args.get("t5_bias_symmetric", False),
            )
        else:
            self.relative_attention_bias_layer = None

    
    def post_init(self, init_model_config):

        pretrained_speech_encoder_path = init_model_config.get('pretrained_speech_encoder_path', None)
        if pretrained_speech_encoder_path:
            if pretrained_speech_encoder_path.startswith("az://"):
                import blobfile as bf
                with bf.BlobFile(pretrained_speech_encoder_path, "rb") as f:
                    model_state = torch.load(f, map_location="cpu")
            else:
                model_state = torch.load(pretrained_speech_encoder_path, map_location="cpu")
                
            if "module" in model_state:
                # loading from ds checkpoint:
                is_ds_checkpoint = True
                model_state = model_state["module"]
            else:
                # loading cascades checkpoint:
                is_ds_checkpoint = False

            encoder_state_dict = {}
            for k, v in model_state.items():
                if is_ds_checkpoint and "encoder." in k:
                    tmp_k = k.replace("model.embed_tokens_extend.encoder.", "")
                    encoder_state_dict[tmp_k] = v
                elif "encoder." in k:
                    tmp_k = k.replace("encoder.", "")
                    encoder_state_dict[tmp_k] = v
            
            if hasattr(self, "encoder_embedding"):
                del self.encoder_embedding
            self.load_state_dict(encoder_state_dict, strict=False)
        
        if not hasattr(self, "encoder_embedding"):
            self.encoder_embedding = MeanVarianceNormLayer(self.encoder_embedding_config["input_size"])
       
        mean_file = init_model_config.get('mean_file', None)
        invstd_file = init_model_config.get('invstd_file', None)
        if mean_file is not None and invstd_file is not None:
            self.encoder_embedding.load_mean_invstd(mean_file, invstd_file)

    def compute_lens_change(self, feature_lens):
        """feature_lens: int
        return updated feature lens.

        This used to return a different lambda function for each case that computed
        the right thing.  That does not work within Torchscript.  If you really
        need this to be faster, create nn.Module()-s for all the cases and return
        one of them.  Torchscript does support that.
        """
        if self.input_layer == "nemo_conv":
            # Handle the special causal case
            subsampling_causal_cond = self.nemo_conv_settings.get("subsampling", "dw_striding") in [
                "dw_striding",
                "striding",
                "striding_conv1d",
            ]
            is_causal = self.nemo_conv_settings.get("is_causal", False)
            if is_causal and subsampling_causal_cond:
                lens_change = (
                    torch.ceil(feature_lens / self.time_reduction).long()
                    if isinstance(feature_lens, Tensor)
                    else math.ceil(feature_lens / self.time_reduction)
                )
                feature_lens_remainder = feature_lens % self.time_reduction
                if isinstance(feature_lens, Tensor):
                    lens_change[feature_lens_remainder != 1] += 1
                elif feature_lens_remainder != 1:
                    lens_change += 1
                return lens_change
            ceil_func = math.ceil if isinstance(feature_lens, int) else torch.ceil
            return ceil_func(feature_lens / self.time_reduction)

        if self.input_layer == "conv2d":
            return self.embed.compute_length_change(feature_lens)

    @abc.abstractmethod
    def forward(self):
        """Abstract forward method implementation."""

    def _chunk_size_selection(self, chunk_size=None, left_chunk=None):
        """If chunk size is a list, we will randomly select a chunk size."""

        if chunk_size is None:
            chunk_size = self.chunk_size
        if left_chunk is None:
            left_chunk = self.left_chunk
        if isinstance(chunk_size, list):
            # Variable chunk size during training
            chunk_size_index = int(torch.randint(low=0, high=len(chunk_size), size=(1,)))
            chunk_size_train_eff = chunk_size[chunk_size_index]
            if not isinstance(left_chunk, list):
                raise ValueError("Since chunk_size is a list, left_chunk must be a list")
            if len(left_chunk) != len(chunk_size):
                raise ValueError(
                    "The length of left_chunk must be the same as length of chunk_size."
                )
            left_chunk_train_eff = left_chunk[chunk_size_index]
        else:
            chunk_size_train_eff = chunk_size
            left_chunk_train_eff = left_chunk

        return chunk_size_train_eff, left_chunk_train_eff

    def _get_embed_class(self, embed):
        # pylint: disable=protected-access
        is_embed_using_act_chkpt = isinstance(embed, CheckpointWrapper)
        is_embed_fsdp_wrapped = isinstance(embed, FullyShardedDataParallel)
        embed_class = embed
        if is_embed_using_act_chkpt:
            embed_class = embed._checkpoint_wrapped_module
        if is_embed_fsdp_wrapped:
            embed_class = embed.module
        return embed_class

    def _forward_embeddings_core(self, input_tensor, masks):
        embed_class = self._get_embed_class(self.embed)
        if isinstance(embed_class, Conv2dSubsampling):
            # Conv2dSubsampling use Ilens instead of mask.
            # this allow to handle padding of non full supraframe utt in batch.
            input_tensor, masks = self.embed(input_tensor, masks.sum(-1))
        elif isinstance(embed_class, NemoConvSubsampling):
            input_tensor, masks = self.embed(input_tensor, masks)    
        return input_tensor, masks

    def _position_embedding(self, input_tensor):
        if self.relative_k:
            x_len = input_tensor.shape[1]
            pos_seq = torch.arange(0, x_len).long().to(input_tensor.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, pos_v = self.pos_emb(pos_seq)
            input_tensor = self.dropout_layer(input_tensor)
        else:
            pos_k = None
            pos_v = None
            if self.relative_attention_bias_layer is None:
                input_tensor = self.pos_emb(input_tensor)  # default to add abs sinusoid embedding
        return pos_k, pos_v

    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk, device=None):
        chunk_size_train_eff, left_chunk_train_eff = self._chunk_size_selection(
            chunk_size, left_chunk
        )
        chunk_size_train_eff = int(chunk_size_train_eff)
        left_chunk_train_eff = int(left_chunk_train_eff)

        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        # Non-streaming case: full attention
        if chunk_size_train_eff <= 0:
            return torch.ones(
                (1, seq_len, seq_len), device=device, dtype=torch.bool
            ).expand(batch_size, -1, -1)

        seq_range = torch.arange(seq_len, device=device, dtype=torch.long)

        # avoid randomness when run evaluation or decoding
        if self.training and np.random.rand() > 0.5:
            # Random chunk offset: first chunk is shorter for training robustness
            # Offset = seq_len % chunk_size (first chunk gets the remainder)
            offset = seq_len % chunk_size_train_eff
            if offset == 0:
                offset = chunk_size_train_eff  # avoid empty first chunk

            # Chunk assignment: pos < offset → chunk 0, else 1 + floor((pos - offset) / cs)
            shifted = torch.clamp(seq_range - offset, min=0)
            chunk_idx = torch.where(
                seq_range < offset,
                torch.zeros_like(seq_range),
                1 + torch.div(shifted, chunk_size_train_eff, rounding_mode="floor")
            )

            # Left visible boundary: start position of (chunk_idx - left_chunk)
            left_visible_chunk = torch.clamp(chunk_idx - left_chunk_train_eff, min=0)
            left_boundary = torch.where(
                left_visible_chunk == 0,
                torch.zeros_like(seq_range),
                offset + (left_visible_chunk - 1) * chunk_size_train_eff
            )

            # Right boundary: end position of current chunk
            right_boundary = torch.where(
                chunk_idx == 0,
                torch.full_like(seq_range, offset),
                torch.clamp(offset + (chunk_idx) * chunk_size_train_eff, max=seq_len)
            )
        else:
            # Regular chunks: [0, cs), [cs, 2*cs), ...
            chunk_idx = torch.div(seq_range, chunk_size_train_eff, rounding_mode="floor")
            left_boundary = torch.clamp(
                chunk_idx - left_chunk_train_eff, min=0
            ) * chunk_size_train_eff
            right_boundary = torch.clamp(
                (chunk_idx + 1) * chunk_size_train_eff, max=seq_len
            )

        seq_range_expand = seq_range.unsqueeze(0)
        mask = (seq_range_expand >= left_boundary.unsqueeze(1)) & (
            seq_range_expand < right_boundary.unsqueeze(1)
        )
        enc_streaming_mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        return enc_streaming_mask

    def forward_embeddings(self, xs_pad, masks, chunk_size_nc=None, left_chunk_nc=None):
        """Forwarding the inputs through the top embedding layers

        Args:
            xs_pad: torch.Tensor
                input tensor
            masks: torch.Tensor
                input mask
            chunk_size_nc: (optional, default is None) chunk size for non-causal layers
            left_chunk_nc: (optional, default is None) # of left chunks for non-causal layers
        """
        # pylint: disable=R0915
        # get new lens.
        seq_len = int(self.compute_lens_change(xs_pad.shape[1]))
        if seq_len <= 0:
            raise ValueError(
                f"""The squence length after time reduction is invalid: {seq_len}.
                Your input feature is too short. Consider filtering out the very
                short sentence from data loader""",
            )

        batch_size = xs_pad.shape[0]
        device = xs_pad.device

        enc_streaming_mask = self._streaming_mask(
            seq_len, batch_size, self.chunk_size, self.left_chunk, device=device
        )

        input_tensor = xs_pad
        input_tensor, masks = self._forward_embeddings_core(input_tensor, masks)

        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = masks & streaming_mask
        elif masks is not None:
            hs_mask = masks
        else:
            hs_mask = streaming_mask

        if chunk_size_nc is not None:
            enc_streaming_mask_nc = self._streaming_mask(
                seq_len, batch_size, chunk_size_nc, left_chunk_nc, device=device
            )
            if masks is not None:
                hs_mask_nc = masks & enc_streaming_mask_nc
            else:
                hs_mask_nc = enc_streaming_mask_nc
        else:
            hs_mask_nc = None

        pos_k, pos_v = self._position_embedding(input_tensor)

        if chunk_size_nc is None:
            return input_tensor, pos_k, pos_v, hs_mask, masks
        return input_tensor, pos_k, pos_v, hs_mask, masks, hs_mask_nc

    def get_offset(self):
        """Returns offset used when retaining inputs for decoding.

        This is essentially, how many additional frames have to be added to
        the front-end CNN input to ensure it can produce a single output.
        So if the "padding" parameter is 0, typically offset will be > 0.
        """
        return get_offset(self.input_layer, self.time_reduction)


def get_offset(input_layer: str, time_reduction: int):
    """Get an offset. We will use the offset for determining #frames of a subsampled feature.

    Args:
        input_layer (str): Type of an input layer
        time_reduction (int): time reduction factor for downsampling a feature
    Returns:
        int: offset
    """
    if input_layer in ("conv2d", "nemo_conv") and time_reduction == 4:
        return 3
    if input_layer in ("conv2d",) and time_reduction == 6:
        return 1
    if input_layer in ("conv2d", "nemo_conv") and time_reduction == 8:
        return 7
    return 0


class ConformerEncoder(TransformerEncoderBase):
    """ConformerEncoder module.
    see original paper for more details:
        https://arxiv.org/abs/2005.08100

    Please set causal = True in streaming model
    Args:
        input_size: int
            input feature dimension.
        chunk_size: int, list(int)
            Number of frames for each chunk
            This variable can take 2 forms:
            int:  Used for inference, or single chunk size training
            list(int) : Used only for variable chunk size training
            Some examples for the 2 cases:
            chunk_size = 12
            chunk_size = [6, 8, 12, 24]
        left_chunk: int, list(int)
            Number of chunks used for masking in streaming mode.
            This variable can take 2 forms:
            int:  Used for inference, or single chunk size training
            list(int) : Used only for variable chunk size training. When
            chunk_size is a list, left_chunk must be a list with same length.
            Some examples for the 2 cases:
            left_chunk = 6
            left_chunk = [12, 9, 6, 3]
        left_chunk: int
            number of chunks used for masking in streaming mode.
        num_lang: int
            This parameter is used to store the number of languages in the lang_dict,
            only used for multiseed/multilingual models. default None.
        attention_dim: int, optional
            attention dimension. default 256.
        attention_heads: int, optional
            the number of heads. default 4
        linear_units:
            the number of units of position-wise feed forward.
            default 2048
        num_block:
            number of Transformer layer. default 6
        dropout_rate: float, optional
            dropout rate. default 0.1
        input_layer: str, optional
            input layer type before Conformer,
            one of ["linear", "conv2d", "custom", "vgg2l", "embed"],
            default "conv2d"
        causal: bool, optional
            if set to True, convolution have no access
             to future frames. default False.
        batch_norm: bool, optional
            if set to True, apply batchnorm before activation
            in ConvModule layer of the conformer.
            default False
        cnn_out: int, optional
            the number of CNN channels before Conformer.
            default -1.
        cnn_layer_norm: bool, optional
            layer norm between Conformer and the first CNN.
            default False.
        ext_pw_out_channel: int, optional
            the number of channel for CNN
            before depthwise_seperable_CNN.
            If 0 then use linear. default 0.
        ext_pw_kernel_size: int, optional
            kernel size of N before depthwise_seperable_CNN.
            only work for ext_pw_out_channel > 0.
            default 1
        depthwise_seperable_out_channel: int, optional
            the number of channel for
            depthwise_seperable_CNN.
            default 256.
        depthwise_multiplier: int, optional
            the number of multiplier for
            depthwise_seperable_CNN.
            default 1.
        chunk_se: int, optional
            0 for offline SE.
            1 for streaming SE, where mean is computed
             by accumulated history until current chunk_se.
            2 for streaming SE, where mean is computed
             by only the current chunk.
            default 0.
        kernel_size: int, optional
            the number of kernels for depthwise_seperable_CNN.
            default 3.
        activation: str, optional
            FeedForward block activation.
            one of ["relu", "swish", "sigmoid"]
            default "relu".
        conv_activation: str, optional
            activation function used in ConvModule part
            of the conformer, default "relu".
        conv_glu_type: str, otional
            activation used use glu in depthwise_seperable_CNN,
            default "sigmoid"
        bias_in_glu: bool, optional
            if set to True, use additive bias in the weight module
             before GLU. default True
        linear_glu_in_convm: bool, optional
            if set to True, use GLULinear module,
             otherwise, used GLUPointWiseConv module.
              default to False.
        attention_glu_type: str
            only work for glu_in_attention !=0
            default "swish".
        export: bool, optional
            if set to True, it remove the padding from convolutional layers
             and allow the onnx conversion for inference.
              default False.
        activation_checkpointing: str, optional
            a dictionarry of {"module","interval","offload"}, where
                "module": str
                    accept ["transformer", "attention"] to select
                    which module should do activation checkpointing.
                "interval": int, default 1,
                    interval of applying activation checkpointing,
                    interval = 1 means that we apply checkpointing
                    on every layer (if activation), otherwise,
                    we apply it every x interval.
                "offload": bool, default False,
                    if set to True, we offload activation to cpu and
                    reload it during backward, otherwise,
                    we recalculate activation in backward.
            default "".
        extra_layer_output_idx: int
            the layer index to be exposed.
        relative_attention_bias_args: dict, optional
            use more efficient scalar bias-based relative multihead attention (Q*K^T + B)
            implemented in cmb.basics.embedding.[T5/ALiBi]RelativeAttentionLogitBias
            usage: relative_attention_bias_args={"type": t5/alibi}
            additional method-specific arguments can be provided (see transformer_base.py)
        time_reduction: int optional
            time reduction factor
            default 4
        use_pt_scaled_dot_product_attention: whether to use pytorch scaled dot product attention
            in training.
            Default: False
        nemo_conv_settings: dict, optional
            A dictionary of settings for NeMo Subsampling.
            default: None
            usage: nemo_conv_settings=
                {
                    "subsampling":
                        dw_striding/striding/dw_striding_conv1d/striding_conv1d,
                    "conv_channels": int,
                    "subsampling_conv_chunking_factor": int,
                    "is_causal": True/False
                }
        conv2d_extra_padding: str, optional
            Add extra padding in conv2d subsampling layers. Choices are
            (feat, feat_time, none, True)
            Default: none
        replication_pad_for_subsample_embedding:  For batched-streaming decoding, use
            "replication" padding for the cache at start of utterance.
             Default: False
        attention_group_size: int, optional
            the number of groups to use for attention, default 1 (Multi-Head Attention),
            1 = typical Multi-Head Attention,
            1 < attention_group_size < attention_heads = Grouped-Query Attention
            attention_group_size = attenion_heads = Multi-Query Attention
    """

    extra_multi_layer_output_idxs: List[int]

    def __init__(  # pylint: disable-all
        self,
        input_size,
        chunk_size,
        left_chunk,
        num_lang=None,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        input_layer="nemo_conv",
        causal=True,
        batch_norm=False,
        cnn_out=-1,
        cnn_layer_norm=False,
        ext_pw_out_channel=0,
        ext_pw_kernel_size=1,
        depthwise_seperable_out_channel=256,
        depthwise_multiplier=1,
        chunk_se=0,
        kernel_size=3,
        activation="relu",
        conv_activation="relu",
        conv_glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        attention_glu_type="swish",
        export=False,
        extra_layer_output_idx=-1,
        extra_multi_layer_output_idxs=[],
        activation_checkpointing="",
        relative_k=False,
        relative_v=False,
        relative_attention_bias_args=None,
        time_reduction=4,
        use_pt_scaled_dot_product_attention=False,
        nemo_conv_settings=None,
        conv_threshold=8,
        conv2d_extra_padding: Literal["feat", "feat_time", "none", True] = "none",
        replication_pad_for_subsample_embedding=False,
        attention_group_size=1,
        encoder_embedding_config=None,
        use_flashattention_t5bias=False,
    ):
        super().__init__(
            input_size,
            chunk_size,
            left_chunk,
            attention_dim,
            attention_heads,
            input_layer,
            cnn_out,
            cnn_layer_norm,
            time_reduction,
            dropout_rate=dropout_rate,
            relative_k=relative_k,
            relative_v=relative_v,
            relative_attention_bias_args=relative_attention_bias_args,
            positional_dropout_rate=0.0,
            nemo_conv_settings=nemo_conv_settings,
            conv2d_extra_padding=conv2d_extra_padding,
            attention_group_size=attention_group_size,
            encoder_embedding_config=encoder_embedding_config,
        )
        self.num_blocks = num_blocks
        self.num_lang = num_lang
        self.kernel_size = kernel_size
        self.embed = embedding_checkpoint_wrapper(activation_checkpointing)(self.embed)
        self.replication_pad_for_subsample_embedding: bool = replication_pad_for_subsample_embedding
        assert self.num_heads % attention_group_size == 0, "attention_group_size must divide n_head"
        self.num_heads_k = self.num_heads // attention_group_size
        self.is_streaming_encoder = (chunk_size > -1)
        self.use_flashattention_t5bias = use_flashattention_t5bias
        if use_flashattention_t5bias:
            _rab_type = (relative_attention_bias_args or {}).get("type", "none")
            print(
                f"[ConformerEncoder] flash_attn_t5bias_cuda kernel ENABLED "
                f"(relative_attention_bias_type={_rab_type}, "
                f"num_blocks={num_blocks}, attention_heads={attention_heads})"
            )

        self.encoders = repeat(
            num_blocks,
            lambda i: encoder_checkpoint_wrapper(
                activation_checkpointing, ConformerEncoderLayer, i
            )(
                ConformerEncoderLayer(
                    d_model=attention_dim,
                    ext_pw_out_channel=ext_pw_out_channel,
                    depthwise_seperable_out_channel=depthwise_seperable_out_channel,
                    depthwise_multiplier=depthwise_multiplier,
                    n_head=attention_heads,
                    d_ffn=linear_units,
                    ext_pw_kernel_size=ext_pw_kernel_size,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                    causal=causal,
                    batch_norm=batch_norm,
                    activation=activation,
                    chunk_se=chunk_se,
                    chunk_size=chunk_size,
                    conv_activation=conv_activation,
                    conv_glu_type=conv_glu_type,
                    bias_in_glu=bias_in_glu,
                    linear_glu_in_convm=linear_glu_in_convm,
                    attention_glu_type=attention_glu_type,
                    activation_checkpointing=attn_checkpointing(activation_checkpointing, i),
                    export=export,
                    use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention,
                    attn_group_sizes=attention_group_size,
                    conv_threshold=conv_threshold,
                    th=i if self.is_streaming_encoder else 0,
                    use_flashattention_t5bias=use_flashattention_t5bias,
                )
            ),
        )
        self.extra_layer_output_idx = extra_layer_output_idx
        self.extra_multi_layer_output_idxs = extra_multi_layer_output_idxs
        # Make a zeros scalar we can use in get_initial_state to determine
        # the device and the needed dtype:
        self.register_buffer("dev_type", torch.zeros(()), persistent=False)

    def init_relative_attention_bias(self, input_tensor):
        if self.relative_attention_bias_layer:
            if self.use_flashattention_t5bias and self.relative_attention_bias_type == "t5":
                return self.relative_attention_bias_layer.get_t5_bias_table(
                    maxpos=input_tensor.size(1),
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                )
            return self.relative_attention_bias_layer(input_tensor)

    def calculate_hs_mask(self, xs_pad, device, mask):
        max_audio_length = xs_pad.shape[1]
        batch_size = xs_pad.shape[0]
        enc_streaming_mask = self._streaming_mask(
            max_audio_length, batch_size, self.chunk_size, self.left_chunk, device=device
        )
        if mask is None:
            return enc_streaming_mask

        feature_lens = mask.sum(1)
        padding_length = feature_lens
        pad_mask = (
            torch.arange(0, max_audio_length, device=device).expand(padding_length.size(0), -1)
            < padding_length.unsqueeze(1)
        )
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask & enc_streaming_mask
        return pad_mask

    @torch.jit.ignore
    def forward(self, xs_pad, masks):
        """Conformer Forward function

        Args:
            xs_pad: torch.Tensor
                input tensor
            masks: torch.Tensor
                post-embedding input lengths
        """
        xs_pad = self.encoder_embedding(xs_pad)
        input_tensor, pos_k, pos_v, hs_mask, masks = self.forward_embeddings(xs_pad, masks)

        unfolded = False
        masks_unfold = masks  # Track mask through unfold operation
        ori_bz, seq_len, D = input_tensor.shape
        max_seq_len = 500 #maxium position for absolute positional encoding
        # chunk long-form audio into 40s chunks for offline encoder
        if seq_len > max_seq_len and not self.is_streaming_encoder:
            # audio sequence is longer than max_seq_len, unfold it into chunks of max_seq_len
            unfolded = True
            # the unfold op will drop residual frames, pad it to the multiple of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0
            if chunk_pad_size > 0:
                input_tensor_pad = F.pad(input_tensor, (0, 0, 0, chunk_pad_size), "constant", 0)
                input_tensor = input_tensor_pad.to(input_tensor.device)

            input_tensor = unfold_tensor(input_tensor, max_seq_len)
            if masks is not None:
                # revise hs_mask here because the previous calculated hs_mask did not consider extra pad
                subsampled_pad_mask = masks.squeeze(1) # [bz, subsampled_unmask_seq_len]
                extra_padded_subsamlped_pad_mask = F.pad(subsampled_pad_mask, (0, chunk_pad_size), "constant", False) # extra padding to the pad mask
                extra_padded_subsamlped_pad_mask = extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                masks_unfold = unfold_tensor(extra_padded_subsamlped_pad_mask, max_seq_len) # unfold the pad mask like we did to the input tensor
                masks_unfold = masks_unfold.squeeze(-1).bool() # unfold op does not support bool tensor
            else:
                masks_unfold = None
            hs_mask = self.calculate_hs_mask(input_tensor, input_tensor.device, masks_unfold) # calculate hs_mask based on the unfolded pad mask
        
        layer_emb = None

        relative_attention_bias = self.init_relative_attention_bias(input_tensor)
        mask_seqlens = None
        # Only use varlen path when seq_len is large enough that pack/unpack overhead
        # is worth it.  At T<256 the fixed pack cost exceeds the kernel speedup.
        _varlen_threshold = 256
        if masks_unfold is not None and not self.is_streaming_encoder and input_tensor.size(1) >= _varlen_threshold:
            if masks_unfold.dim() == 3 and masks_unfold.size(1) == 1:
                padding_lens = masks_unfold.squeeze(1).sum(dim=-1).to(dtype=torch.int32)
            elif masks_unfold.dim() == 2:
                padding_lens = masks_unfold.sum(dim=-1).to(dtype=torch.int32)
            else:
                padding_lens = None
            if padding_lens is not None:
                mask_seqlens = (padding_lens, padding_lens, False)

        _simplified_path = (
            self.extra_layer_output_idx == -1
            and relative_attention_bias is None
        )

        if _simplified_path:
            input_tensor, *_ = self.encoders(input_tensor, pos_k, pos_v, hs_mask, mask_seqlens)
        else:
            for i, layer in enumerate(self.encoders):
                input_tensor, _, _, _ = layer(
                    input_tensor,
                    pos_k,
                    pos_v,
                    hs_mask,
                    mask_seqlens=mask_seqlens,
                    relative_attention_bias=relative_attention_bias,
                )

                if i == self.extra_layer_output_idx:
                    layer_emb = input_tensor
        if unfolded:
            embed_dim = input_tensor.shape[-1]
            input_tensor = input_tensor.reshape(ori_bz, -1, embed_dim)
            # if we ever padded before unfolding, we need to remove the padding
            if chunk_pad_size > 0:
                input_tensor = input_tensor[:, :-chunk_pad_size, :]
        return input_tensor, masks #, layer_emb

    def gradient_checkpointing_enable(self):
        pass
