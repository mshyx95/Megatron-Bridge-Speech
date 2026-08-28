# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""cascade basic blocks"""
import math
import os
import random
import numpy as np

try:
    import backoff
except ImportError:  # optional retry dependency; only used for blob/text loading, not encoder math
    from types import SimpleNamespace

    def _no_retry(*_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    backoff = SimpleNamespace(on_exception=_no_retry, expo=None)
from typing import Optional, Tuple, Union, Literal
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class Block(nn.Module):
    """Block abstract module"""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

def get_activation(name="relu"):
    """Select an activation function by name

    Args:
        name: str
            activation function name,
            one of ["relu", "gelu", "swish", "sigmoid"],
            default "relu".
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return Swish()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    return nn.Identity()

def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
    """
    The function is very important for Transformer Transducer Streaming mode
    Args:
        xs_len (int): sequence length
        chunk_start_idx (list): first idx of each chunk, such as [0,18,36,48]. It also supports adaptive chunk size [0,10,15,45]
        left_window (int): how many left chunks can be seen
        right_window (int): how many right chunks can be seen. It is used for chunk overlap model.
        Returns:
            mask (torch.Tensor): a mask tensor for streaming model
            Torch 1.0.1
            tensor([[1., 1., 0., 0.],
                    [0., 1., 1., 0.],
                    [0., 0., 1., 1.]])
            Torch 1.4.1
            tensor([[True., True., False., False.],
                    [False., True., True., False.],
                    [False., False., True., True.]])
    """
    chunk_start_idx = torch.Tensor(
        chunk_start_idx
    ).long()  # first idx of each chunk, such as [0,18,36,48].
    start_pad = torch.nn.functional.pad(
        chunk_start_idx, (1, 0)
    )  # append 0 to the beginning, so it becomes [0, 0, 18, 36, 48]
    end_pad = torch.nn.functional.pad(
        chunk_start_idx, (0, 1), value=x_len
    )  # append x_len to the end, so it becomes [0,18,36,48, x_len]
    seq_range = torch.arange(0, x_len).unsqueeze(-1)  # seq_range size: [x_len, 1]
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]  # idx size: [x_len]
    boundary = end_pad[idx]  # boundary size: [x_len]
    seq_range_expand = (
        torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)
    )  # seq_range_expand size [x_len, x_len]
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
    return mask_left & mask_right

class Swish(nn.Module):
    """Implement Swish activation module.
    From https://arxiv.org/pdf/2005.03191.pdf

    """

    def __init__(self) -> None:
        super().__init__()
        self.act_fn = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Apply Swish function

        Args:
            x: torch.Tensor
                Input.
        """
        return x * self.act_fn(x)

class GLU(nn.Module):
    """Implement Gated Linear Unit (GLU) module"""

    def __init__(self, dim: int = -1, act_name: str = "sigmoid") -> None:
        super().__init__()
        self.dim = dim
        self.act_name = act_name.lower()

        if self.act_name == "relu":
            self.act_fn = nn.ReLU(inplace=True)
        elif self.act_name == "gelu":
            self.act_fn = nn.GELU()
        elif self.act_name == "swish":
            self.act_fn = Swish()
        elif self.act_name == "sigmoid":
            self.act_fn = nn.Sigmoid()
        else:
            self.act_fn = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """GLU forward
        Apply Swish function on the first half of input matrices
        with sigmoid of the second half.

        Args:
            x: torch.Tensor
                Input.

        """
        half_x, gate = x.chunk(2, dim=self.dim)
        return half_x * self.act_fn(gate)

# TODO: Abdel, this can be improved using GLU module
class GLUPointWiseConv(nn.Module):
    """GLUPointWiseConv module
    used for conformer architecture,
    for more details see:
    https://arxiv.org/pdf/2005.08100v1.pdf

    Args:
        input_dim: int
            input channel size.
        output_dim: int
            output channel size.
        kernel_size: int
            kernel size
        glu_type: str, optional
            activation function one of
             ["sigmoid", "relu", "gelu"]
              default "sigmoid".
        bias_in_glu: bool, optional
            use addtive bias in glu
        causal: bool, optional
            if set to True, padding is set to the half of
             kernel size, ie, convolution can't see future frames.
              default False.

    """

    def __init__(
        self, input_dim, output_dim, kernel_size, glu_type="sigmoid", bias_in_glu=True, causal=False
    ):
        super().__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim
        self.bias_in_glu = bias_in_glu
        if causal:
            self.ext_pw_conv_1d = nn.Conv1d(
                input_dim, output_dim * 2, kernel_size, 1, padding=(kernel_size - 1)
            )
        else:
            self.ext_pw_conv_1d = nn.Conv1d(
                input_dim, output_dim * 2, kernel_size, 1, padding=(kernel_size - 1) // 2
            )

        if glu_type == "sigmoid":
            self.glu_act = nn.Sigmoid()
        elif glu_type == "relu":
            self.glu_act = nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = nn.GELU()
        elif glu_type == "swish":
            self.glu_act = Swish()
        else:
            raise ValueError(f"Unsupported activation type {self.glu_act}")

        if bias_in_glu:
            self.b1 = nn.Parameter(torch.zeros(1, output_dim, 1))
            self.b2 = nn.Parameter(torch.zeros(1, output_dim, 1))

    def forward(self, x):
        """
        Args:
            x: torch.Tensor
                input tensor
        """
        # to be consistent with GLULinear, we assume the input always has the #channel (#dim) in the last dimension of the tensor, so need to switch the dimension first for 1D-Conv case
        x = x.permute([0, 2, 1])
        x = self.ext_pw_conv_1d(x)
        if self.glu_type == "bilinear":
            if self.bias_in_glu:
                x = (x[:, 0 : self.output_dim, :] + self.b1) * (
                    x[:, self.output_dim : self.output_dim * 2, :] + self.b2
                )
            else:
                x = (x[:, 0 : self.output_dim, :]) * (
                    x[:, self.output_dim : self.output_dim * 2, :]
                )
        else:
            if self.bias_in_glu:
                x = (x[:, 0 : self.output_dim, :] + self.b1) * self.glu_act(
                    x[:, self.output_dim : self.output_dim * 2, :] + self.b2
                )
            else:
                x = (x[:, 0 : self.output_dim, :]) * self.glu_act(
                    x[:, self.output_dim : self.output_dim * 2, :]
                )

        x = x.permute([0, 2, 1])
        return x


class DepthWiseSeperableConv1d(nn.Module):
    """DepthWiseSeperableConv1d module used in Convnet module
    for the conformer, for more details see:
    https://arxiv.org/pdf/2005.08100v1.pdf

    Args:
        input_dim: int
            input channel size.
        depthwise_seperable_out_channel: int
            if set different to 0, the number of depthwise_seperable_out_channel
             will be used as a channel_out of the second conv1d layer.
             otherwise, it equal to 0, the second conv1d layer is skipped.
        kernel_size: int
            kernel_size
        depthwise_multiplier: int
            number of input_dim channels duplication. this value
            will be used to compute the hidden channels of the Conv1D.
        padding: int, optional
            padding for the conv1d,
             default: 0.

    """

    def __init__(
        self,
        input_dim,
        depthwise_seperable_out_channel,
        kernel_size,
        depthwise_multiplier,
        padding=0,
    ):
        super().__init__()

        self.dw_conv = nn.Conv1d(
            input_dim,
            input_dim * depthwise_multiplier,
            kernel_size,
            1,
            padding=padding,
            groups=input_dim,
        )

        if depthwise_seperable_out_channel != 0:
            self.pw_conv = nn.Conv1d(
                input_dim * depthwise_multiplier, depthwise_seperable_out_channel, 1, 1, 0
            )
        else:
            self.pw_conv = nn.Identity()
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel

    def forward(self, x):
        """

        Args:
            x: torch.Tensor
                input tensor
        """
        x = self.dw_conv(x)
        if self.depthwise_seperable_out_channel != 0:
            x = self.pw_conv(x)
        return x


class ConvModule(nn.Module):
    """ConvModule Module for the conformer block.
    for more details see:
    https://arxiv.org/pdf/2005.08100v1.pdf

    Args:
        input_dim: int
            input channel size.
        ext_pw_out_channel: int
            if > 0, ext_pw_out_channel is a dim channel size
             for the last pointwise conv after swish activation.
        depthwise_seperable_out_channel: int
            if set different to 0, the number of depthwise_seperable_out_channel
             will be used as a channel_out of the second conv1d layer.
             otherwise, it equal to 0, the second conv1d layer is skipped.
        ext_pw_kernel_size: int
            kernel size of the conv pointwise of the conformer.
        kernel_size: int
            kernel size.
        depthwise_multiplier: int
            number of input_dim channels duplication. this value
             will be used to compute the hidden channels of the Conv1D.
        dropout_rate: float
            dropout rate.
        causal: bool, optional
            if set to True, convolution have no access
             to future frames. default False.
        batch_norm: bool, optional
            if set to True, apply batchnorm before activation.
            default False
        chunk_se: int, optional
            0 for offline SE.
            1 for streaming SE, where mean is computed
             by accumulated history until current chunk_se.
            2 for streaming SE, where mean is computed
             by only the current chunk.
        chunk_size: int, optional
            chunk size for cnn. default 18
        activation: str, optional
            activation function used in ConvModule,
            default: "relu".
        glu_type: str, optional
            activation function used for the glu,
            default: "sigmoid".
        bias_in_glu: bool, optional
            if set to True, use additive bias in the weight module
             before GLU.
        linear_glu_in_convm: bool, optional
            if set to True, use GLULinear module,
             otherwise, used GLUPointWiseConv module.
              default to False.
        export: bool, optional,
            if set to True, padding is equal to 0.  This is for inference,
             or onnx export.  Typically this is set by the export program or
             the decoder program, and it isn't present in your config file.
             default False
    """

    def __init__(
        self,
        input_dim,
        ext_pw_out_channel,
        depthwise_seperable_out_channel,
        ext_pw_kernel_size,
        kernel_size,
        depthwise_multiplier,
        dropout_rate,
        causal=False,
        batch_norm=False,
        chunk_se=0,
        chunk_size=18,
        activation="relu",
        glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        export=False,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_dim = input_dim
        self.ext_pw_out_channel = ext_pw_out_channel
        self.ext_pw_kernel_size = ext_pw_kernel_size
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel
        self.glu_type = glu_type
        self.bias_in_glu = bias_in_glu
        self.linear_glu_in_convm = linear_glu_in_convm
        self.causal = causal

        self._add_ext_pw_layer()

        self.batch_norm = batch_norm
        self.kernel_size = kernel_size

        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(input_dim)

        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate)
        self.export = export

        if causal:
            if export:  # Inference only.
                padding = 0  # A cache is concatenated to the left. No padding in the kernel.
            else:
                # Training only. Padding will be added symmetrically on both sides.
                # After convolution, clip off kernel_size-1 points on the right.
                padding = kernel_size - 1
        else:
            padding = (kernel_size - 1) // 2

        self.dw_sep_conv_1d = DepthWiseSeperableConv1d(
            input_dim,
            depthwise_seperable_out_channel,
            kernel_size,
            depthwise_multiplier,
            padding=padding,
        )

        if depthwise_seperable_out_channel != 0:
            if input_dim != depthwise_seperable_out_channel:
                self.ln2 = nn.Linear(depthwise_seperable_out_channel, input_dim)
        else:
            if depthwise_multiplier != 1:
                self.ln2 = nn.Linear(input_dim * depthwise_multiplier, input_dim)

    def _add_ext_pw_layer(self):
        """
        This function is an extension of __init__ function
        and dedicated to the convolution module creation
        of the conformer.
        """
        self.ln1 = self.glu = self.bn_layer = self.ext_pw_conv_1d = nn.Identity()  # jit hacks.
        self.squeeze_excitation = nn.Identity()  # jit.
        self.apply_ln1 = self.fix_len1 = False  # jit.

        if self.ext_pw_out_channel != 0:
            if self.causal:
                self.ext_pw_conv_1d = nn.Conv1d(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    1,
                    padding=(self.ext_pw_kernel_size - 1),
                )
                if self.ext_pw_kernel_size > 1:
                    self.fix_len1 = True
                else:
                    self.fix_len1 = False
            else:
                self.ext_pw_conv_1d = nn.Conv1d(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    1,
                    padding=(self.ext_pw_kernel_size - 1) // 2,
                )
                self.fix_len1 = False

            if self.linear_glu_in_convm:
                self.glu = GLULinear(
                    self.input_dim, self.ext_pw_out_channel, self.glu_type, self.bias_in_glu
                )
            else:
                self.glu = GLUPointWiseConv(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    self.glu_type,
                    self.bias_in_glu,
                    self.causal,
                )

            if self.input_dim != self.ext_pw_out_channel:
                self.apply_ln1 = True
                self.ln1 = nn.Linear(self.ext_pw_out_channel, self.input_dim)
            else:
                self.apply_ln1 = False
        else:
            self.pw_conv_simplify_w = torch.nn.Parameter(torch.ones(3))
            self.pw_conv_simplify_b = torch.nn.Parameter(torch.zeros(3))

    def forward(self, x):
        """ConvModule Forward.

        Args:
            x: torch.Tensor
                input tensor.
        """
        x = self.layer_norm(x)

        if self.ext_pw_out_channel != 0:
            x = self.glu(x)
            if self.causal and self.ext_pw_kernel_size > 1:
                x = x[:, : -(self.ext_pw_kernel_size - 1), :]
            if self.apply_ln1:
                x = self.ln1(x)
        else:
            x_0 = x * self.pw_conv_simplify_w[0] + self.pw_conv_simplify_b[0]
            x_1 = x * self.pw_conv_simplify_w[1] + self.pw_conv_simplify_b[1]
            x = x_0 + x_1

        x = x.permute([0, 2, 1])

        x = self.dw_sep_conv_1d(x)
        if self.causal and self.kernel_size > 1:
            x = x[:, :, : -(self.kernel_size - 1)]
        if hasattr(self, "ln2"):
            x = x.permute([0, 2, 1])
            x = self.ln2(x)
            x = x.permute([0, 2, 1])
        if self.batch_norm:
            x = self.bn_layer(x)
        x = self.act(x)

        if self.ext_pw_out_channel != 0:
            x = self.ext_pw_conv_1d(x)
            if self.fix_len1:
                x = x[:, :, : -(self.ext_pw_kernel_size - 1)]

            if self.apply_ln1:
                x = x.permute([0, 2, 1])
                x = self.ln1(x)
                x = x.permute([0, 2, 1])

            x = x.permute([0, 2, 1])
        else:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
            x = x * self.pw_conv_simplify_w[2] + self.pw_conv_simplify_b[2]
            x = x.squeeze(1)

        x = self.dropout(x)
        return x

class GLULinear(nn.Module):
    """Linear + GLU module

    Args:
        input_dim: int
            input size
        output_dim: int
            output size.
        glu_type:
            activation function name used in glu module.
            default "sigmoid" (swish function).
        bias_in_glu: bool, optional
            If True, the addtive bias is added. Default False.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        glu_type="sigmoid",
        bias_in_glu=True,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2, bias_in_glu)
        self.glu_act = GLU(-1, glu_type)

    def forward(self, x):
        """GLULinear forward

        Args:
            x: torch.Tensor
                inpute tensor.
        """
        x = self.linear(x)
        return self.glu_act(x)

class FeedForward(nn.Module):
    """FeedForward Module.
    For more details see Conformer paper:
        https://arxiv.org/pdf/2005.08100.pdf

    Args:
        d_model: int
            input size.
        d_inner: int
            output size.
        dropout_rate: float,
            dropout rate.
        activation: str,
            activation function name,
            one of ["relu", "swish", "sigmoid"],
            sigmoid activation is only used with "glu_in_fnn=True",
            default "sigmoid".
        bias_in_glu: bool, optional
    """

    def __init__(
        self,
        d_model,
        d_inner,
        dropout_rate,
        activation="sigmoid",
        bias_in_glu=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner

        self.layer_norm = nn.LayerNorm(d_model)
        module = GLULinear(d_model, d_inner, activation, bias_in_glu)
        self.net = nn.Sequential(
            module,
            nn.Dropout(dropout_rate),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """FeedForward forward function.

        Args:
            x: torch.Tensor
                input tensor.
        """
        out = self.net(self.layer_norm(x))
    
        return out

#### positional encoding starts here
def _pre_hook(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)

class RelativePositionalEncoding(nn.Module):
    """Relative Positional Encoding"""

    def __init__(self, d_model, maxlen=1000, embed_v=False):
        super().__init__()

        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(2 * maxlen, d_model)
        if embed_v:
            self.pe_v = torch.nn.Embedding(2 * maxlen, d_model)
        else:
            self.pe_v = None  # For torchscript

        # inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        # self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # x_len = pos_seq.size(0)
        # sinusoid_inp = pos_seq.unsqueeze(-1) * self.inv_freq
        # pos_emb = torch.cat([sinusoid_inp.unsqueeze(-1).sin(), sinusoid_inp.unsqueeze(-1).cos()], dim=-1)
        # return pos_emb.view(x_len, x_len, self.d_model)

        pos_seq = pos_seq.masked_fill(pos_seq < -self.maxlen, -self.maxlen)
        pos_seq = pos_seq.masked_fill(pos_seq > self.maxlen - 1, self.maxlen - 1)
        # clamp is more simple use, but generate errors with onnx conversion
        # eq issue: https://github.com/microsoft/onnxruntime/issues/10399
        # pos_seq.clamp_(- self.maxlen, self.maxlen - 1)
        pos_seq = pos_seq + self.maxlen
        if self.pe_v is not None:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None

class T5RelativeAttentionLogitBias(nn.Module):
    """
    This module implements the relative position bias described in Section 2.1 of
    the T5 paper: https://arxiv.org/pdf/1910.10683.pdf

    The Huggingface implementation is used as a reference
    https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/t5/modeling_t5.py#L435

    Modifies attention as Q*K^T + B, where B is a learned scalar bias based on relative position
    of the query and key. It is HxNxN, where H is the number of heads, N is the sequence length.

    I've made these modifications to the original T5 bias:
    - Skipping of the bucketing step. Original T5 bias converted rel position distances into
      logarithmically increasing buckets. This is supposed to help with length generalization.
    - I just directly use rel position index as bias values, as we don't need length
      generalization (40s max is good enough for ASR encoder), and it keeps ONNX export simple.
    - I've also extended it so that biases can be asymmetric, the default implementation treats
      L->R and R->L the same. Asymmetric was found to yield better results in my experiments.

    Args:
        num_heads: int
            Number of attention heads
        num_buckets: int
            Number of buckets to use for relative attention bias. This is the size of the learnable
            bias parameter. Bucketing is not yet supported, so this defaults to -1 which means
            no bucketing is used (max_distance determines size of bias param).
        max_distance: int
            Maximum distance to use for relative attention bias. With num_buckets=-1, this directly
            controls the max size of the bias parameter. When num_buckets > 0 is supported, this
            will control the maximum distance for logarithmic bucketing after which all positions
            are in the same bucket.
        symmetric: bool
            Whether to use symmetric or asymmetric biases. symmetric=False uses 2x number of bias
            params to distinguish L->R from R->L. This was found to be better for the encoder.
    """

    def __init__(self, num_heads, num_buckets=-1, max_distance=1000, symmetric=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.symmetric = symmetric
        self._skip_bucketing = self.num_buckets < 0
        if self._skip_bucketing:
            self.num_buckets = max_distance
        else:
            raise NotImplementedError("T5 attention bias with bucketed positions is not yet tested")
        if not self.symmetric:
            self.num_buckets *= 2
        self.bias_values = nn.Embedding(self.num_buckets, self.num_heads)
        self._bias_index_cache = {}
        self._table_cache_key = None
        self._table_cache_value = None

    def _make_bias_idx(self, maxpos: int, device: torch.device):
        key = (maxpos, str(device))
        if key in self._bias_index_cache:
            return self._bias_index_cache[key]

        context_position = torch.arange(maxpos, device=device, dtype=torch.long)[:, None]
        memory_position = torch.arange(maxpos, device=device, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position = relative_position.masked_fill(
            relative_position < -self.max_distance, -self.max_distance
        )
        relative_position = relative_position.masked_fill(
            relative_position > self.max_distance - 1, self.max_distance - 1
        )

        if self._skip_bucketing:
            bias_idx = relative_position
        else:
            bias_idx = self._bucket_relative_position(relative_position)

        if self.symmetric:
            bias_idx = bias_idx.abs()
        else:
            bias_idx = bias_idx + self.num_buckets // 2

        self._bias_index_cache[key] = bias_idx
        return bias_idx

    def get_t5_bias_table(self, maxpos: int, device: torch.device, dtype: torch.dtype):
        """Build compact T5 bias table [H, 2*maxpos] for flash_t5bias kernel."""
        key = (
            maxpos,
            str(device),
            str(dtype),
            int(self.bias_values.weight.data_ptr()),
            int(self.bias_values.weight._version),
        )
        if not self.training and self._table_cache_key == key and self._table_cache_value is not None:
            return self._table_cache_value

        rel = torch.arange(-(maxpos - 1), maxpos, device=device, dtype=torch.long)
        rel = rel.masked_fill(rel < -self.max_distance, -self.max_distance)
        rel = rel.masked_fill(rel > self.max_distance - 1, self.max_distance - 1)

        if self._skip_bucketing:
            idx = rel
        else:
            idx = self._bucket_relative_position(rel)

        if self.symmetric:
            idx = idx.abs()
        else:
            idx = idx + self.num_buckets // 2

        table = self.bias_values(idx).transpose(0, 1).to(dtype=dtype)
        # Kernel indexing convention expects rel_pos=0 at column maxpos.
        # We keep column 0 as a guard/unused slot and shift valid entries by +1.
        out = torch.zeros(self.num_heads, 2 * maxpos, device=device, dtype=dtype)
        out[:, 1 : 1 + table.size(1)] = table

        if not self.training:
            self._table_cache_key = key
            self._table_cache_value = out
        return out

    def forward(self, x):
        # instantiate bias compatible with shape of x
        maxpos = x.size(1)
        bias_idx = self._make_bias_idx(maxpos, x.device)

        t5_rel_att_bias = self.bias_values(bias_idx)  # [L, L, H]
        t5_rel_att_bias = t5_rel_att_bias.permute(2, 0, 1).unsqueeze(0)  # [1, H, L, L]

        return t5_rel_att_bias

    def _bucket_relative_position(self, relative_position):
        # this is a placeholder (isn't tested, likely buggy) using HuggingFace implem as a reference
        # this also needs to be extended to support asymmetric +/- ve positions
        relative_buckets = 0
        if not self.causal:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

class AbsolutePositionalEncoding(nn.Module):
    """Absolute Positional encoding module.
    This module implement Absolute sinusoidal positional encoding
    from: https://arxiv.org/pdf/1706.03762.pdf

    Args:
        d_model: int
            Input embedding size.
        dropout_rate: float
            dropout rate
        max_len: int, optional
            Maximum input length sequence, Default 5000

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings.

        Args:
            x: torch.Tensor
        """
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x: torch.Tensor
                Input tensor. shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)

#### forward embedding layers starts here

@backoff.on_exception(backoff.expo, Exception, max_tries=10)
def np_loadtxt_with_retry(filepath):
    """np.loadtxt with retry

    Args:
        filepath: str
            file path to the numpy array.
    """
    if filepath.startswith("az://"):
        import blobfile as bf
        with bf.BlobFile(filepath, "rb") as f:
            result = np.loadtxt(f, dtype="f")
    else:
        result = np.loadtxt(filepath, dtype="f")
    return result

class MeanVarianceNormLayer(nn.Module):
    """Mean/variance normalization layer.

    Will substract mean and multiply input by inverted standard deviation.
    Typically used as a very first layer in a model.

    Args:
        input_size: int
            layer input size.
    """

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.register_buffer("global_mean", torch.zeros(input_size))
        self.register_buffer("global_invstd", torch.ones(input_size))
        self.global_mean: Optional[Tensor]
        self.global_invstd: Optional[Tensor]

    def forward(self, input_: Tensor) -> Tensor:
        """MeanVarianceNormLayer Forward

        Args:
            input_: torch.Tensor
                input tensor.
        """
        return (input_ - self.global_mean.to(input_.dtype)) * self.global_invstd.to(input_.dtype)

    def load_mean_invstd(self, mean_file, invstd_file, cuside_features=False):
        """Load feature mean and variance used for normalization.

        Args:
            mean_file: str
                path to the feature mean statistics file.
            invstd_file: str
                path to the features inverted standard deviation
                 statistics file.
            cuside_features: bool
                Boolean that indicates CUSIDE is being used.
                The statistics of CUSIDE features are copied
                from the normal features
        """
        self.global_mean.data = torch.from_numpy(np_loadtxt_with_retry(mean_file))
        self.global_invstd.data = torch.from_numpy(np_loadtxt_with_retry(invstd_file))

        if cuside_features:
            self.global_mean.data = torch.cat((self.global_mean.data, self.global_mean.data), 0)
            self.global_invstd.data = torch.cat(
                (self.global_invstd.data, self.global_invstd.data), 0
            )

class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would have limited access to locations on its right or left
    All arguments are the same as nn.Conv1d except padding.

    If padding is set None, then paddings are set automatically to make it a causal convolution where each location would not see any steps on its right.

    If padding is set as a list (size of 2), then padding[0] would be used as left padding and padding[1] as right padding.
    It would make it possible to control the number of steps to be accessible on the right and left.
    This mode is not supported when stride > 1. padding[0]+padding[1] should be equal to (kernel_size - 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif (
                isinstance(padding, list)
                and len(padding) == 2
                and padding[0] + padding[1] == kernel_size - 1
            ):
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, : -self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1) :]
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        else:
            return x, cache


class CausalConv2D(nn.Conv2d):
    """
    A causal version of nn.Conv2d where each location in the 2D matrix would have no access to locations on its right or down
    All arguments are the same as nn.Conv2d except padding which should be set as None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        if padding is not None:
            raise ValueError("Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1

        padding = 0
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(
        self,
        x,
    ):
        if self.training:
            x = F.pad(
                x,
                pad=(
                    self._left_padding,
                    self._right_padding,
                    self._left_padding,
                    self._right_padding,
                ),
            )
        else:
            x = F.pad(
                x,
                pad=(self._left_padding, self._right_padding, 0, 0),
            )
        x = super().forward(x)
        return x


class NemoConvSubsampling(torch.nn.Module):
    """Convlutional subsampling module, taken from NeMo ASR
    (https://github.com/NVIDIA/NeMo/blob/b367413645d5c72db3c2c96e46e95a34501479cf/nemo/collections/asr/parts/submodules/subsampling.py)

    Striding Subsampling: "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for
    Speech Recognition" by Linhao Dong et al. (https://ieeexplore.ieee.org/document/8462506)


    Compared with the EncoderConv2D (`input_layer: custom`), this is a much simplified approach,
    and uses no LayerNorm and far fewer Conv2Ds.  Moreover, depthwise convolutions are used to reduce
    FLOPs, but the first layer is kept as a regular convolution so as not to degrade accuracy.

    `Striding` and `dw_striding` are the same except that the latter uses depthwise convolutions
    after the first layer, whereas the former does not.

    Args:
        subsampling_factor (int): Time reduction factor
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        subsampling (str): The subsampling technique, choose from
            {"striding", "dw-striding", "striding_conv1d", "dw_striding_conv1d"}
        conv_channels (int): Number of channels for the convolution layers, default is 256.
        subsampling_conv_chunking_factor (int): Input chunking factor which can be -1 (no chunking)
            1 (auto) or a power of 2. Default is 1
        activation (Module): activation function, default is nn.ReLU()
        is_causal (bool): whether to use causal Conv1/2D, where each step will have limited access
            to locations on its right or left
    """

    def __init__(
        self,
        feat_in,
        feat_out,
        subsampling_factor=4,
        subsampling="dw_striding",
        conv_channels=256,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        super().__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal
        self.subsampling_causal_cond = subsampling in ("dw_striding", "striding", "striding_conv1d")

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        if subsampling == "dw_striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            # Layer 1
            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    )
                )
            else:
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            groups=in_channels,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        )
                    )

                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "striding_conv1d":
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv1D(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "dw_striding_conv1d":
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2

            # Layer 1
            layers.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        groups=in_channels,
                    ),
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=feat_out if self._sampling_num == 1 else conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    ),
                ]
            )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                layers.extend(
                    [
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        ),
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 2 else conv_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                        ),
                    ]
                )
                layers.append(activation)
                in_channels = conv_channels

        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        if subsampling in ["dw_striding", "striding"]:
            # Compute output length as pure Python to avoid meta-tensor .item() errors
            out_length = float(feat_in)
            add_pad = float((self._left_padding + self._right_padding) - self._kernel_size)
            for _ in range(self._sampling_num):
                out_length = (out_length + add_pad) / self._stride + 1.0
                if self._ceil_mode:
                    out_length = float(math.ceil(out_length))
                else:
                    out_length = float(math.floor(out_length))
            self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
            self.conv2d_subsampling = True
        elif subsampling in ["striding_conv1d", "dw_striding_conv1d"]:
            self.out = None
            self.conv2d_subsampling = False
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        self.conv = torch.nn.Sequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def forward(self, x, mask):
        """
        Forward method for NeMo subsampling.

        Args:
            x[Batch, Time, Filters]: torch.Tensor
                input tensor
            x_mask: torch.Tensor
                input mask

        Returns:
            x: torch.Tensor
                Resulting tensor from subsampling (B, T // time_reduction_factor, feat_out)
            pad_mask: torch.Tensor
                tensor of padded hidden state sequences (B, 1, T // time_reduction_factor)
        """
        # Unsqueeze Channel Axis
        if self.conv2d_subsampling:
            x = x.unsqueeze(1)
        # Transpose to Channel First mode
        else:
            x = x.transpose(1, 2)

        # split inputs if chunking_factor is set
        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:
                # if subsampling_conv_chunking_factor is 1, we split only if needed
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                x_ceil = 2**31 / (self._conv_channels * self._stride * self._stride)
                if torch.numel(x) > x_ceil:
                    need_to_split = True
                else:
                    need_to_split = False
            else:
                # if subsampling_conv_chunking_factor > 1 we always split
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:  # if unable to split by batch, try by channel
                    if self._subsampling == "dw_striding":
                        x = self.conv_split_by_channel(x)
                    else:
                        x = self.conv(x)  # try anyway
            else:
                x = self.conv(x)
        else:
            x = self.conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)

        if mask is None:
            return x, None

        max_audio_length = x.shape[1]
        feature_lens = mask.sum(1)
        padding_length = torch.ceil(feature_lens / self.subsampling_factor)
        if self.is_causal and self.subsampling_causal_cond:
            feature_lens_remainder = feature_lens % self.subsampling_factor
            padding_length[feature_lens_remainder != 1] += 1
        pad_mask = (
            torch.arange(0, max_audio_length, device=x.device).expand(padding_length.size(0), -1)
            < padding_length.unsqueeze(1)
        )
        return x, pad_mask.unsqueeze(1)

    def reset_parameters(self):
        # initialize weights
        if self._subsampling == "dw_striding":
            with torch.no_grad():
                # init conv
                scale = 1.0 / self._kernel_size
                dw_max = (self._kernel_size**2) ** -0.5
                pw_max = self._conv_channels**-0.5

                torch.nn.init.uniform_(self.conv[0].weight, -scale, scale)
                torch.nn.init.uniform_(self.conv[0].bias, -scale, scale)

                for idx in range(2, len(self.conv), 3):
                    torch.nn.init.uniform_(self.conv[idx].weight, -dw_max, dw_max)
                    torch.nn.init.uniform_(self.conv[idx].bias, -dw_max, dw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].weight, -pw_max, pw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].bias, -pw_max, pw_max)

                # init fc (80 * 64 = 5120 from https://github.com/kssteven418/Squeezeformer/blob/13c97d6cf92f2844d2cb3142b4c5bfa9ad1a8951/src/models/conformer_encoder.py#L487
                fc_scale = (self._feat_out * self._feat_in / self._sampling_num) ** -0.5
                torch.nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
                torch.nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)

    def conv_split_by_batch(self, x):
        """Tries to split input by batch, run conv and concat results"""
        b, _, _, _ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2**31 / (self._conv_channels * self._stride * self._stride)
            p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
            cf = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:  # input is too big
            return x, False

        return torch.cat([self.conv(chunk) for chunk in torch.split(x, new_batch_size, 0)]), True

    def conv_split_by_channel(self, x):
        """For dw convs, tries to split input by time, run conv and concat results"""
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
            else:
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                p = math.ceil(math.log(torch.numel(x) / 2**31, 2))
                cf = 2**p

            new_c = int(c // cf)
            if new_c == 0:
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                new_t = 1

            x = self.channel_chunked_conv(self.conv[i * 3 + 2], new_c, x)  # conv2D, depthwise

            # splitting pointwise convs by time
            x = torch.cat(
                [self.conv[i * 3 + 3](chunk) for chunk in torch.split(x, new_t, 2)], 2
            )  # conv2D, pointwise
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(
                    chunk,
                    pad=(
                        self._kernel_size - 1,
                        self._stride - 1,
                        self._kernel_size - 1,
                        self._stride - 1,
                    ),
                )
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)

    def change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int):
        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

def make_conv_layers(idim, kernels, strides, paddings, channels, extra_pad: str = "none"):
    """
    make convolutional layers

    Args:
        idim (int): feat dimension
        kernels (List): list of kernel sizes of convolutional layers
        strides (List): list of stride sizes of convolutional layers
        paddings (List): list of paddings of convolutional layers
        channels (int): output channels
        extra_pad (str): which dimension to pad

    Returns:
        convolution layers
        intermediate dimension
    """
    ConvCls = Conv2dPadRight if extra_pad == "feat_time" else torch.nn.Conv2d
    layers = []
    for i, (k, s, p) in enumerate(zip(kernels, strides, paddings)):
        if extra_pad == "feat_time":
            p_feat = s - 1  # Conv2dPadRight add s-1 paddings
        elif extra_pad == "feat":
            p = (p, s - 1)
            p_feat = (s - 1) * 2  # torch.nn.Conv2d add (s-1)*2 paddings
        else:
            p_feat = p * 2
        in_channels = 1 if i == 0 else channels
        layers.append(ConvCls(in_channels, channels, k, s, p))
        layers.append(torch.nn.ReLU())
        idim = (idim - k + p_feat) // s + 1
    return torch.nn.Sequential(*layers), idim
    
class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length or 1/8 length).

    Args:
        idim: int
            input size.
        odim: int
            output size.
        kernel_size: list, optional
            kernel size for the 2 stacked cnn layer,
             default [3, 3].
        stride: list, optional
            stride for the 2 stacked cnn layer,
             default [2, 2].
        padding: list, optional
             padding for the 2 stacked cnn layer,
              default [0, 0].
        cnn_out: int, optional
            the number of channels. the number of channel impacts model
             training speed, model size and training stabilization.
              A better way is to use larger Transformer models,
               and keeps the number of CNN channels < 512. default -1.
        layer_norm: bool, optional
            if set to True, use LayerNorm after convolution.
             default False.
        whether or not use layer norm between cnn and transformer
        time_reduction: int, optional
            Time reduction factor, 4 and 8 supported
        extra_padding: str, optional
            use Conv2Right if "feat_time".
            Otherwise use Conv2dsubsampling default CNN stride.
            if set to True, use extra_padding on default Conv2dsubsampling.
        intermediate_dim: int, optional
            the intermediate dimension after convolution layers
    """

    def __init__(
        self,
        idim,
        odim,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[0, 0],
        cnn_out=-1,
        layer_norm=False,
        time_reduction=4,
        extra_padding: Literal["feat", "feat_time", "none", True] = "none",
        intermediate_dim=0,
    ):
        super().__init__()

        assert extra_padding in ("feat", "feat_time", "none", True)
        self._extra_padding = extra_padding == True or extra_padding == "feat_time"
        self._pad_in_training = extra_padding == True
        assert time_reduction in (4, 6, 8), "Time reduction must one of (4, 6, 8)."

        self.time_reduction = time_reduction
        self.time_reduction = time_reduction

        if cnn_out <= 0:
            cnn_out = odim

        if time_reduction == 4:
            self.offset = 3
        elif time_reduction == 6:
            stride = [stride[0], 3]
            self.offset = 1
        elif time_reduction == 8:
            kernel_size = [kernel_size[0], kernel_size[1], kernel_size[1]]
            stride = [stride[0], stride[1], stride[1]]
            padding = [padding[0], padding[1], padding[1]]
            self.offset = 7

        self.conv, idim_out = make_conv_layers(
            idim, kernel_size, stride, padding, cnn_out, extra_padding
        )

        if intermediate_dim == 0:
            intermediate_dim = idim_out
        self.out = torch.nn.Linear(cnn_out * intermediate_dim, odim)
        self.norm = None
        if layer_norm:
            self.norm = torch.nn.LayerNorm(
                odim, elementwise_affine=False
            )  # The layer norm aims to stablize training. elementwise_affine = False is important. Setting it to true also causes NaN issue
        # The following are for eager quantization in Torchscript.  If
        # you are not doing that they become NOOPs and don't affect performance.
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.register_buffer(
            "cmvn",
            torch.zeros(idim),
            persistent=False,
        )

    def compute_length_change(self, length):
        """Compute the length change after convolution.

        Assumes kernel size is 3 and stride is either 2 or 3.
        """
        if self.time_reduction == 4:
            seq_lens = (length - 3) / 4
        elif self.time_reduction == 6:
            seq_lens = (length - 1) / 6
        else:
            assert self.time_reduction == 8, "Bad time reduction"
            seq_lens = (length - 7) / 8
        # get upper value for all non int (non supraframe value)
        if self._extra_padding and (self._pad_in_training or self.training):
            if isinstance(seq_lens, torch.Tensor):
                # tensor [Batch]: feature_lens case.
                seq_lens = seq_lens.ceil_().long()
            else:
                # int case: feature.shape[1]
                seq_lens = math.ceil(seq_lens)
        else:
            if isinstance(seq_lens, torch.Tensor):
                # tensor [Batch]: feature_lens case.
                seq_lens = seq_lens.floor_().long()
            else:
                # int case: feature.shape[1]
                seq_lens = math.floor(seq_lens)
        return seq_lens

    def forward(self, x, ilens):
        """Forward Subsample.

        Args:
            x[Batch, Time, Filters]: torch.Tensor
                input tensor
            ilens: torch.Tensor
                input length [Batch]

        Returns:
            x[Batch, Time, Channels]: torch.Tensor
                subsampled input.
            mask: torch.Tensor
                mask for next input.
        """
        # ilens and subsampling_ilens compute
        assert len(x.shape) == 3, "input frames must be 3D tensor."
        if ilens is not None:
            mask_length = self.compute_length_change(torch.tensor(x.size(1)))
            subsampling_ilens = self.compute_length_change(ilens)
            # need [B,1] dimension for mask building.
            subsampling_ilens.unsqueeze_(1)
        # fix longest utt in batch if needed.
        # this is only possible
        if self._extra_padding and self._pad_in_training:
            pad_missing = (self.time_reduction - (ilens.max() - self.offset)) % self.time_reduction
            # if there is at least 1utt having a pad_missing in a batch ==> fix it.
            if pad_missing != 0:
                # If so, then extend input frames after CMVN by addition pad_missing -mean/invstd.
                x = torch.cat((x, self.cmvn.expand(x.size(0), pad_missing, -1)), dim=1)

        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        if self.norm is not None:
            x = self.norm(x)

        if ilens is None:
            return x, None
        # handle padding
        x_mask = (
            torch.arange(mask_length, device=ilens.device).expand(subsampling_ilens.size(0), -1)
            < subsampling_ilens
        ).unsqueeze(1)
        if x_mask.shape[1] == 1:
            return x, x_mask
        raise ValueError("Doesn't support 2D mask reduction yet.")

    def streaming_forward(self, xs, pad_left: int = 1, pad_right: int = 1):
        """
        Streaming interface.  Cached-state handling done above.
        """
        # add channel dim
        if self._extra_padding:
            # No mask in streaming mode,
            # handle only longest utt in batch.
            pad_missing = (self.time_reduction - (xs.shape[1] - self.offset)) % self.time_reduction
            # if pad>0. otherwise no pad will be added.
            xs = torch.cat((xs, self.cmvn.expand(xs.size(0), pad_missing, -1)), dim=1)

        x = xs.unsqueeze(1)

        x = self.conv(x)

        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if self.norm is not None:
            x = self.norm(x)
        return x

    def batched_streaming_forward(
        self, x: Tensor, x_lens: Tensor, end_utterance: Tensor, cache: Tensor
    ):
        """
        Args:
           x[Batch, Time, Channels]
           x_lens[Batch]
           end_utterance[Batch]
           cache[Batch, Time=2, Channels] --- saved state on the left. (Causal)
        """
        x = torch.cat((cache, x), dim=1)

        index = (
            x_lens[:, None, None]
            + torch.arange(self.time_reduction - 1, device=x.device)[None, :, None]
        )
        cache = torch.gather(x, dim=1, index=index.expand(-1, -1, x.shape[2]))

        x = self.quant(x)
        x = x.unsqueeze(1)  # Create a vertical axis for the 2d convolution
        x = self.conv(x)

        b, t = x.shape[0], x.shape[2]
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, -1))
        if self.norm is not None:
            x = self.norm(x)
        x = self.dequant(x)

        cache.masked_fill_(end_utterance[:, None, None], 0.0)

        return x, cache

def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)

####  multihead attention starts here
class AttModule(nn.Module):
    """Attention abstraction module"""

    def __init__(self):
        super().__init__()
        self.export_mode = False

    def set_export(self, mode=True):
        """set the export mode"""
        self.export_mode = mode

    def forward(
        self,
        x: Tensor,
        memory: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        att_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """AttModule forward

        Args:
            x: torch.Tensor
                input tensor.
            memory: torch.Tensor, optional
                memory tensor.
            pos_emb: torch.Tensor, optional
                positional encoder embedding.
            att_mask: torch.Tensor, optional
                attention mask tensor.
        """
        return x, memory, pos_emb, att_mask


class AttBlock(Block, AttModule):
    """Attention Block module to support both Attention and Block module."""

    def memory_dims(self, max_len=False):
        """memory dimensions"""
        return (1, self.input_size)

def masked_softmax(
    scores,
    mask: Optional[Tensor],
):
    if mask is not None:
        mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
        scores = scores.masked_fill(mask, -torch.inf)
        attn = torch.softmax(scores.float(), dim=-1).masked_fill(mask, 0.0).to(scores.dtype)  # (batch, head, time1, time2)
    else:
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)  # (batch, head, time1, time2)
    return attn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer with optional relative position embedding and GLU.

    Args:
        n_head: int
            the number of heads.
        n_feat: int
            input size features.
        dropout_rate: float
            dropout rate.
        use_LN: bool
            apply layer norm or not
        dropout_at_output: bool
            whether to apply dropout at output
        attention_inner_dim: int, optional
            the attention dimension used in the class,
            it can be different from the input dimension n_feat.
            default: -1 (equal to n_feat).
        use_pt_scaled_dot_product_attention: bool, optional
            if set True, use pytorch scaled dot product attention in training.  NOTE: this will NOT
            be used in ONNX decoding due to a lack of support.  In that case, we use the original
            attention implementation, which shows no regression.
            default: False.
        n_value: int, optional
            if set to values other than -1, use a different dimension for value. With the default value (i.e. -1), it is backward compatible.
        group_size: int, optional. must divide `n_head`
            if group_size > 1:       GQA
            if group_size = 1:       MHA
            if group_size = n_head:  MQA
        use_flashattention_t5bias: bool, optional
            if True, use custom FlashAttention with T5 relative bias kernel.
            default: False.
    """

    inv_sqrt_d_k: torch.jit.Final[float]
    h: torch.jit.Final[int]
    h_k: torch.jit.Final[int]
    g: torch.jit.Final[int]

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        attention_inner_dim=-1,
        glu_type="swish",
        bias_in_glu=True,
        use_pt_scaled_dot_product_attention=False,
        n_value=-1,
        group_size: int = 1,
        use_flashattention_t5bias: bool = False,
    ):
        super().__init__()
        if n_value == -1:
            n_value = n_feat
        if attention_inner_dim == -1:
            attention_inner_dim = n_feat
        assert attention_inner_dim % n_head == 0

        # We assume d_v always equals d_k
        self.d_k = attention_inner_dim // n_head
        self.inv_sqrt_d_k = 1.0 / math.sqrt(self.d_k)
        self.h = n_head
        assert n_head % group_size == 0, "group_size must divide n_head"
        self.g = group_size
        self.h_k = n_head // group_size
        
        self.linear_q = nn.Linear(n_feat, attention_inner_dim)
        self.linear_k = nn.Linear(n_feat, attention_inner_dim // group_size)
        self.linear_v = nn.Linear(n_value, attention_inner_dim // group_size)
        self.linear_out = nn.Linear(attention_inner_dim // group_size, n_value)
        
        self.attn = torch.jit.Attribute(None, Optional[Tensor])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate
        self.use_pt_scaled_dot_product_attention = use_pt_scaled_dot_product_attention
        self.use_flashattention_t5bias = use_flashattention_t5bias

        if use_pt_scaled_dot_product_attention and group_size > 1:
            raise ValueError("Cannot use PT Scaled Attention with GQA")

        if use_flashattention_t5bias and group_size > 1:
            raise ValueError("Cannot use FlashAttention T5 bias with GQA")

        # Torchscript eager quantization.  Note that these functions below are
        # NOOPs and have very little impact on performance unless quantization is
        # enabled.
        self.quant_q = torch.ao.quantization.QuantStub()
        self.quant_x = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.ffunc = torch.ao.nn.quantized.FloatFunctional()
        
        # For tracking which kernel is used (logging/debugging)
        self._kernel_used = "manual"
        self._flash_t5bias_success_logged = False
        self._flash_t5bias_mask_fallback_logged = False
        self._t5_bias_cache_key = None
        self._t5_bias_cache_table = None
        self._require_t5_no_fallback = os.getenv("FLASH_T5BIAS_REQUIRE_NO_FALLBACK", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _raise_if_t5_no_fallback_required(self, reason: str):
        if self.use_flashattention_t5bias and self._require_t5_no_fallback:
            raise RuntimeError(f"FLASH_T5BIAS_REQUIRE_NO_FALLBACK=1 but fallback was required: {reason}")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_k: Tensor,
        pos_v: Tensor,
        mask: Optional[Tensor],
        mask_seqlens: Optional[Tuple[Tensor, Tensor, bool]] = None,
        relative_attention_bias: Optional[Tensor] = None,
    ):
        """Compute 'Scaled Dot Product Attention'.

        Args:
            query: torch.Tensor
                query tensor (batch, time1, size)
            key: torch.Tensor
                key tensor (batch, time2, size)
            value: torch.Tensor
                value tensor (batch, time1, size)
            pos_k: torch.Tensor
                key tensor used for relative positional embedding.
            pos_v: torch.Tensor
                value tensor used for relative positional embedding.
            mask: torch.Tensor
                mask tensor (batch, time1, time2)
            relative_attention_bias: torch.Tensor
                bias added to attention logits w.r.t. relative positions (1, n_head, time1, time2)
        """
        # Priority 1: Try to use FlashAttention T5 bias kernel if explicitly enabled.
        # Supports no-mask and prefix-padding mask (via varlen cu_seqlens path).
        #
        # NOTE: The FlashAttention T5-bias path does not include the additional relative-position
        # contributions from `pos_k` / `pos_v` used by the manual path. If those tensors are
        # provided, route through the manual implementation for correctness.
        if (
            self.use_flashattention_t5bias
            and relative_attention_bias is not None
            and pos_k is None
            and pos_v is None
            and not torch.jit.is_scripting()
            and self.h == self.h_k  # Only for standard MHA
        ):
            try:
                x = self._forward_with_flash_attn_t5bias(
                    query,
                    key,
                    value,
                    pos_k,
                    pos_v,
                    mask,
                    mask_seqlens,
                    relative_attention_bias,
                )
                if self._kernel_used == "flash_attn_t5bias_cuda" and not self._flash_t5bias_success_logged:
                    print("Info: Using flash_attn_t5bias_cuda kernel")
                    self._flash_t5bias_success_logged = True
                return x
            except Exception as e:
                self._raise_if_t5_no_fallback_required(f"kernel execution failed: {e}")
                # Fall back to manual if FA2 fails
                print(f"Warning: FlashAttention T5 bias failed: {e}, falling back to manual")
                self._kernel_used = "manual"
                # Continue to fallback logic below
        
        # Priority 2: Use PyTorch scaled_dot_product_attention if enabled (default False)
        # or Priority 3: Use original manual attention implementation
        self._kernel_used = "manual"
        return self._forward_manual(
            query, key, value, pos_k, pos_v, mask, relative_attention_bias
        )

    def _forward_with_flash_attn_t5bias(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_k: Tensor,
        pos_v: Tensor,
        mask: Optional[Tensor],
        mask_seqlens: Optional[Tuple[Tensor, Tensor, bool]],
        relative_attention_bias: Tensor,
    ):
        """Forward pass using FlashAttention with T5 relative bias.
        
        Args:
            relative_attention_bias: [1, n_head, time1, time2] T5 bias tensor
        
        Returns:
            output: [batch, time1, n_feat]
        """
        try:
            import flash_attn_t5bias_cuda
        except ImportError:
            raise ImportError("flash_attn_t5bias_cuda not available. Ensure it's installed.")
        
        n_batch = query.size(0)
        
        # Project q, k, v in the shape expected by flash_attn_t5bias_cuda:
        # (batch, seqlen, num_heads, head_dim)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)  # (b, t, h, d)
        k = self.linear_k(key).view(n_batch, -1, self.h_k, self.d_k)  # (b, t, h, d)
        v = self.linear_v(value).view(n_batch, -1, self.h_k, self.d_k)  # (b, t, h, d)
        
        # Extract/cached T5 bias table [h, 2*max_dist] from relative_attention_bias [1, h, t1, t2].
        t5_bias_table = self._get_t5_bias_table_cached(relative_attention_bias)
        
        b, t, h, d = q.shape
        
        # Call FlashAttention with T5 bias
        # Signature: fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, p_dropout, softmax_scale, 
        #                is_causal, window_size_left, window_size_right, return_softmax, gen, t5_bias_table)
        softmax_scale = self.inv_sqrt_d_k
        p_dropout = self.dropout_rate if self.training else 0.0
        
        try:
            # ------------------------------------------------------------------
            # Decide whether to use varlen (packed) path.
            # Dense flash_attn_t5bias_cuda.fwd here does not consume an explicit
            # padding mask tensor, so masked correctness must come from either
            # varlen seqlens or manual fallback.
            # ------------------------------------------------------------------
            q_lens = None
            k_lens = None
            is_causal_mask = False
            mask_from_tensor = False

            # If full mask tensor is present, treat it as source of truth for
            # q/k valid lengths to avoid semantic mismatch from caller-provided
            # mask_seqlens (e.g., q_lens compressed while mask keeps all queries valid).
            if mask is not None:
                _q, _k, _c = self._analyze_prefix_mask(mask, t, k.size(1))
                if _q is not None and _k is not None:
                    q_lens, k_lens, is_causal_mask = _q, _k, _c
                    mask_from_tensor = True

            if mask_seqlens is not None:
                # Caller supplied lengths directly (fast path, no mask analysis).
                if not mask_from_tensor:
                    if len(mask_seqlens) == 3:
                        q_lens, k_lens, is_causal_mask = mask_seqlens
                    elif len(mask_seqlens) == 2:
                        q_lens, k_lens = mask_seqlens
                    else:
                        raise ValueError("mask_seqlens must be (q_lens, k_lens) or (q_lens, k_lens, is_causal)")

                    q_lens = q_lens.to(device=q.device, dtype=torch.int32)
                    k_lens = k_lens.to(device=q.device, dtype=torch.int32)

            elif mask_seqlens is None and mask is not None:
                # No explicit lengths, but mask provided — try to extract them.
                use_masked = not bool(torch.all(mask).item())
                if use_masked:
                    q_lens, k_lens, is_causal_mask = self._analyze_prefix_mask(mask, t, k.size(1))

            has_real_padding = False
            if q_lens is not None and k_lens is not None:
                all_full_q = bool(torch.all(q_lens == t).item())
                all_full_k = bool(torch.all(k_lens == k.size(1)).item())
                has_real_padding = not (all_full_q and all_full_k)
                if not has_real_padding:
                    q_lens, k_lens = None, None

            if q_lens is not None and k_lens is not None:
                output = self._run_flash_attn_t5bias_masked_varlen(
                    flash_attn_t5bias_cuda,
                    q, k, v,
                    q_lens, k_lens,
                    p_dropout, softmax_scale,
                    t5_bias_table, is_causal_mask,
                )
            else:
                # Dense path: full-length attention with optional mask applied inside kernel.
                # For small T or non-prefix masks, this is faster than pack/unpack.
                use_masked = mask is not None and not bool(torch.all(mask).item())
                if use_masked and mask_seqlens is None:
                    # Try analyze in case caller forgot to set mask_seqlens but has prefix mask.
                    _q, _k, _c = self._analyze_prefix_mask(mask, t, k.size(1))
                    if _q is not None:
                        output = self._run_flash_attn_t5bias_masked_varlen(
                            flash_attn_t5bias_cuda, q, k, v,
                            _q, _k, p_dropout, softmax_scale, t5_bias_table, _c,
                        )
                    else:
                        raise RuntimeError(
                            "Mask pattern is not prefix-padding-compatible"
                        )
                else:
                    _has_fwd_infer = hasattr(flash_attn_t5bias_cuda, "fwd_infer")
                    if _has_fwd_infer and p_dropout == 0.0:
                        # Inference-only path: kernel only returns {out}, no saved q/k/v.
                        _out_c = c10_optional_none = None  # typed alias
                        _alibi = None
                        result = flash_attn_t5bias_cuda.fwd_infer(
                            q, k, v,
                            _out_c, _alibi,
                            softmax_scale,
                            False, -1, -1,
                            t5_bias_table,
                        )
                        output = result[0]
                        del result, q, k, v
                    else:
                        result = flash_attn_t5bias_cuda.fwd(
                            q, k, v,
                            None, None,
                            p_dropout, softmax_scale,
                            False, -1, -1, False, None,
                            t5_bias_table,
                        )
                        output = result[0]
                        del result, q, k, v
                    if output.dim() != 4:
                        raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
                    if output.shape[1] == t and output.shape[2] == h:
                        output = output.contiguous()
                    elif output.shape[1] == h and output.shape[2] == t:
                        output = output.transpose(1, 2).contiguous() if h != t else output.contiguous()
                    else:
                        raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
        except Exception as e:
            raise RuntimeError(f"FlashAttention forward failed: {e}")

        # Dropout is applied inside the FlashAttention kernel via p_dropout
        # (on attention weights), matching the manual path behavior.
        # Do not apply a second output dropout here.
        self._kernel_used = "flash_attn_t5bias_cuda"

        # Project output
        x = output.reshape(n_batch, -1, self.h_k * self.d_k)
        x = self.linear_out(x)
        
        return x

    def _analyze_prefix_mask(self, mask: Tensor, t_q: int, t_k: int):
        """Analyze mask as prefix-padding and optional causal pattern.

        Returns:
            (q_lens, k_lens, is_causal_mask)
            or (None, None, False) when mask is not prefix-padding-compatible.
        """
        m = mask
        if m.dim() == 4 and m.size(1) == 1:
            m = m.squeeze(1)
        if m.dim() != 3:
            return None, None, False

        if m.dtype != torch.bool:
            m = m != 0

        if m.size(1) != t_q or m.size(2) != t_k:
            return None, None, False

        key_valid = m.any(dim=1)  # [b, t_k]
        query_valid = m.any(dim=2)  # [b, t_q]
        k_lens = key_valid.sum(dim=1, dtype=torch.int32)
        q_lens = query_valid.sum(dim=1, dtype=torch.int32)

        idx_k = torch.arange(t_k, device=m.device, dtype=torch.int32).unsqueeze(0)
        idx_q = torch.arange(t_q, device=m.device, dtype=torch.int32).unsqueeze(0)
        key_prefix = idx_k < k_lens.unsqueeze(1)
        query_prefix = idx_q < q_lens.unsqueeze(1)

        if not bool(torch.all(key_valid == key_prefix).item()):
            return None, None, False
        if not bool(torch.all(query_valid == query_prefix).item()):
            return None, None, False

        # Detect whether the mask inside the valid prefix area is causal.
        # Use a fully vectorized check to avoid per-sample Python loops.
        is_causal_mask = False
        if t_q == t_k and bool(torch.all(q_lens == k_lens).item()):
            causal_template = (
                torch.arange(t_k, device=m.device).view(1, t_k)
                <= torch.arange(t_q, device=m.device).view(t_q, 1)
            )  # [t, t]
            expected = causal_template.unsqueeze(0) & query_prefix.unsqueeze(2) & key_prefix.unsqueeze(1)
            is_causal_mask = bool(torch.equal(m, expected))

        return q_lens, k_lens, is_causal_mask

    def _run_flash_attn_t5bias_masked_varlen(
        self,
        flash_attn_t5bias_cuda,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_lens: Tensor,
        k_lens: Tensor,
        p_dropout: float,
        softmax_scale: float,
        t5_bias_table: Tensor,
        is_causal_mask: bool,
    ) -> Tensor:
        """Run custom T5-bias kernel via varlen packed API.

        Mirrors FA2 bert_padding.unpad_input / pad_input exactly:
          packed = x.reshape(B*T, H, D)[flat_indices]   # one GPU gather kernel
          out[flat_indices] = packed_out                 # one GPU scatter kernel
        No Python loops, no .tolist().  Only 4 unavoidable .item() syncs
        (same count as FA2's unpad_input: max_seqlen + total_tokens).
        """
        bsz, t_q, h, d = q.shape
        t_k = k.size(1)
        dev = q.device

        q_lens = q_lens.to(device=dev, dtype=torch.int32)
        k_lens = k_lens.to(device=dev, dtype=torch.int32)

        # cu_seqlens: two cumsum kernels, fully async
        cu_q = F.pad(q_lens.cumsum(0, dtype=torch.int32), (1, 0))  # [B+1]
        cu_k = F.pad(k_lens.cumsum(0, dtype=torch.int32), (1, 0))  # [B+1]

        # 4 unavoidable host syncs – same as FA2's unpad_input
        max_seqlen_q = int(q_lens.max())
        max_seqlen_k = int(k_lens.max())
        total_q      = int(cu_q[-1])
        total_k      = int(cu_k[-1])

        if total_q == 0 or total_k == 0:
            return q.new_zeros(bsz, t_q, h, d)

        # --- Pack: FA2 bert_padding.unpad_input pattern ---
        # nonzero on [B, T] bool  =>  flat indices of valid tokens, sorted by (batch, pos).
        # Two GPU kernels + one implicit sync (variable-length output) – same as FA2.
        q_idx = (
            (torch.arange(t_q, device=dev).unsqueeze(0) < q_lens.unsqueeze(1))
            .flatten().nonzero(as_tuple=False).squeeze(1)
        )  # [total_q], int64
        k_idx = (
            (torch.arange(t_k, device=dev).unsqueeze(0) < k_lens.unsqueeze(1))
            .flatten().nonzero(as_tuple=False).squeeze(1)
        )  # [total_k], int64

        # Single gather kernel per tensor – same as FA2's index_first_axis
        q_pack = q.reshape(bsz * t_q, h, d)[q_idx]
        k_pack = k.reshape(bsz * t_k, h, d)[k_idx]
        v_pack = v.reshape(bsz * t_k, h, d)[k_idx]

        # --- Attention kernel ---
        _has_varlen_infer = hasattr(flash_attn_t5bias_cuda, "varlen_fwd_infer")
        if _has_varlen_infer and p_dropout == 0.0:
            _none = None
            result = flash_attn_t5bias_cuda.varlen_fwd_infer(
                q_pack, k_pack, v_pack,
                _none,
                cu_q, cu_k,
                _none, _none,
                max_seqlen_q, max_seqlen_k,
                softmax_scale,
                bool(is_causal_mask),
                -1, -1,
                t5_bias_table,
            )
        else:
            result = flash_attn_t5bias_cuda.fwd_t5_auto(
                q_pack, k_pack, v_pack,
                cu_q, cu_k,
                None, None, None,
                max_seqlen_q, max_seqlen_k,
                p_dropout, softmax_scale,
                False, bool(is_causal_mask),
                -1, -1, False, None,
                t5_bias_table,
            )
        out_pack = result[0]
        del result, q_pack, k_pack, v_pack
        if out_pack.dim() != 3:
            raise RuntimeError(f"Unexpected varlen output shape: {tuple(out_pack.shape)}")

        # --- Unpack: FA2 bert_padding.pad_input pattern ---
        # out[flat_indices] = packed_out  =>  single scatter kernel
        out = q.new_zeros(bsz * t_q, h, d)
        out[q_idx] = out_pack
        return out.reshape(bsz, t_q, h, d)

    def _extract_t5_bias_table(self, relative_attention_bias: Tensor):
        """Extract T5 bias table from relative_attention_bias tensor.
        
        Args:
            relative_attention_bias: [1, h, time1, time2] tensor
            
        Returns:
            t5_bias_table: [h, 2*max_dist] tensor with bias values for relative distances
        """
        # Compact table path: [h, 2*max_dist] (or close to this width).
        if relative_attention_bias.dim() == 2:
            return relative_attention_bias

        # relative_attention_bias has shape [1, h, t1, t2]
        # Extract bias values based on relative position indices
        # For a position pair (i, j), relative position is (j - i)
        # Bias should be indexed by relative position
        
        _, h, t1, t2 = relative_attention_bias.shape
        max_dist = max(t1, t2)

        t5_bias_table = torch.zeros(
            h,
            2 * max_dist,
            dtype=relative_attention_bias.dtype,
            device=relative_attention_bias.device,
        )

        # Vectorized fill by relative position without Python loop.
        rel = torch.arange(-(t1 - 1), t2, device=relative_attention_bias.device, dtype=torch.long)
        i = torch.clamp(-rel, min=0)
        j = i + rel
        # Compact table keeps column 0 as guard; rel_pos=0 maps to column max_dist.
        table_idx = rel + max_dist
        t5_bias_table[:, table_idx] = relative_attention_bias[0, :, i, j]

        return t5_bias_table

    def _get_t5_bias_table_cached(self, relative_attention_bias: Tensor) -> Tensor:
        """Return cached extracted T5 bias table when input tensor is unchanged."""
        key = (
            int(relative_attention_bias.data_ptr()),
            tuple(relative_attention_bias.shape),
            str(relative_attention_bias.dtype),
            str(relative_attention_bias.device),
            int(relative_attention_bias._version) if hasattr(relative_attention_bias, "_version") else -1,
        )
        if self._t5_bias_cache_key == key and self._t5_bias_cache_table is not None:
            return self._t5_bias_cache_table

        table = self._extract_t5_bias_table(relative_attention_bias)
        self._t5_bias_cache_key = key
        self._t5_bias_cache_table = table
        return table

    def _forward_manual(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_k: Tensor,
        pos_v: Tensor,
        mask: Optional[Tensor],
        relative_attention_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Original manual attention implementation."""
        n_batch = query.size(0)

        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)  # (b, t, d)
        k = self.linear_k(key).view(n_batch, -1, self.h_k, self.d_k)  # (b, t, d)
        v = self.linear_v(value).view(n_batch, -1, self.h_k, self.d_k)
        q = (
            q.transpose(1, 2)
            if self.use_pt_scaled_dot_product_attention and not torch.jit.is_scripting()
            else q.transpose(1, 2) * self.inv_sqrt_d_k
        )
        k = k.transpose(1, 2)  # (batch, head_k, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head_k, time2, d_k)
        
        if self.use_pt_scaled_dot_product_attention and not torch.jit.is_scripting():
            attn_mask = None
            if mask is not None:
                mask = mask.unsqueeze(1)
                if relative_attention_bias is not None:
                    attn_mask = mask + relative_attention_bias
                else:
                    attn_mask = mask
                if mask.dtype != q.dtype:
                    attn_mask = attn_mask.to(q.dtype)

            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            ):
                x = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_rate,
                )
        else:
            if self.h != self.h_k:
                q = q.reshape(n_batch, self.g, self.h_k, -1, self.d_k)
                A = torch.einsum("b g h t d, b h s d -> b h t s", q, k)
            else:
                A = torch.matmul(q, k.transpose(-2, -1))
            if pos_k is not None:
                if self.h != self.h_k:
                    B = torch.einsum("b g h t d, t s d -> b h t s", q, pos_k)
                else:
                    reshape_q = (
                        q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0, 1)
                    )  # (t1,nh,dk)
                    B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))  # pos_k: (t1,dk,t2)
                    B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
                scores = A + B
            else:
                scores = A

            if relative_attention_bias is not None:
                if relative_attention_bias.dim() == 2:
                    # Expand compact table [H, 2*T] into [1, H, T, T] for manual fallback.
                    t1 = scores.size(-2)
                    t2 = scores.size(-1)
                    max_dist = max(t1, t2)
                    table = relative_attention_bias
                    if table.size(1) < 2 * max_dist:
                        pad = torch.zeros(
                            table.size(0),
                            2 * max_dist - table.size(1),
                            dtype=table.dtype,
                            device=table.device,
                        )
                        table = torch.cat([table, pad], dim=1)
                    rel = torch.arange(t2, device=scores.device).view(1, t2) - torch.arange(
                        t1, device=scores.device
                    ).view(t1, 1)
                    # Compact table keeps column 0 as guard; rel_pos=0 maps to column max_dist.
                    table_idx = torch.clamp(rel + max_dist, 0, table.size(1) - 1)
                    rel_bias = table[:, table_idx].unsqueeze(0)
                    scores = scores + rel_bias
                else:
                    scores = scores + relative_attention_bias

            attn = masked_softmax(scores, mask)  # (batch, head, time1, time2)

            self.attn = attn

            p_attn = self.dropout(attn)
            x = torch.matmul(p_attn.to(v.dtype), v)  # (batch, head, time1, d_k)
            if pos_v is not None:
                reshape_attn = (
                    p_attn.contiguous()
                    .view(n_batch * self.h, pos_v.size(0), pos_v.size(1))
                    .transpose(0, 1)
                )  # (t1, bh, t2)

                attn_v = (
                    torch.matmul(reshape_attn, pos_v)
                    .transpose(0, 1)
                    .contiguous()
                    .view(n_batch, self.h, pos_v.size(0), self.d_k)
                )
                x = x + attn_v
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h_k * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)


def unfold_tensor(xs_pad, max_seq_len):
    """
    For a given tensor with shape of (N, T, D), if sequence length T is longer than max_seq_len,
    this function unfold it to a (NT', max_seq_len, D) where T' is T // max_seq_len.
    Args:
        xs_pad: N, T, D
    """
    _, _, D = xs_pad.shape
    xs_pad = xs_pad.transpose(-1, -2) # convert to N, D, T
    # N x D x 1 x T => N x (D x max_seq_len) x T'
    xs_pad = F.unfold(
        xs_pad[..., None, :],
        kernel_size=(1, max_seq_len),
        stride=(1, max_seq_len),
    )

    new_bsz, _, slen = xs_pad.shape
    # N x D x max_seq_len x T'
    xs_pad = xs_pad.view(new_bsz, -1, max_seq_len, slen)
    # N x T' x max_seq_len x D
    xs_pad = xs_pad.permute(0, 3, 2, 1).contiguous()
    # NT' x max_seq_len x D
    xs_pad = xs_pad.view(-1, max_seq_len, D)
    return xs_pad
