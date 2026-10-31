"""Four physical Qwen3.5 MTP layers with depth-local, rollback-safe KV caches.

Only the small MTP stack lives here. The target, speech encoder, embeddings,
LM head and speculative verification remain vLLM-owned. Positions at each
depth match phyagi._mtp_run_depth: h[t] is paired with embedding[t+depth+1].
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps) * (1 + self.weight.float())).to(x.dtype)


@dataclass
class DepthCache:
    keys: torch.Tensor
    values: torch.Tensor
    hidden: torch.Tensor


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
        self.theta = cfg.rope_parameters.get("rope_theta", 10000000)
        self.q_proj = nn.Linear(cfg.hidden_size, self.heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def rotate(self, x, positions):
        inv_freq = 1 / (self.theta ** (torch.arange(0, self.rotary_dim, 2, device=x.device).float() / self.rotary_dim))
        angles = positions.float()[:, None] * inv_freq[None, :]
        angles = torch.cat((angles, angles), -1)[None, :, :]
        rot, rest = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        a, b = rot.chunk(2, -1)
        rot = rot * angles.cos().to(x.dtype) + torch.cat((-b, a), -1) * angles.sin().to(x.dtype)
        return torch.cat((rot, rest), -1)

    def forward(self, x, prefix, cache):
        q, gate = self.q_proj(x).view(-1, self.heads, self.head_dim * 2).chunk(2, -1)
        q = self.q_norm(q).transpose(0, 1)
        k = self.k_norm(self.k_proj(x).view(-1, self.kv_heads, self.head_dim)).transpose(0, 1)
        v = self.v_proj(x).view(-1, self.kv_heads, self.head_dim).transpose(0, 1)
        positions = torch.arange(prefix, prefix + x.shape[0], device=x.device)
        q, k = self.rotate(q, positions), self.rotate(k, positions)
        if prefix:
            k = torch.cat((cache.keys[:, :prefix], k), 1)
            v = torch.cat((cache.values[:, :prefix], v), 1)
        mask = None
        if prefix:
            mask = torch.arange(k.shape[1], device=x.device)[None, :] <= positions[:, None]
        attended = (
            F.scaled_dot_product_attention(
                q[None],
                k[None],
                v[None],
                attn_mask=mask,
                is_causal=not prefix,
                enable_gqa=True,
            )[0]
            .transpose(0, 1)
            .reshape(x.shape[0], -1)
        )
        return self.o_proj(attended * gate.reshape(x.shape[0], -1).sigmoid()), k, v


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, prefix, cache):
        attn, k, v = self.self_attn(self.input_layernorm(x), prefix, cache)
        x = x + attn
        return x + self.mlp(self.post_attention_layernorm(x)), k, v


class ExpandedMTP4(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.mtp_num_hidden_layers != 4:
            raise ValueError("ExpandedMTP4 requires four trained physical layers")
        self.fc = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_fc_norm_embedding = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(4)])
        self.depth_calls = [0] * 4

    def forward_depth(self, depth, hidden, shifted_embeds, stable_prefix=0, cache=None):
        # Cached depth d is stable only before target_update_start-d: later
        # positions may depend on previously rejected draft-token embeddings.
        prefix = min(stable_prefix, cache.hidden.shape[0]) if cache is not None else 0
        prefix = max(0, min(prefix, hidden.shape[0] - 1))
        x = self.fc(
            torch.cat(
                (self.pre_fc_norm_embedding(shifted_embeds[prefix:]), self.pre_fc_norm_hidden(hidden[prefix:])), -1
            )
        )
        x, keys, values = self.layers[depth](x, prefix, cache)
        x = self.norm(x)
        if prefix:
            x = torch.cat((cache.hidden[:prefix], x), 0)
        self.depth_calls[depth] += 1
        return x, DepthCache(keys, values, x)
