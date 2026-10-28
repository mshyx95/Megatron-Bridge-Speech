# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Greedy autoregressive generation for :class:`SpeechCascadesModel`.

The generation is cache-free (the full sequence is re-run each step) and memory-safe: only the last
position's logits are materialized by capturing the output-layer input via a forward pre-hook and
projecting a single position, avoiding a full ``[seq, vocab]`` logits tensor. Audio features are
supplied every step since the negative-id audio placeholders live in the prompt prefix, which is
preserved as new (non-negative) tokens are appended.
"""

from collections.abc import Sequence

import torch
from megatron.core.transformer import MegatronModule


class _StopForward(Exception):
    pass


def _unwrap(model: MegatronModule) -> MegatronModule:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def greedy_generate(
    model: MegatronModule,
    input_ids: torch.Tensor,
    *,
    audio_input_features: torch.Tensor | None = None,
    audio_attention_mask: torch.Tensor | None = None,
    audio_embed_sizes: torch.Tensor | None = None,
    max_new_tokens: int = 16,
    eos_token_ids: Sequence[int] = (),
) -> torch.Tensor:
    """Greedily decode continuation tokens from an audio+text prompt.

    Args:
        model: The (possibly wrapped) :class:`SpeechCascadesModel`.
        input_ids: ``[1, prompt_len]`` prompt with negative-id audio placeholders.
        audio_input_features: Padded audio features ``[num_audio, T, F]`` (or ``None`` for text-only).
        audio_attention_mask: Audio frame mask ``[num_audio, T]``.
        audio_embed_sizes: Per-audio placeholder counts ``[num_audio]``.
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_ids: Stop generation when one of these ids is produced.

    Returns:
        ``[1, prompt_len + generated]`` token ids including the prompt.
    """
    inner = _unwrap(model)
    gpt = inner.language_model
    weight = gpt.shared_embedding_or_output_weight() if gpt.share_embeddings_and_output_weights else None
    eos = set(int(t) for t in eos_token_ids)

    generated = input_ids
    for _ in range(max_new_tokens):
        captured: dict[str, torch.Tensor] = {}

        def _capture(_module, args):
            captured["hidden"] = args[0]
            raise _StopForward()

        handle = gpt.output_layer.register_forward_pre_hook(_capture)
        forward_args = {"input_ids": generated, "position_ids": None, "attention_mask": None}
        if audio_input_features is not None:
            forward_args["audio_input_features"] = audio_input_features
            forward_args["audio_attention_mask"] = audio_attention_mask
            forward_args["audio_embed_sizes"] = audio_embed_sizes
        try:
            inner(**forward_args)
        except _StopForward:
            pass
        finally:
            handle.remove()

        hidden_last = captured["hidden"][-1:]  # [1, batch, hidden]
        logits, _ = gpt.output_layer(hidden_last, weight=weight)  # [1, batch, vocab]
        next_token = logits[-1, :, :].argmax(dim=-1, keepdim=True)  # [batch, 1]
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.numel() == 1 and int(next_token.item()) in eos:
            break

    return generated
