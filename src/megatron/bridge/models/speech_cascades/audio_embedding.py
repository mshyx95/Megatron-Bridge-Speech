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

"""Cascades audio embedding: the vendored ConformerEncoder plus the audio projection.

This mirrors the ``get_audio_features`` path of the production speech model for the
``cascades`` audio processor with an ``mlp``/``linear`` projection. Optional branches that the
production run does not use (QFormer, conv/temporal/bottleneck downsampling, CTC) are intentionally
not migrated and raise on request.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from megatron.bridge.models.speech_cascades.encoder import ConformerEncoder


_FULL_MODEL_AUDIO_PREFIX = "model.embed_tokens_extend."


@dataclass(frozen=True)
class AudioEmbeddingLoadReport:
    """Result of loading audio-embedding weights from a full-model checkpoint."""

    encoder_tensor_count: int
    projection_tensor_count: int


class CascadesAudioEmbedding(nn.Module):
    """Compose the Cascades ConformerEncoder with the audio projection to the LLM hidden size.

    Args:
        encoder_config: Keyword arguments for :class:`ConformerEncoder`.
        hidden_size: Target LLM hidden size the audio features are projected to.
        projection_cls: ``"mlp"`` (Linear-GELU-Linear) or ``"linear"`` (single Linear).
        compression_rate: Pre-encoder frame padding multiple factor from the run config.
        downsample_rate: Post-encoder temporal downsample factor for the projection input.
    """

    def __init__(
        self,
        encoder_config: dict[str, Any],
        *,
        hidden_size: int,
        projection_cls: str = "mlp",
        compression_rate: int = 1,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        if projection_cls not in ("mlp", "linear"):
            raise NotImplementedError(f"Unsupported projection_cls: {projection_cls}")

        self.encoder = ConformerEncoder(**encoder_config)
        # Build the CMVN layer so encoder weights (including nested encoder_embedding) can be loaded.
        self.encoder.post_init({})

        self.audio_dim_out = encoder_config["attention_dim"]
        self.hidden_size = hidden_size
        self.projection_cls = projection_cls
        self.compression_rate = compression_rate
        self.downsample_rate = downsample_rate

        # No QFormer/conv downsample in this migration slice, so the projection consumes encoder frames directly.
        self.linear_downsample_rate = downsample_rate
        self.post_encoder_downsample_rate = downsample_rate if projection_cls == "mlp" else 1
        self.pre_encoder_pad_multiple = (
            compression_rate * self.post_encoder_downsample_rate if self.post_encoder_downsample_rate != 1 else 1
        )

        projection_in = self.audio_dim_out * self.linear_downsample_rate
        if projection_cls == "linear":
            self.audio_projection: nn.Module = nn.Linear(projection_in, hidden_size)
        else:
            self.audio_projection = nn.Sequential(
                nn.Linear(projection_in, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )

    def _pad_input_embeds_to_multiple(
        self, input_embeds: torch.Tensor, audio_attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pad_multiple = self.pre_encoder_pad_multiple
        if pad_multiple <= 1:
            return input_embeds, audio_attention_mask
        padding = (-input_embeds.size(1)) % pad_multiple
        if padding == 0:
            return input_embeds, audio_attention_mask
        input_embeds = F.pad(input_embeds, (0, 0, 0, padding), "constant", 0)
        if audio_attention_mask is not None:
            audio_attention_mask = F.pad(audio_attention_mask, (0, padding), "constant", 0)
        return input_embeds, audio_attention_mask

    def get_audio_features(
        self, input_embeds: torch.Tensor, audio_attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return projected audio features of shape ``(B, T', hidden_size)``."""
        input_embeds, audio_attention_mask = self._pad_input_embeds_to_multiple(input_embeds, audio_attention_mask)
        audio_features, _ = self.encoder(input_embeds, audio_attention_mask)

        if self.linear_downsample_rate != 1:
            batch, seq_len, feat_dim = audio_features.size()
            remainder = seq_len % self.linear_downsample_rate
            if remainder > 0:
                audio_features = F.pad(
                    audio_features, (0, 0, 0, self.linear_downsample_rate - remainder), "constant", 0
                )
            seq_len = audio_features.size(1)
            audio_features = audio_features.view(
                batch, seq_len // self.linear_downsample_rate, feat_dim * self.linear_downsample_rate
            )

        return self.audio_projection(audio_features)

    def forward(
        self, input_embeds: torch.Tensor, audio_attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.get_audio_features(input_embeds, audio_attention_mask)

    def load_full_model_weights(
        self, state_dict: dict[str, torch.Tensor], *, prefix: str = _FULL_MODEL_AUDIO_PREFIX
    ) -> AudioEmbeddingLoadReport:
        """Load encoder (with nested CMVN) and projection weights from a full-model state dict.

        Args:
            state_dict: The ``module`` payload of a consolidated DeepSpeed checkpoint.
            prefix: Key prefix for the audio submodule in the full model.

        Raises:
            ValueError: If the encoder or projection weights are missing or unexpected.
        """
        encoder_prefix = prefix + "encoder."
        projection_prefix = prefix + "audio_projection."
        encoder_state = OrderedDict(
            (key[len(encoder_prefix) :], value)
            for key, value in state_dict.items()
            if key.startswith(encoder_prefix)
        )
        projection_state = OrderedDict(
            (key[len(projection_prefix) :], value)
            for key, value in state_dict.items()
            if key.startswith(projection_prefix)
        )
        if not encoder_state or not projection_state:
            raise ValueError(
                f"No audio weights under prefix {prefix!r}; "
                f"encoder={len(encoder_state)}, projection={len(projection_state)}"
            )

        encoder_incompatible = self.encoder.load_state_dict(encoder_state, strict=False)
        if encoder_incompatible.missing_keys or encoder_incompatible.unexpected_keys:
            raise ValueError(
                "Encoder weights are not one-to-one; "
                f"missing={sorted(encoder_incompatible.missing_keys)}, "
                f"unexpected={sorted(encoder_incompatible.unexpected_keys)}"
            )
        self.audio_projection.load_state_dict(projection_state, strict=True)

        return AudioEmbeddingLoadReport(
            encoder_tensor_count=len(encoder_state),
            projection_tensor_count=len(projection_state),
        )
