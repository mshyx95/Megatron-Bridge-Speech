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

"""Native Megatron-Core composition of the Cascades audio encoder with a Qwen3.5 language model.

Follows the repo's Qwen3-ASR pattern: an audio module produces embeddings that are scattered into
the language model's token embedding stream at ``audio_token_id`` positions, then the Megatron-Core
GPT model runs on the merged ``decoder_input``.
"""

from typing import Any

import torch
from megatron.core.transformer import MegatronModule

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.speech_cascades.audio_embedding import CascadesAudioEmbedding
from megatron.bridge.models.speech_cascades.mtp import SpeechCascadesMTP


class SpeechCascadesModel(MegatronModule):
    """Qwen3.5 language model with a Cascades audio encoder attached (audio-only, no vision)."""

    def __init__(
        self,
        config: GPTModelProvider,
        *,
        audio_token_id: int,
        audio_encoder_config: dict[str, Any],
        audio_projection_cls: str = "mlp",
        audio_compression_rate: int = 1,
        audio_downsample_rate: int = 1,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.audio_token_id = audio_token_id

        # Reuse the tested GPT builder for the Qwen3.5 (GDN hybrid) language model.
        self.language_model = GPTModelProvider.provide(
            config, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        self.audio_model: CascadesAudioEmbedding | None = None
        if pre_process:
            self.audio_model = CascadesAudioEmbedding(
                audio_encoder_config,
                hidden_size=config.hidden_size,
                projection_cls=audio_projection_cls,
                compression_rate=audio_compression_rate,
                downsample_rate=audio_downsample_rate,
            )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

        # Optional multi-token-prediction depth heads (WP-C1). Off by default: when
        # ``mtp_num_hidden_layers == 0`` the model has no ``mtp`` module and the ASR forward path
        # below is byte-identical to the pre-MTP model.
        self.mtp: SpeechCascadesMTP | None = None
        # Per-depth hidden states from the most recent MTP forward, exposed for the per-depth loss
        # (WP-C2). ``None`` whenever MTP is disabled or inactive (e.g. no labels).
        self.mtp_hidden_states: list[torch.Tensor] | None = None
        # Main decoder hidden states (bshd ``[B, S, H]``) from the most recent forward, exposed for
        # the KD response-token selection (WP-A2). ``None`` unless the caller requested capture.
        self.captured_hidden_states: torch.Tensor | None = None
        mtp_num_depth_layers = int(getattr(config, "mtp_num_hidden_layers", 0) or 0)
        if mtp_num_depth_layers > 0:
            self.mtp = self._build_mtp(config, mtp_num_depth_layers, vp_stage)

    def _build_mtp(
        self, config: GPTModelProvider, num_depth_layers: int, vp_stage: int | None
    ) -> SpeechCascadesMTP:
        """Build the MTP depth heads from the language-model decoder's first-layer spec."""
        from megatron.core.transformer.spec_utils import build_module

        decoder = self.language_model.decoder
        layer_spec = decoder.submodules.layer_specs[0]
        pg_collection = getattr(decoder, "pg_collection", None)

        def _layer_builder():
            return build_module(
                layer_spec,
                config=config,
                layer_number=1,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
            )

        # init-from-first: every depth block starts as a copy of the first decoder layer.
        source_state_dict = None
        if getattr(config, "mtp_init_extra_layers_from_first", False):
            source_state_dict = decoder.layers[0].state_dict()

        return SpeechCascadesMTP(
            hidden_size=config.hidden_size,
            num_depth_layers=num_depth_layers,
            layer_builder=_layer_builder,
            source_state_dict=source_state_dict,
            norm_eps=config.layernorm_epsilon,
            zero_centered_gamma=getattr(config, "layernorm_zero_centered_gamma", False),
        )

    def shared_embedding_or_output_weight(self):
        """Surface the language model word embeddings for gradient all-reduce."""
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "SpeechCascadesModel expects a single input tensor"
        self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        *,
        freeze_language_model: bool = False,
        freeze_audio_model: bool = False,
        freeze_audio_encoder: bool = False,
    ) -> None:
        """Freeze the language model, the whole audio module, and/or just the audio encoder.

        ``freeze_audio_encoder`` freezes only the Conformer encoder inside the audio module, leaving
        the projection (and the MTP depth heads) trainable, as required by the stage-2 KD recipe.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_audio_model and self.audio_model is not None:
            modules.append(self.audio_model)
        if (
            freeze_audio_encoder
            and self.audio_model is not None
            and getattr(self.audio_model, "encoder", None) is not None
        ):
            modules.append(self.audio_model.encoder)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _merge_audio_embeddings(
        self,
        input_ids: torch.Tensor,
        audio_input_features: torch.Tensor,
        audio_attention_mask: torch.Tensor | None,
        audio_embed_sizes: torch.Tensor | None,
    ) -> torch.Tensor:
        # Audio features are (num_audio, T', hidden); keep only valid frames per audio when sizes are given.
        audio_features = self.audio_model(audio_input_features, audio_attention_mask)
        if audio_embed_sizes is not None:
            audio_features = torch.cat(
                [audio_features[i, : int(size)] for i, size in enumerate(audio_embed_sizes)], dim=0
            )
        else:
            audio_features = audio_features.reshape(-1, audio_features.size(-1))

        # Cascades marks audio placeholder positions with negative token ids; clamp them for the
        # embedding lookup, then scatter audio features into those positions.
        audio_mask = input_ids < 0
        safe_input_ids = input_ids.clamp_min(0)
        combined = self.language_model.embedding(input_ids=safe_input_ids, position_ids=None).clone()
        combined = combined.transpose(0, 1).contiguous()
        combined[audio_mask] = audio_features.to(combined.dtype)
        return combined.transpose(0, 1).contiguous()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        audio_input_features: torch.Tensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
        audio_embed_sizes: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        decoder_input = None
        if self.pre_process:
            if audio_input_features is not None:
                decoder_input = self._merge_audio_embeddings(
                    input_ids, audio_input_features, audio_attention_mask, audio_embed_sizes
                )
            else:
                decoder_input = self.language_model.embedding(
                    input_ids=input_ids.clamp_min(0), position_ids=None
                )

        if position_ids is None:
            # decoder_input is precomputed, so the LM cannot infer positions from input_ids.
            sequence = torch.arange(input_ids.shape[1], device=input_ids.device)
            position_ids = sequence.unsqueeze(0).expand(input_ids.shape[0], -1)

        self.captured_hidden_states = None
        self.mtp_hidden_states = None

        # ASR path (no hidden-state capture requested, MTP disabled, or no labels): byte-identical
        # to the pre-KD/pre-MTP model. Only KD (``return_hidden_states``) or MTP takes the capture
        # branch below.
        capture_hidden = return_hidden_states or self.mtp is not None
        if not capture_hidden or labels is None:
            return self.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                loss_mask=loss_mask,
                **kwargs,
            )

        # Capture path: stash the main decoder hidden states (for the KD response selection and/or
        # the MTP depth heads) while the capture hook reproduces the standard output-layer loss so
        # the returned CE output is unchanged.
        holder: dict[str, torch.Tensor] = {}
        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            output_processor=self._capture_hidden_states(holder),
            **kwargs,
        )
        full_hidden_states = holder.get("hidden_states")
        if full_hidden_states is not None:
            # Expose the main hidden states in bshd for the KD response selection in step.py.
            self.captured_hidden_states = full_hidden_states.transpose(0, 1).contiguous()
            if self.mtp is not None:
                self.mtp_hidden_states = self.mtp(
                    full_hidden_states,
                    decoder_input,
                    rotary_pos_emb_fn=self._mtp_rotary_fn(),
                    attention_mask=None,
                )
        return output

    def text_forward_hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Text-only teacher forward: run the frozen language model with NO audio.

        Runs the language model on ``input_ids`` (audio spans already replaced by their aligned
        transcript tokens by the KD collator) and returns the decoder hidden states as bshd
        ``[B, S, H]``. Intended to be called under ``torch.no_grad()`` at the KD call site; it builds
        no audio graph and only materializes hidden states (no full-vocab logits). Right-padded
        teacher sequences are handled by the causal attention, matching the student path.

        Args:
            input_ids: ``[B, S]`` teacher text token ids (no negative audio placeholders).
            attention_mask: Optional attention mask forwarded to the language model. The training
                pipeline passes ``None`` (causal), which is exact for right-padded sequences.

        Returns:
            Decoder hidden states ``[B, S, H]`` (bshd).
        """
        decoder_input = None
        if self.pre_process:
            decoder_input = self.language_model.embedding(input_ids=input_ids.clamp_min(0), position_ids=None)

        sequence = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_ids = sequence.unsqueeze(0).expand(input_ids.shape[0], -1)

        holder: dict[str, torch.Tensor] = {}
        self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=None,
            output_processor=self._capture_hidden_states_only(holder),
        )
        return holder["hidden_states"].transpose(0, 1).contiguous()

    def _mtp_rotary_fn(self):
        """Return a ``seq_len -> rotary_pos_emb`` callable from the LM's rotary embedding, or None."""
        rotary_pos_emb = getattr(self.language_model, "rotary_pos_emb", None)
        if rotary_pos_emb is None:
            return None
        return lambda seq_len: rotary_pos_emb(seq_len)

    @staticmethod
    def _capture_hidden_states(holder: dict[str, torch.Tensor]):
        """Output-processor hook: stash the decoder hidden states and reproduce the standard output."""

        def _process(
            *,
            hidden_states,
            output_layer,
            output_weight,
            labels,
            runtime_gather_output,
            compute_language_model_loss,
            scale_logits,
            **_,
        ):
            holder["hidden_states"] = hidden_states
            logits, _ = output_layer(
                hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
            )
            logits = scale_logits(logits)
            if labels is None:
                return logits.transpose(0, 1).contiguous()
            return compute_language_model_loss(labels, logits)

        return _process

    @staticmethod
    def _capture_hidden_states_only(holder: dict[str, torch.Tensor]):
        """Output-processor hook that captures decoder hidden states without materializing logits.

        Used by the text-only teacher forward, which only needs the hidden states; skipping the
        output-layer projection avoids a full ``[seq, vocab]`` logits tensor.
        """

        def _process(*, hidden_states, **_):
            holder["hidden_states"] = hidden_states
            return hidden_states

        return _process
