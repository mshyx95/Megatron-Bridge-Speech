"""vLLM Eagle interface adapter for the checkpoint's four independent depths.

Paged KV and GDN rollback of the target remain handled by vLLM. The MTP's
small depth-local caches explicitly discard speculative-dependent suffixes.
No recurrent reuse of layer zero, no extra encoder or backbone instance.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from vllm.logger import init_logger
from vllm.v1.spec_decode.eagle import EagleProposer

from speech_eval.mtp4.mtp_model import ExpandedMTP4


logger = init_logger(__name__)


class ExpandedMTP4Proposer(EagleProposer):
    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config, device, runner)
        if runner is None or vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("ExpandedMTP4 currently requires one local GPU and its model runner")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("ExpandedMTP4 requires --no-enable-prefix-caching for complete target hidden histories")
        if not self.speculative_config.disable_padded_drafter_batch:
            raise ValueError("ExpandedMTP4 requires disable_padded_drafter_batch=true")
        self.runner = runner
        # Consume the target's actual merged embeddings with exactly the same
        # accepted-token compaction as hidden states, not shifted audio masks.
        self.supports_mm_inputs = False
        self.current_target_embeddings = None
        self.contexts = {}
        self.proposal_count = 0
        self.runtime_depth_calls = [0] * 4
        self._draft_attn_layer_names = set()
        self._last_draft_probs = None

    def load_model(self, target_model):
        root = Path(self.vllm_config.model_config.model)
        cfg = SimpleNamespace(**json.loads((root / "config.json").read_text()))
        self.model = ExpandedMTP4(cfg).to(device=self.device, dtype=self.dtype).eval()
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        weights = {}
        for shard in sorted({s for k, s in index.items() if k.startswith("mtp.")}):
            with safe_open(root / shard, framework="pt", device="cpu") as source:
                for name in source.keys():
                    if name.startswith("mtp."):
                        weights[name.removeprefix("mtp.")] = source.get_tensor(name)
        self.model.load_state_dict(weights, strict=True)
        print(f"EXPANDED_MTP4_LOADED tensors={len(weights)} physical_layers=4", flush=True)
        # Register/share references, never make another text backbone or encoder.
        self.target = target_model
        self.embedding = target_model.model.embed_tokens
        self.lm_head = target_model.lm_head
        logger.info(
            "ExpandedMTP4 loaded %d MTP tensors; physical depths=0,1,2,3; shared target embedding/head", len(weights)
        )
        # Budget private target/embedding/MTP hidden and per-depth K/V caches.
        bytes_per_token = (6 * cfg.hidden_size + 8 * cfg.num_key_value_heads * cfg.head_dim) * 2
        self.cache_budget_bytes = self.max_batch_size * self.max_model_len * bytes_per_token

    def initialize_attn_backend(self, *args, **kwargs):
        # SDPA depth caches are private, not vLLM paged-attention layers.
        self.draft_attn_groups = []
        # Still request query/sequence metadata from a target cache group.
        # Leaving the base sentinel -1 makes vLLM silently skip proposing.
        self.kv_cache_gid = 0

    def validate_same_kv_cache_group(self, *args, **kwargs):
        pass

    def set_target_embeddings(self, embeddings):
        self.current_target_embeddings = embeddings

    def dummy_run(self, num_tokens, use_cudagraphs=True, is_graph_capturing=False, **kwargs):
        if is_graph_capturing:
            return  # The target keeps CUDA graphs; this proposer is eager.
        reserve = torch.empty(self.cache_budget_bytes, dtype=torch.uint8, device=self.device)
        n = min(num_tokens, self.max_num_tokens)
        hidden = torch.zeros((n, self.hidden_size), dtype=self.dtype, device=self.device)
        for depth in range(4):
            hidden, cache = self.model.forward_depth(depth, hidden, hidden)
        del reserve, hidden, cache

    @torch.inference_mode()
    def propose(
        self,
        num_speculative_tokens,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu=None,
        slot_mappings=None,
    ):
        if num_speculative_tokens not in (0, 4):
            raise ValueError("Expected four speculative tokens")
        if not sampling_metadata.all_greedy:
            raise ValueError("ExpandedMTP4 serving currently supports greedy decoding only (temperature=0)")
        req_ids = list(self.runner.input_batch.req_ids)
        for req in list(self.contexts):
            if req not in self.runner.requests:
                del self.contexts[req]
        cad = common_attn_metadata
        actual_embeds = self.current_target_embeddings
        if actual_embeds is None or actual_embeds.shape != target_hidden_states.shape:
            raise ValueError("Accepted target hidden/merged-embedding shapes differ")
        boundaries = cad.query_start_loc_cpu.tolist()
        positions = (target_positions[0] if target_positions.ndim == 2 else target_positions).tolist()
        drafts = []
        for row, req in enumerate(req_ids):
            lo, hi = boundaries[row : row + 2]
            start = positions[lo]
            if positions[lo:hi] != list(range(start, start + hi - lo)):
                raise ValueError("Non-contiguous accepted target positions")
            if start == 0:
                self.contexts.pop(req, None)
            old = self.contexts.get(req)
            if old is None and start != 0:
                raise ValueError("Missing target prefix hidden states; prefix caching must be disabled")
            if old is not None and old["hidden"].shape[0] < start:
                raise ValueError("Gap in target hidden history")
            hidden = target_hidden_states[lo:hi].clone()
            embeds = actual_embeds[lo:hi].clone()
            caches = [None] * 4 if old is None else old["caches"]
            if start:
                hidden = torch.cat((old["hidden"][:start], hidden), 0)
                embeds = torch.cat((old["embeds"][:start], embeds), 0)
            context = {"hidden": hidden, "embeds": embeds, "caches": caches}
            self.contexts[req] = context
            # Include target-selected c0 then shift per physical MTP depth.
            embeds = torch.cat((embeds[1:], self.embedding(next_token_ids[row : row + 1].clamp(min=0))), 0)
            ids = []
            for depth in range(4):
                hidden, caches[depth] = self.model.forward_depth(
                    depth, hidden, embeds, stable_prefix=max(0, start - depth - 1), cache=caches[depth]
                )
                self.runtime_depth_calls[depth] += 1
                token = self.target.compute_logits(hidden[-1:]).argmax(-1)
                ids.append(token[0])
                if depth < 3:
                    embeds = torch.cat((embeds[1:], self.embedding(token)), 0)
            drafts.append(torch.stack(ids))
        self.current_target_embeddings = None
        # Counts are actual physical-layer forwards, not requested draft lengths.
        logger.info_once(
            "ExpandedMTP4 active: four physical depth forwards per proposal; depth-local KV caches; greedy target verification"
        )
        self.proposal_count += 1
        audit = Path("/tmp/expanded_mtp4_runtime.json")
        audit.with_suffix(".tmp").write_text(
            json.dumps(
                {
                    "proposal_batches": self.proposal_count,
                    "runtime_depth_calls": self.runtime_depth_calls,
                    "physical_layers": 4,
                    "loaded_mtp_tensors": 48,
                    "speculative_tokens": 4,
                    "target_backbones": 1,
                    "speech_encoders": 1,
                    "cache_mode": "depth_local_rollback_safe",
                }
            )
        )
        audit.with_suffix(".tmp").replace(audit)
        return torch.stack(drafts)[:, :num_speculative_tokens]
