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

"""External-dataloader provider over the vendored streaming speech-chunk pipeline.

Mirrors the baseline ``pretrain_speech.py`` integration: it loads the OmegaConf data config,
builds the weighted streaming ``blob_chunk_alm`` dataset and its dynamic ALM collator, and returns a
ready ``DataLoader`` iterator through the ``external`` dataloader path. The collator does dynamic
(frame-budget) batching, so the loader uses ``batch_size=1`` and each ``next`` yields one packed
micro-batch with the schema the speech-cascades forward step consumes.
"""

from dataclasses import dataclass, field
from typing import Any

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


@dataclass(kw_only=True)
class SpeechChunkStreamingDatasetProvider(DatasetProvider):
    """Stream real speech data via the vendored ``speech_chunk`` pipeline (``external`` loader).

    Args:
        seq_length: Sequence length; must match ``model.seq_length``.
        speech_config_yaml: OmegaConf data config with ``dataset`` entries and ``dataset_collator``.
        force_local: Read from mounted paths instead of the Azure storage-API stream.
        data_spec_override: Optional list of ``dataset_spec`` JSON paths restricting the first entry.
        token_budget_override: Optional value for each entry's ``seq_len`` / ``batcher_batch_size``
            (shrinks utterances per packed batch for memory-limited runs).
    """

    seq_length: int
    speech_config_yaml: str
    force_local: bool = False
    data_spec_override: list[str] | None = None
    token_budget_override: int | None = None
    dataloader_type: str = "external"
    num_workers: int = 1
    prefetch_factor: int = 8
    skip_getting_attention_mask_from_dataset: bool = True
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Any, None, None]:
        from omegaconf import OmegaConf
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer

        from megatron.bridge.data.speech_chunk_streaming import (
            build_speech_chunk_collator,
            build_speech_chunk_dataset,
        )

        cfg = OmegaConf.to_container(OmegaConf.load(self.speech_config_yaml), resolve=True)
        entries = cfg["dataset"]
        if cfg["dataset_collator"].get("knowledge_distillation", False):
            for entry in entries:
                transform = entry.get("transform_cls")
                if (
                    not entry.get("allow_pure_text_return", False)
                    and isinstance(transform, dict)
                    and transform.get("name") == "AudioTextConversationTransform"
                ):
                    transform.setdefault("knowledge_distillation", True)
        if self.data_spec_override:
            first = dict(entries[0])
            first["dataset_spec"] = list(self.data_spec_override)
            entries = [first]
        if self.token_budget_override:
            entries = [
                {**dict(e), "seq_len": self.token_budget_override, "batcher_batch_size": self.token_budget_override}
                for e in entries
            ]

        tokenizer = AutoTokenizer.from_pretrained(
            entries[0]["tokenizer"]["pretrained_model_name_or_path"], trust_remote_code=True
        )
        dp_rank, dp_world = self._data_parallel_rank_and_size(context)
        dataset = build_speech_chunk_dataset(entries, rank=dp_rank, world_size=dp_world, force_local=self.force_local)
        collator = build_speech_chunk_collator(cfg["dataset_collator"], tokenizer)
        # Deep prefetch hides the Azure-download / shuffle-buffer warmup that otherwise shows as
        # intermittent GPU-idle stalls tapering over the first ~1-2k steps.
        loader_kwargs: dict[str, Any] = {
            "batch_size": 1,
            "collate_fn": collator,
            "num_workers": self.num_workers,
            "pin_memory": getattr(self, "pin_memory", False),
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = getattr(self, "persistent_workers", False)
        loader = DataLoader(dataset, **loader_kwargs)
        return iter(loader), None, None

    @staticmethod
    def _data_parallel_rank_and_size(context: DatasetBuildContext) -> tuple[int, int]:
        pg = getattr(context, "pg_collection", None)
        if pg is not None and getattr(pg, "dp", None) is not None:
            import torch.distributed as dist

            return dist.get_rank(group=pg.dp), dist.get_world_size(group=pg.dp)
        from megatron.core import parallel_state as mpu

        return mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
