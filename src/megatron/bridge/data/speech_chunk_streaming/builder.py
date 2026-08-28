"""Speech-to-text chunk dataset builders (migrated pipeline, no phyagi dependency).

Builds the streaming audio-chunk dataset and its collator from plain, already-resolved
config dicts, using the migrated `chunk` package. Designed to plug into Megatron's
existing external dataloader path (``--dataloader-type external``): a provider builds
the dataset + collator and hands Megatron a ready ``DataLoader``.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional

from .blob_chunk_alm.blob_chunk_alm_dataset_provider import (
    BlobChunkALMDatasetProvider,
    BlobChunkALMDatasetProviderConfig,
)
from .concat_dataset import WeightedConcatIterableDataset
from .data_collator import ALMDataDynamicCollator

# Per-entry provider registry: maps the ``dataset_provider`` key found on each
# config entry to a ``(ProviderClass, ProviderConfigClass)`` pair. This mirrors the
# reference's per-entry ``dataset_provider`` dispatch, letting a single weighted
# stream mix heterogeneous sources. Register new providers here; each provider must
# expose ``from_config`` / ``get_train_dataset`` and yield samples the configured
# collator can batch.
DATASET_PROVIDER_REGISTRY: dict[str, tuple[type, type]] = {
    "blob_chunk_alm": (BlobChunkALMDatasetProvider, BlobChunkALMDatasetProviderConfig),
    # Aliases so configs may name the provider by its class name.
    "BlobChunkALMDatasetProvider": (
        BlobChunkALMDatasetProvider,
        BlobChunkALMDatasetProviderConfig,
    ),
}

_DEFAULT_PROVIDER = "blob_chunk_alm"


def _resolve_provider(entry: dict) -> tuple[type, type]:
    name = entry.get("dataset_provider", _DEFAULT_PROVIDER) or _DEFAULT_PROVIDER
    if name not in DATASET_PROVIDER_REGISTRY:
        raise KeyError(
            f"Unknown dataset_provider {name!r}; "
            f"registered providers: {sorted(DATASET_PROVIDER_REGISTRY)}"
        )
    return DATASET_PROVIDER_REGISTRY[name]


def _normalize_entry(entry: dict, force_local: bool) -> dict:
    entry = copy.deepcopy(entry)
    entry.pop("dataset_provider", None)  # dispatch key, not a provider-config field
    if force_local:
        entry["use_storage_api"] = False
        entry["multi_threaded_download"] = False
        entry.pop("path_prefix_in_orange", None)
    return entry


def _build_provider(entry: dict, force_local: bool):
    provider_cls, config_cls = _resolve_provider(entry)
    return provider_cls.from_config(config_cls.from_dict(_normalize_entry(entry, force_local)))


def build_speech_chunk_dataset(
    dataset_entries: list[dict],
    rank: int = 0,
    world_size: int = 1,
    force_local: bool = True,
) -> Any:
    """Build the (weighted, streaming) training dataset from resolved config entries.

    Each entry is dispatched to a provider via its ``dataset_provider`` key (see
    ``DATASET_PROVIDER_REGISTRY``), so a single weighted stream can mix heterogeneous
    sources. Entries are always wrapped in a ``WeightedConcatIterableDataset`` (even a
    single entry) so iteration order and shuffling match the reference exactly.
    """
    providers = [_build_provider(e, force_local) for e in dataset_entries]
    datasets = [p.get_train_dataset() for p in providers]
    weights = [getattr(p, "weight", 1.0) for p in providers]

    dataset = WeightedConcatIterableDataset(datasets, weights)
    if hasattr(dataset, "set_rank_and_world_size"):
        dataset.set_rank_and_world_size(rank, world_size)
    return dataset


def build_speech_chunk_collator(collator_cfg: dict, tokenizer) -> Callable:
    """Build the dynamic ALM collator from a resolved ``dataset_collator`` config."""
    cfg = dict(collator_cfg)
    cfg.pop("cls", None)
    cfg["tokenizer"] = tokenizer
    return ALMDataDynamicCollator(**cfg)
