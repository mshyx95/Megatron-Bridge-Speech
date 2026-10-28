"""Data path for the speech-to-text extension (migrated, phyagi-free)."""

from .builder import (
    build_speech_chunk_collator,
    build_speech_chunk_dataset,
)

__all__ = [
    "build_speech_chunk_dataset",
    "build_speech_chunk_collator",
]
