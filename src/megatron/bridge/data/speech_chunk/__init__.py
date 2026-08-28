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

"""Speech-cascades audio-language-model (ALM) data pipeline.

This package provides the deterministic batch-assembly components used to feed
the ``SpeechCascadesModel``: a padding collator that produces the exact batch
schema consumed by the model forward (``input_ids``, ``labels``,
``inputs_embeds``, ``audio_attention_mask``, ``audio_embed_sizes``,
``audio_frames``) and a local iterable dataset that yields collated batches from
pre-serialized example records. The batch schema matches the Cascades ALM
baseline so captured reference batches can be reproduced bit-identically.
"""

from megatron.bridge.data.speech_chunk.collator import ALMDataCollator, pad_features
from megatron.bridge.data.speech_chunk.dataset import ALMLocalDataset, ALMLocalDatasetProvider
from megatron.bridge.data.speech_chunk.local_dataset import ALMLocalChunkDataset
from megatron.bridge.data.speech_chunk.streaming_provider import SpeechChunkStreamingDatasetProvider


__all__ = [
    "ALMDataCollator",
    "ALMLocalChunkDataset",
    "ALMLocalDataset",
    "ALMLocalDatasetProvider",
    "SpeechChunkStreamingDatasetProvider",
    "pad_features",
]
