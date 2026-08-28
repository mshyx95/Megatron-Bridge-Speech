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

"""Determinism and schema tests for the speech-cascades ALM data pipeline."""

import pytest


torch = pytest.importorskip("torch")

from megatron.bridge.data.speech_chunk import (  # noqa: E402
    ALMDataCollator,
    ALMLocalChunkDataset,
    pad_features,
)


def _make_example(audio_lengths, feat_dim=8, pad_token_id=0):
    """Build one ALM example with negative-id audio placeholders."""
    # per-audio log-mel features (un-padded), plus an embed size per audio
    features = [torch.arange(t * feat_dim, dtype=torch.float32).reshape(t, feat_dim) for t in audio_lengths]
    embed_sizes = [max(1, t // 2) for t in audio_lengths]

    input_ids = []
    labels = []
    for k, size in enumerate(embed_sizes):
        input_ids.extend([-(k + 1)] * size)
        labels.extend([-100] * size)
    # a couple of real text tokens after the audio
    input_ids.extend([10, 11, 12])
    labels.extend([10, 11, 12])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "input_embeds": features,
        "audio_embed_sizes": torch.tensor(embed_sizes, dtype=torch.long),
        "audio_frames": torch.tensor(list(audio_lengths), dtype=torch.long),
        "is_text_sample": False,
    }


def test_pad_features_shapes_and_mask():
    feats = [torch.ones(3, 4), torch.ones(5, 4) * 2]
    padded, nframe = pad_features(feats, return_pt=True)
    assert padded.shape == (2, 5, 4)
    assert nframe == [3, 5]
    # padding region is zero
    assert torch.equal(padded[0, 3:], torch.zeros(2, 4))
    assert torch.equal(padded[1], torch.ones(5, 4) * 2)


def test_collator_schema_and_audio_mask():
    collator = ALMDataCollator(pad_token_id=0, padding_mask=True)
    ex = _make_example([4, 6])
    batch = collator([ex])

    assert set(batch) == {
        "input_ids",
        "labels",
        "attention_mask",
        "inputs_embeds",
        "audio_attention_mask",
        "audio_embed_sizes",
        "audio_frames",
    }
    assert batch["inputs_embeds"].shape == (2, 6, 8)
    assert batch["audio_attention_mask"].dtype == torch.long
    # mask row lengths equal the audio frame counts
    assert batch["audio_attention_mask"][0].sum().item() == 4
    assert batch["audio_attention_mask"][1].sum().item() == 6
    assert batch["audio_embed_sizes"].tolist() == [2, 3]
    assert batch["audio_frames"].tolist() == [4, 6]
    # negative placeholder count equals sum of embed sizes
    assert int((batch["input_ids"] < 0).sum()) == 5


def test_collator_is_deterministic():
    collator = ALMDataCollator(pad_token_id=0, padding_mask=True)
    ex = _make_example([4, 6, 3])
    b1 = collator([_make_example([4, 6, 3])])
    b2 = collator([ex])
    for key in b1:
        assert torch.equal(b1[key], b2[key]), key


def test_collator_pads_across_batch():
    collator = ALMDataCollator(pad_token_id=7)
    short = _make_example([2])
    long = _make_example([4, 4])
    batch = collator([short, long])
    # input_ids padded to the longer sequence with pad_token_id
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == max(short["input_ids"].numel(), long["input_ids"].numel())
    assert (batch["input_ids"][0, short["input_ids"].numel() :] == 7).all()
    # labels padded with -100
    assert (batch["labels"][0, short["labels"].numel() :] == -100).all()
    # 1 + 2 = 3 audio segments total
    assert batch["inputs_embeds"].shape[0] == 3


def test_local_dataset_yields_nonempty_batches(tmp_path):
    shard = [_make_example([4, 6]), _make_example([3]), _make_example([5, 2])]
    shard_path = tmp_path / "shard_000.pt"
    torch.save(shard, shard_path)

    collator = ALMDataCollator(pad_token_id=0, padding_mask=True)
    ds = ALMLocalChunkDataset([str(shard_path)], collator, batch_size=2)
    batches = list(ds)

    assert len(batches) == 2  # 3 examples -> batch of 2 + batch of 1
    for b in batches:
        assert b["input_ids"].shape[0] >= 1
        assert b["inputs_embeds"].numel() > 0
        assert b["audio_embed_sizes"].numel() > 0


def test_local_dataset_rank_sharding(tmp_path):
    paths = []
    for s in range(4):
        p = tmp_path / f"shard_{s:03d}.pt"
        torch.save([_make_example([4])], p)
        paths.append(str(p))

    collator = ALMDataCollator(pad_token_id=0)
    r0 = list(ALMLocalChunkDataset(paths, collator, batch_size=1, rank=0, world_size=2))
    r1 = list(ALMLocalChunkDataset(paths, collator, batch_size=1, rank=1, world_size=2))
    # each rank sees half the shards, and together they cover all of them
    assert len(r0) == 2
    assert len(r1) == 2
