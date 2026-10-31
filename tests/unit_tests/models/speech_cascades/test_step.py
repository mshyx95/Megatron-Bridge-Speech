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

"""Unit tests for the speech-cascades training step (causal label shift) and dataset provider."""

import pytest


torch = pytest.importorskip("torch")

from megatron.bridge.data.speech_chunk import ALMDataCollator, ALMLocalDataset, ALMLocalDatasetProvider  # noqa: E402
from megatron.bridge.models.speech_cascades.step import get_alm_batch  # noqa: E402


def _iter_once(batch):
    yield batch


def test_get_alm_batch_applies_causal_shift():
    labels = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    input_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    batch = {"input_ids": input_ids, "labels": labels}

    out = get_alm_batch(_iter_once(batch))

    # shifted[i] = labels[i+1]; last position ignored.
    assert out["labels"].tolist() == [[6, 7, 8, 0]]
    assert out["loss_mask"].tolist() == [[1.0, 1.0, 1.0, 0.0]]
    assert out["audio_input_features"] is None


def test_get_alm_batch_masks_ignored_targets():
    labels = torch.tensor([[-100, -100, 42, 43]], dtype=torch.long)
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long), "labels": labels}

    out = get_alm_batch(_iter_once(batch))

    # shifted = [-100, 42, 43, -100] -> clamp negatives to 0, mask marks the two valid targets.
    assert out["labels"].tolist() == [[0, 42, 43, 0]]
    assert out["loss_mask"].tolist() == [[0.0, 1.0, 1.0, 0.0]]


def test_get_alm_batch_passes_audio_features():
    batch = {
        "input_ids": torch.tensor([[-1, -1, 9]], dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 9]], dtype=torch.long),
        "inputs_embeds": torch.ones(2, 4, 8),
        "audio_attention_mask": torch.ones(2, 4, dtype=torch.long),
        "audio_embed_sizes": torch.tensor([1, 1], dtype=torch.long),
    }
    out = get_alm_batch(_iter_once(batch))
    assert out["audio_input_features"] is not None
    assert out["audio_input_features"].shape == (2, 4, 8)
    assert out["audio_embed_sizes"].tolist() == [1, 1]


def _make_record(audio_lengths=(4, 5), feat_dim=8):
    embed_sizes = [max(1, t // 2) for t in audio_lengths]
    input_ids, labels = [], []
    for k, size in enumerate(embed_sizes):
        input_ids += [-(k + 1)] * size
        labels += [-100] * size
    input_ids += [10, 11]
    labels += [10, 11]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "input_embeds": [torch.ones(t, feat_dim) for t in audio_lengths],
        "audio_embed_sizes": torch.tensor(embed_sizes, dtype=torch.long),
        "audio_frames": torch.tensor(list(audio_lengths), dtype=torch.long),
        "is_text_sample": False,
    }


def test_local_dataset_has_collate_fn_and_records(tmp_path):
    shard = tmp_path / "train_000.pt"
    torch.save([_make_record(), _make_record((3,))], shard)
    ds = ALMLocalDataset([str(shard)], ALMDataCollator(pad_token_id=0))
    assert len(ds) == 2
    assert hasattr(ds, "collate_fn")
    batch = ds.collate_fn([ds[0], ds[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["inputs_embeds"].numel() > 0


def test_dataset_provider_builds_splits(tmp_path):
    train = tmp_path / "train_000.pt"
    val = tmp_path / "val_000.pt"
    torch.save([_make_record(), _make_record((3,))], train)
    torch.save([_make_record((2,))], val)

    provider = ALMLocalDatasetProvider(
        seq_length=64,
        shard_paths=[str(train)],
        val_shard_paths=[str(val)],
        pad_token_id=0,
    )
    context = type("Ctx", (), {"train_samples": 4, "valid_samples": 1, "test_samples": 0, "tokenizer": None})()
    train_ds, valid_ds, test_ds = provider.build_datasets(context)

    assert len(train_ds) == 2
    assert len(valid_ds) == 1
    assert test_ds is None
    assert hasattr(train_ds, "collate_fn")
