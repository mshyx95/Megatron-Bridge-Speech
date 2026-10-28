"""Exercise imported native readers and the Azure SDK adapter with in-memory transport."""

import io
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm import az_downloader
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.blobchunk import (
    DatasetSpec,
    _BlobChunkIterableDatasetIterator,
)


pytestmark = pytest.mark.unit


def _chunk(kind, payloads):
    result = kind.encode() + struct.pack("<I", 1)
    for index, payload in enumerate(payloads):
        result += struct.pack("<II", index, len(payload)) + payload
    return result


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "answer_root", ["https://rewrite.blob.core.windows.net/answers/set", "az://rewrite/answers/set"]
)
def test_native_api_reader_spill_and_resume_preserve_audio_and_teacher(monkeypatch, tmp_path, parallel, answer_root):
    audio = io.BytesIO()
    sf.write(audio, np.zeros(160, dtype=np.float32), 16000, format="WAV")
    payloads = {
        ("https://source.blob.core.windows.net", "data", "original/audio/chunk0.audio"): _chunk(
            "audio", [audio.getvalue()] * 3
        ),
        ("https://source.blob.core.windows.net", "data", "original/transcripts/chunk0.transcription"): _chunk(
            "transcription", [b"source-0", b"source-1", b"source-2"]
        ),
        ("https://rewrite.blob.core.windows.net", "answers", "set/chunk0.answer"): _chunk(
            "answer", [b"teacher-0", b"teacher-1", b"teacher-2"]
        ),
    }
    created = []
    reads = []

    class Container:
        def __init__(self, *, account_url, container_name, **kwargs):
            self.account = account_url.rstrip("/")
            self.container = container_name
            created.append((self.account, container_name))

        def download_blob(self, blob, *, offset, **kwargs):
            key = (self.account, self.container, blob)
            reads.append(key)
            payload = payloads[key][offset:]
            return SimpleNamespace(chunks=lambda **kwargs: iter([payload]))

    monkeypatch.setenv("UAI_CLIENT_ID", "test-managed-identity")
    monkeypatch.setenv("AZ_DATA_ACCOUNT", "https://source.blob.core.windows.net/data/")
    monkeypatch.setattr(az_downloader, "ManagedIdentityCredential", lambda **kwargs: object())
    monkeypatch.setattr(az_downloader, "ContainerClient", Container)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"fileInfo": [{"name": "chunk0", "count": 3, "duration": 1.0}]}))
    spec = DatasetSpec(
        {
            "data_sources": [
                {
                    "manifest_file": str(manifest),
                    "chunk_type": ["audio", "transcription", "answer"],
                    "chunk_path": "/datablob/original/audio",
                    "trans_path": "/datablob/original/transcripts",
                    "additional_data_path": {"answer": answer_root},
                    "language": "en",
                }
            ]
        },
        use_storage_api=True,
    )

    def iterate(spill, resume=0):
        if spill:
            monkeypatch.setenv("BLOBCHUNK_SPILL_DIR", str(tmp_path / "spill"))
        else:
            monkeypatch.delenv("BLOBCHUNK_SPILL_DIR", raising=False)
        return _BlobChunkIterableDatasetIterator(spec, 1, 3, resume, 1, 0, 17, parallel, False)

    in_memory = iterate(False)
    expected = [next(in_memory) for _ in range(6)]
    spilled = iterate(True)
    actual = [next(spilled) for _ in range(6)]
    resumed = iterate(True, resume=1)
    after_resume = [next(resumed) for _ in range(5)]
    assert [sample["transcription"] for sample in actual] == [sample["transcription"] for sample in expected]
    assert [sample["answer"] for sample in after_resume] == [sample["answer"] for sample in expected[1:]]
    for sample in actual:
        waveform, sample_rate = sample["audio"]
        assert waveform.shape == (160,) and sample_rate == 16000
        assert sample["answer"].replace("teacher", "source") == sample["transcription"]
    assert set(reads) == set(payloads)
    assert created == [
        ("https://source.blob.core.windows.net", "data"),
        ("https://rewrite.blob.core.windows.net", "answers"),
    ]
    assert list((tmp_path / "spill").iterdir()) == []
