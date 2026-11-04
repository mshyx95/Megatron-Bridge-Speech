import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location(
        "elift_data_test", ROOT / "examples/models/speech_cascades/prepare_elift_data.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _config(tmp_path):
    sources = []
    for index in range(2):
        chunks = tmp_path / f"chunks-{index}"
        chunks.mkdir()
        descriptors = []
        for number in range(10):
            name = f"chunk-{number}"
            (chunks / f"{name}.feature").touch()
            (chunks / f"{name}.transcription").touch()
            descriptors.append({"name": name, "count": 1, "duration": 3000 * 3600})
        manifest = _write(tmp_path / f"manifest-{index}.json", {"fileInfo": descriptors})
        sources.append({"manifest_file": str(manifest), "chunk_path": str(chunks), "language": "en"})
    spec = _write(
        tmp_path / "en.json", {"chunk_type": ["feature", "transcription"], "language": "en", "data_sources": sources}
    )
    return {
        "dataset": [
            {
                "dataset_spec": [{"file": [str(spec)], "sample_ratio": 0.5}],
                "weight": 2.0,
                "filter_options": {"max_utterance_frames": 4000},
            },
            {"dataset_spec": [str(spec)], "weight": 1.0},
        ]
    }


@pytest.mark.unit
def test_sampling_preserves_weights_nested_membership_and_all_four_hour_sizes(tool, tmp_path):
    config = _config(tmp_path)
    weights = tool._spec_weights(config)
    assert list(weights.values()) == [2.0]
    output = tmp_path / "sampled"
    tool.prepare(config, output, mode="tier1_per_language", seed=17, languages=frozenset({"en"}))
    assert tool.HOURS == (2000, 5000, 10000, 20000)
    previous = set()
    for size in ("2k", "5k", "10k", "20k"):
        members = set()
        for path in (output / "en" / size / "manifests").rglob("*.json"):
            members.update(
                (path.parent.name, path.name, entry["name"]) for entry in json.loads(path.read_text())["fileInfo"]
            )
        assert previous <= members
        previous = members
        data = yaml.safe_load((output / "en" / size / "configs" / f"data-v3.5-tier1-{size}.yaml").read_text())
        assert [entry["weight"] for entry in data["dataset"]] == [2.0, 1.0]
        assert data["dataset"][0]["filter_options"] == config["dataset"][0]["filter_options"]
        assert data["dataset"][0]["dataset_spec"][0]["sample_ratio"] == 0.5
    with pytest.raises(FileExistsError):
        tool.prepare(config, output, mode="tier1_per_language", seed=17, languages=frozenset({"en"}))


@pytest.mark.unit
def test_partial_sampling_retains_evidence_but_is_not_success(tool, tmp_path):
    config = _config(tmp_path)
    for manifest in tmp_path.glob("manifest*.json"):
        payload = json.loads(manifest.read_text())
        for entry in payload["fileInfo"]:
            entry["duration"] = 1
        manifest.write_text(json.dumps(payload))
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="Insufficient hours"):
        tool.prepare(config, output, mode="tier1_per_language", seed=17, languages=frozenset({"en"}))
    summary = json.loads((output / "en/summary.json").read_text())
    assert not any(row["complete"] for row in summary["targets"].values())
    with pytest.raises(ValueError, match="incomplete"):
        tool.publish(
            output / "en",
            tmp_path / "published",
            mount_root="/fixture/data",
            mode="tier1_per_language",
            language="en",
            variant_suffix="",
        )


@pytest.mark.unit
def test_missing_tier1_chunks_are_excluded_with_a_warning(tool, tmp_path, caplog):
    config = _config(tmp_path)
    (tmp_path / "chunks-0/chunk-0.transcription").unlink()
    components = tool._components(config, mode="tier1_per_language", seed=7, hindi_spec=None, count_tokens=len)
    assert len(components[0].chunks) == 9
    assert "Excluding 1 incomplete en chunks" in caplog.text


@pytest.mark.unit
def test_publication_uses_shared_20k_answers_and_does_not_mutate_base_for_new_models(tool, tmp_path):
    output = tmp_path / "sampled"
    tool.prepare(_config(tmp_path), output, mode="tier1_per_language", seed=17, languages=frozenset({"en"}))
    published = tmp_path / "published"
    kwargs = {"mount_root": "/fixture/data", "mode": "tier1_per_language", "language": "en"}
    tool.publish(output / "en", published, variant_suffix="", **kwargs)
    before = {path: path.read_bytes() for path in (published / "configs").glob("*.json")}
    tool.publish(output / "en", published, variant_suffix="qwen38_27b", **kwargs)
    assert all(path.read_bytes() == content for path, content in before.items())
    for size in ("2k", "5k", "10k", "20k"):
        overlay = json.loads(
            (published / "asr_nothink_qwen38_27b/configs" / f"asr_chunk_tier1_en_{size}.json").read_text()
        )
        base = json.loads((published / "configs" / f"asr_chunk_tier1_en_{size}.json").read_text())
        assert len(overlay["data_sources"]) == len(base["data_sources"])
        assert "answer" in overlay["chunk_type"]
        answers = [source["additional_data_path"]["answer"] for source in overlay["data_sources"]]
        assert len(set(answers)) == len(answers)
        assert all("-20k/source-" in answer for answer in answers)
        assert all("asr_nothink_qwen38_27b" in answer for answer in answers)
        for source in overlay["data_sources"]:
            index = int(Path(source["chunk_path"]).name.split("-")[-1])
            assert source["additional_data_path"]["answer"].endswith(f"/source-{index:03d}")
        assert (published / "asr_continuation_qwen38_27b" / f"specs-{size}.txt").is_file()
    with pytest.raises(FileExistsError):
        tool.publish(output / "en", published, variant_suffix="qwen38_27b", **kwargs)


def _records(path, extension, payloads):
    with path.open("wb") as handle:
        handle.write(extension.encode("ascii") + struct.pack("<I", len(payloads)))
        for index, payload in enumerate(payloads):
            handle.write(struct.pack("<II", index, len(payload)) + payload)


@pytest.mark.unit
def test_hindi_effective_hours_match_frame_lid_and_label_filtering(tool, tmp_path):
    features = [bytes(frames * 80 * 4) for frames in (100, 100, 4001, 100)]
    infos = [json.dumps({"lid_mix_label": value}).encode() for value in ("Native", "Other", "Mix", "Mix")]
    labels = [b"valid", b"valid", b"valid", b"x" * 801]
    _records(tmp_path / "chunk.feature", "feature", features)
    _records(tmp_path / "chunk.info", "info", infos)
    _records(tmp_path / "chunk.transcription", "transcription", labels)
    assert tool._hindi_seconds(tmp_path, "chunk", transcription_directory=tmp_path, count_tokens=len) == 1
    assert tool._hindi_seconds(tmp_path, "chunk", transcription_directory=tmp_path, count_tokens=lambda text: 2) == 2
    with (tmp_path / "chunk.info").open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(ValueError, match="Truncated"):
        tool._hindi_seconds(tmp_path, "chunk", transcription_directory=tmp_path, count_tokens=len)


@pytest.mark.unit
def test_hindi_requires_explicit_source_and_tokenizer(tool, tmp_path):
    with pytest.raises(ValueError, match="tokenizer"):
        tool.prepare({}, tmp_path / "sampled", mode="hindi", seed=17)
    with pytest.raises(ValueError, match="hindi-spec"):
        tool.prepare({}, tmp_path / "sampled", mode="hindi", seed=17, count_tokens=len)


@pytest.mark.unit
def test_missing_requested_languages_do_not_silently_shrink_sampling_scope(tool, tmp_path):
    with pytest.raises(ValueError, match="No eligible Tier1 data"):
        tool.prepare(_config(tmp_path), tmp_path / "sampled", mode="tier1_per_language", seed=17)


@pytest.mark.unit
def test_tier1_availability_is_indexed_once_per_directory(tool, tmp_path, monkeypatch):
    config = _config(tmp_path)
    scans = []
    scandir = tool.os.scandir

    def counted(directory):
        scans.append(directory)
        return scandir(directory)

    monkeypatch.setattr(tool.os, "scandir", counted)
    components = tool._components(config, mode="tier1_per_language", seed=17, hindi_spec=None, count_tokens=len)
    assert sum(len(component.chunks) for component in components) == 20
    assert scans == [tmp_path / "chunks-0", tmp_path / "chunks-1"]
