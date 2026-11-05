import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[4]
CONFIGS = ROOT / "projects/ELIFT"
CATALOG = yaml.safe_load((CONFIGS / "catalog.yaml").read_text())
LAUNCH = yaml.safe_load((CONFIGS / "launch.yaml").read_text())


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
@pytest.mark.parametrize(
    "stage,digest",
    [
        # Semantic hashes of the eight standalone snapshots before consolidation, not credentials.
        ("b1", "67b48f373fe3ffcc308fd73de1deff0dfb3dddec931a15f645704f1f6abf347c"),  # pragma: allowlist secret
        ("b2", "440db3ef61e4c869f9d4b6b0b31e8f198cff909c177d4358d64dc13abde26c51"),  # pragma: allowlist secret
        ("b3", "6bd174423c8a7026fbcaa1df9af5a9139436d6af7a24fd9d9072e09361dd4382"),  # pragma: allowlist secret
        (
            "e1r-hindi-2k",
            "99ec8359ef87d83e14d1a05331b5f117b716f613bd9898fc2d571a063aed54c7",  # pragma: allowlist secret
        ),
        ("e1r-2609", "1652200a77fd2c2c5ca813e5c6aa2d4052561e001d129717d6cab6b2a93e425f"),  # pragma: allowlist secret
        (
            "e1r-2609-hindi-2k",
            "77ad31904767c74c50ed91c7adf7b99fdaa82bf9f2a095c8241f791cc02b2a9e",  # pragma: allowlist secret
        ),
        (
            "new-tier1-hindi-2k-e1r",
            "cd1659b8d4f38cb15a20dad949c7946b0439058608a5e19232fe395d578c440c",  # pragma: allowlist secret
        ),
        (
            "new-tier1-hindi-5k-e1r-8k",
            "4c2bae374bc171f3735d5f4a5b26791ada8e6f5048ce82beadabc419b0aa7b6a",  # pragma: allowlist secret
        ),
    ],
)
def test_composed_configs_match_every_historical_snapshot(tool, stage, digest):
    profile = CATALOG["stages"][stage]
    config = tool.build_data_config(
        CONFIGS / "data",
        recipe=profile["data_recipe"],
        tokenizer=LAUNCH["base_models"][profile["language_model"]],
        hindi=profile.get("data_hindi"),
        streaming_compression=profile.get("data_streaming_compression"),
    )
    resolved = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    assert hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest() == digest


@pytest.mark.unit
@pytest.mark.parametrize("recipe", ["base", "new-2k", "new-5k"])
@pytest.mark.parametrize("hindi", ["2k", "5k"])
def test_hindi_is_an_independent_addition_in_all_three_tasks(tool, recipe, hindi):
    kwargs = {"recipe": recipe, "tokenizer": "/fixture/tokenizer"}
    original = tool.build_data_config(CONFIGS / "data", **kwargs)
    added = tool.build_data_config(CONFIGS / "data", hindi=hindi, **kwargs)
    hindi_sources = yaml.safe_load((CONFIGS / "data/sources.yaml").read_text())["hindi"][hindi]
    for task, slots in tool.INHOUSE_SLOTS.items():
        index = slots["multilingual"]
        expected = [*original["dataset"][index]["dataset_spec"], hindi_sources[task]]
        assert added["dataset"][index]["dataset_spec"] == expected
        added["dataset"][index]["dataset_spec"].pop()
    assert added == original


@pytest.mark.unit
@pytest.mark.parametrize("recipe,english", [("new-2k", "5k"), ("new-5k", "10k")])
def test_new_data_replaces_only_inhouse_and_keeps_english_tier_mapping(tool, recipe, english):
    base = tool.build_data_config(CONFIGS / "data", recipe="base", tokenizer="/fixture/tokenizer")
    new = tool.build_data_config(CONFIGS / "data", recipe=recipe, tokenizer="/fixture/tokenizer")
    for slots in tool.INHOUSE_SLOTS.values():
        assert (
            new["dataset"][slots["english"]]["dataset_spec"][-1]
            == base["dataset"][slots["english"]]["dataset_spec"][-1]
        )
        assert len(new["dataset"][slots["multilingual"]]["dataset_spec"]) == 8
    assert new["dataset"][0]["dataset_spec"][0].endswith(f"asr_chunk_tier1_en_{english}.json")
    inhouse_slots = {index for slots in tool.INHOUSE_SLOTS.values() for index in slots.values()}
    for index, (old, replacement) in enumerate(zip(base["dataset"], new["dataset"])):
        replacement.pop("path_prefix_in_orange")
        if index in inhouse_slots:
            replacement["dataset_spec"] = old["dataset_spec"]
        assert replacement == old
    assert [entry["weight"] for entry in new["dataset"]] == [1.0] * 17


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"recipe": "new-10k"}, "Unmapped data recipe"),
        ({"recipe": "new-20k"}, "Unmapped data recipe"),
        ({"hindi": "20k"}, "Unmapped Hindi tier"),
        ({"tokenizer": ""}, "tokenizer is required"),
        ({"streaming_compression": "false"}, "must be a bool"),
    ],
)
def test_unmapped_sources_and_invalid_model_settings_fail_explicitly(tool, overrides, message):
    kwargs = {"recipe": "base", "tokenizer": "/fixture/tokenizer", **overrides}
    with pytest.raises((ValueError, TypeError), match=message):
        tool.build_data_config(CONFIGS / "data", **kwargs)


@pytest.mark.unit
def test_malformed_mixtures_and_partial_task_replacements_are_rejected(tool, tmp_path):
    configs = tmp_path / "data"
    configs.mkdir()
    for name in ("mixture.yaml", "sources.yaml"):
        shutil.copyfile(CONFIGS / "data" / name, configs / name)
    mixture = yaml.safe_load((configs / "mixture.yaml").read_text())
    mixture["dataset"].pop()
    (configs / "mixture.yaml").write_text(yaml.safe_dump(mixture))
    with pytest.raises(ValueError, match="17 sampling slots"):
        tool.build_data_config(configs, recipe="base", tokenizer="/fixture/tokenizer")
    shutil.copyfile(CONFIGS / "data/mixture.yaml", configs / "mixture.yaml")
    sources = yaml.safe_load((configs / "sources.yaml").read_text())
    del sources["recipes"]["new-2k"]["inhouse"]["continuation"]
    (configs / "sources.yaml").write_text(yaml.safe_dump(sources))
    with pytest.raises(ValueError, match="ASR, rewrite and continuation together"):
        tool.build_data_config(configs, recipe="new-2k", tokenizer="/fixture/tokenizer")


@pytest.mark.unit
@pytest.mark.parametrize("field,value", [("weight", 0.5), ("weight", True), ("formatter", "audio-response")])
def test_composition_rejects_changed_task_weights_and_reordered_slots(tool, tmp_path, field, value):
    configs = tmp_path / "data"
    configs.mkdir()
    shutil.copyfile(CONFIGS / "data/sources.yaml", configs / "sources.yaml")
    mixture = yaml.safe_load((CONFIGS / "data/mixture.yaml").read_text())
    entry = mixture["dataset"][0]
    if field == "formatter":
        entry["transform_cls"]["prompt_formatter"] = value
    else:
        entry["weight"] = value
    (configs / "mixture.yaml").write_text(yaml.safe_dump(mixture))
    with pytest.raises(ValueError):
        tool.build_data_config(configs, recipe="base", tokenizer="/fixture/tokenizer")


@pytest.mark.unit
def test_config_cli_is_reusable_but_refuses_a_different_tokenizer(tool, tmp_path):
    output = tmp_path / "data.yaml"
    command = [
        sys.executable,
        str(ROOT / "examples/models/speech_cascades/prepare_elift_data.py"),
        "config",
        "--config-root",
        str(CONFIGS / "data"),
        "--recipe",
        "base",
        "--tokenizer",
        "/fixture/model-a",
        "--output",
        str(output),
    ]
    for _ in range(2):
        subprocess.run(command, check=True, capture_output=True, text=True)
    command[command.index("--tokenizer") + 1] = "/fixture/model-b"
    with pytest.raises(subprocess.CalledProcessError) as error:
        subprocess.run(command, check=True, capture_output=True, text=True)
    assert "differs; use a fresh output directory" in error.value.stderr
    assert yaml.safe_load(output.read_text())["tokenizer"]["pretrained_model_name_or_path"] == "/fixture/model-a"


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
