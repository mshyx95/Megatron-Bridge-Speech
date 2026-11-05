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

"""Compose, prepare and publish ELIFT data configs without submitting jobs or generating answers.

Ports the source sampler's weighted deterministic chunk ordering, Hindi effective-hour filters,
per-language availability checks and publisher's shared-20k, source-namespaced answer layout.
Run real-data preparation on the designated data-preparation host, not a development GPU host.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Callable, TypeAlias

import yaml
from omegaconf import OmegaConf


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


LOGGER = logging.getLogger(__name__)
JSONValue: TypeAlias = "None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]"
HOURS = (2000, 5000, 10000, 20000)
LANGUAGES = frozenset({"en", "de", "es", "fr", "it", "ja", "ko", "pt", "zh"})
INHOUSE_SLOTS = {
    "asr": {"english": 0, "multilingual": 1},
    "rewrite": {"english": 7, "multilingual": 8},
    "continuation": {"english": 12, "multilingual": 13},
}


def _json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(f"Not a JSON-compatible value: {type(value).__name__}")


def _mapping(value: JSONValue) -> dict[str, JSONValue]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping, got {type(value).__name__}")
    return value


def _list(value: JSONValue) -> list[JSONValue]:
    if not isinstance(value, list):
        raise TypeError(f"Expected a list, got {type(value).__name__}")
    return value


def _string(value: JSONValue) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected a string, got {type(value).__name__}")
    return value


def _number(value: JSONValue) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return float(value)


def _read_json(path: Path) -> dict[str, JSONValue]:
    return _mapping(_json_value(json.loads(path.read_text(encoding="utf-8"))))


def _write_json(path: Path, value: JSONValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_yaml(path: Path, value: dict[str, JSONValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)


def _load_config(path: Path) -> dict[str, JSONValue]:
    return _mapping(_json_value(OmegaConf.to_container(OmegaConf.load(path), resolve=True)))


def build_data_config(
    configs: Path,
    *,
    recipe: str,
    tokenizer: str,
    hindi: str | None = None,
    streaming_compression: bool | None = None,
) -> dict[str, JSONValue]:
    """Compose the shared mixture with model-independent inhouse and Hindi sources.

    Args:
        configs: Directory containing ``mixture.yaml`` and ``sources.yaml``.
        recipe: Verified inhouse recipe from ``sources.yaml``.
        tokenizer: Model tokenizer path or Hugging Face identifier.
        hindi: Optional independently selected Hindi tier.
        streaming_compression: Optional encoder/runtime override, not a data recipe.

    Returns:
        Standalone streaming config retaining the original sampling slots and weights.

    Raises:
        ValueError: A recipe or Hindi tier is unmapped, or the mixture contract is invalid.
        TypeError: A source mapping or runtime override has an invalid type.
    """
    if not tokenizer.strip():
        raise ValueError("A model tokenizer is required")
    if streaming_compression is not None and not isinstance(streaming_compression, bool):
        raise TypeError("streaming_compression must be a bool or None")
    config = _mapping(_json_value(yaml.safe_load((configs / "mixture.yaml").read_text())))
    sources = _mapping(_json_value(yaml.safe_load((configs / "sources.yaml").read_text())))
    recipes = _mapping(sources["recipes"])
    if recipe not in recipes:
        raise ValueError(f"Unmapped data recipe {recipe!r}; verified recipes: {', '.join(recipes)}")
    selected = _mapping(recipes[recipe])
    replacements = None if selected["inhouse"] is None else _mapping(selected["inhouse"])
    if replacements is not None and set(replacements) != set(INHOUSE_SLOTS):
        raise ValueError("Inhouse replacements must cover ASR, rewrite and continuation together")
    hindi_sources = _mapping(sources["hindi"])
    if hindi is not None and hindi not in hindi_sources:
        raise ValueError(f"Unmapped Hindi tier {hindi!r}; verified tiers: {', '.join(hindi_sources)}")
    additions = None if hindi is None else _mapping(hindi_sources[hindi])
    if additions is not None and set(additions) != set(INHOUSE_SLOTS):
        raise ValueError("Hindi must cover ASR, rewrite and continuation together")
    entries = _list(config["dataset"])
    if len(entries) != 17:
        raise ValueError("ELIFT mixture must retain its 17 sampling slots")
    for task, slots in INHOUSE_SLOTS.items():
        formatter = {"asr": "asr", "rewrite": "audio-response", "continuation": "audio-continuation"}[task]
        task_sources = None if replacements is None else _mapping(replacements[task])
        if task_sources is not None and set(task_sources) != set(slots):
            raise ValueError(f"{task} replacements must cover English and multilingual slots")
        for language, index in slots.items():
            entry = _mapping(entries[index])
            if _mapping(entry["transform_cls"])["prompt_formatter"] != formatter or _number(entry["weight"]) != 1.0:
                raise ValueError(f"Invalid {task} slot {index}: preserve the original formatter and 1:1:1 weights")
            if task_sources is not None:
                specs: list[JSONValue] = [_string(value) for value in _list(task_sources[language])]
                if not specs:
                    raise ValueError(f"Empty {task}/{language} inhouse sources")
                entry["dataset_spec"] = list(specs)
            if additions is not None and language == "multilingual":
                _list(entry["dataset_spec"]).append(_string(additions[task]))
    if not isinstance(selected["use_storage_api"], bool):
        raise TypeError("Recipe use_storage_api must be a bool")
    config["use_storage_api"] = selected["use_storage_api"]
    if "path_prefix_in_orange" in selected:
        for value in entries:
            _mapping(value)["path_prefix_in_orange"] = _string(selected["path_prefix_in_orange"])
    _mapping(config["tokenizer"])["pretrained_model_name_or_path"] = tokenizer
    if streaming_compression is not None:
        config["streaming_compression"] = streaming_compression
    OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    return config


def _write_data_config(path: Path, config: dict[str, JSONValue]) -> None:
    if path.exists():
        existing = _mapping(_json_value(yaml.safe_load(path.read_text())))
        if existing != config:
            raise ValueError(f"Existing data config {path} differs; use a fresh output directory")
        LOGGER.info("Reusing verified data config: %s", path)
        return
    _write_yaml(path, config)
    LOGGER.info("Wrote standalone data config: %s", path)


def _spec_weights(config: dict[str, JSONValue]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in _list(config["dataset"]):
        entry = _mapping(value)
        weight = _number(entry.get("weight", 1.0))
        for spec in _list(entry["dataset_spec"]):
            files: list[JSONValue]
            if isinstance(spec, str):
                files, ratio = [spec], 1.0
            else:
                reference = _mapping(spec)
                files_value = reference["file"]
                files = [files_value] if isinstance(files_value, str) else _list(files_value)
                ratio = _number(reference.get("sample_ratio", 1.0))
            if weight <= 0 or ratio <= 0:
                raise ValueError("ELIFT sampling weights and sample ratios must be positive")
            for file in files:
                path = _string(file)
                weights[path] = weights.get(path, 0.0) + weight * ratio
    return weights


def _language(spec: dict[str, JSONValue], source: dict[str, JSONValue], path: str) -> str:
    for value in (source.get("language"), spec.get("language")):
        if isinstance(value, str) and value:
            return re.split(r"[-_]", value.lower(), maxsplit=1)[0]
    for language in sorted(LANGUAGES):
        if re.search(rf"(^|[/_.-]){language}(?:[a-z]{{2}})?([/_-]|$)", path.lower()):
            return language
    return "unknown"


def _record(handle: BinaryIO) -> tuple[int, bytes] | None:
    header = handle.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise ValueError("Truncated speech-chunk record header")
    index, size = struct.unpack("<II", header)
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError(f"Truncated speech-chunk record {index}")
    return index, payload


def _hindi_seconds(
    directory: Path,
    name: str,
    *,
    transcription_directory: Path,
    count_tokens: Callable[[str], int],
) -> float:
    seconds = 0.0
    with (
        (directory / f"{name}.feature").open("rb") as features,
        (directory / f"{name}.info").open("rb") as infos,
        (transcription_directory / f"{name}.transcription").open("rb") as transcriptions,
    ):
        for handle, extension in ((features, "feature"), (infos, "info"), (transcriptions, "transcription")):
            if handle.read(len(extension)).lower() != extension.encode("ascii") or len(handle.read(4)) != 4:
                raise ValueError(f"Invalid {extension} file header in Hindi chunk {name}")
        index = 0
        while True:
            records = [_record(handle) for handle in (features, infos, transcriptions)]
            if all(record is None for record in records):
                break
            if any(record is None or record[0] != index for record in records):
                raise ValueError(f"Unaligned records in Hindi chunk {name} at {index}")
            feature, info, text = records
            if feature is None or info is None or text is None:
                raise AssertionError("Record alignment was not checked")
            frames, remainder = divmod(len(feature[1]), 80 * 4)
            if remainder:
                raise ValueError(f"Hindi feature {name}#{index} is not float32 x 80")
            metadata = _mapping(_json_value(json.loads(info[1])))
            labels = text[1]
            if (
                frames <= 4000
                and metadata.get("lid_mix_label") in ("Mix", "Native")
                and labels
                and (len(labels) <= 800 or count_tokens(labels.decode("utf-8")) <= 800)
            ):
                seconds += frames / 100.0
            index += 1
    return seconds


@dataclass
class _Chunk:
    descriptor: dict[str, JSONValue]
    seconds: float
    shuffle_key: str


@dataclass
class _Component:
    spec_path: str
    source_index: int
    spec: dict[str, JSONValue]
    source: dict[str, JSONValue]
    manifest: dict[str, JSONValue]
    language: str
    weight: float
    chunks: list[_Chunk]


def _chunk_available(directory: Path, name: str, extension: str, cache: dict[Path, dict[str, frozenset[str]]]) -> bool:
    if "/" in name:
        return (directory / f"{name}.{extension}").is_file()
    if directory not in cache:
        names: dict[str, set[str]] = {"feature": set(), "transcription": set()}
        with os.scandir(directory) as entries:
            for entry in entries:
                for suffix in names:
                    if entry.name.endswith("." + suffix):
                        names[suffix].add(entry.name[: -(len(suffix) + 1)])
        cache[directory] = {suffix: frozenset(values) for suffix, values in names.items()}
    return name in cache[directory][extension]


def _components(
    config: dict[str, JSONValue],
    *,
    mode: str,
    seed: int,
    hindi_spec: Path | None,
    count_tokens: Callable[[str], int],
) -> list[_Component]:
    if mode == "hindi" and hindi_spec is None:
        raise ValueError("Hindi sampling requires an explicit --hindi-spec")
    weights = {str(hindi_spec): 1.0} if mode == "hindi" else _spec_weights(config)
    result = []
    available: dict[Path, dict[str, frozenset[str]]] = {}
    for spec_path, weight in weights.items():
        spec = _read_json(Path(spec_path))
        spec_id = Path(spec_path).stem + "-" + hashlib.sha256(spec_path.encode()).hexdigest()[:10]
        for source_index, value in enumerate(_list(spec["data_sources"])):
            source = _mapping(value)
            language = _language(spec, source, spec_path)
            if mode != "hindi" and language not in LANGUAGES:
                continue
            manifest_path = Path(_string(source["manifest_file"]))
            manifest = _read_json(manifest_path)
            directory = Path(_string(source.get("chunk_path", str(manifest_path.parent))))
            transcription_directory = Path(_string(source.get("trans_path", str(directory))))
            chunks = []
            missing = 0
            for descriptor_value in _list(manifest["fileInfo"]):
                descriptor = _mapping(descriptor_value)
                name = _string(descriptor["name"])
                if mode == "tier1_per_language" and not (
                    _chunk_available(directory, name, "feature", available)
                    and _chunk_available(transcription_directory, name, "transcription", available)
                ):
                    missing += 1
                    continue
                seconds = (
                    _hindi_seconds(
                        directory, name, transcription_directory=transcription_directory, count_tokens=count_tokens
                    )
                    if mode == "hindi"
                    else _number(descriptor["duration"])
                )
                if seconds <= 0:
                    continue
                shuffle = hashlib.sha256(f"{seed}\0{spec_id}:{source_index}\0{name}".encode()).hexdigest()
                chunks.append(_Chunk(descriptor, seconds, shuffle))
            chunks.sort(key=lambda chunk: chunk.shuffle_key)
            if missing:
                LOGGER.warning("Excluding %d incomplete %s chunks from %s", missing, language, spec_path)
            result.append(_Component(spec_path, source_index, spec, source, manifest, language, weight, chunks))
    if not result or not any(component.chunks for component in result):
        raise ValueError(f"No eligible {mode} chunks")
    return result


def _nested_order(components: list[_Component]) -> list[tuple[int, _Chunk]]:
    ranked = []
    for component_index, component in enumerate(components):
        distribution = sum(chunk.seconds for chunk in component.chunks) * component.weight
        cumulative = 0.0
        for chunk in component.chunks:
            priority = (cumulative + chunk.seconds / 2) / distribution
            ranked.append((priority, chunk.shuffle_key, component_index, chunk))
            cumulative += chunk.seconds
    ranked.sort(key=lambda row: (row[0], row[1]))
    return [(index, chunk) for _, _, index, chunk in ranked]


def _replace(value: JSONValue, mapping: dict[str, str]) -> JSONValue:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, mapping) for key, item in value.items()}
    return value


def _source_namespace(source: dict[str, JSONValue]) -> str:
    name = Path(_string(source["manifest_file"])).stem
    if not re.fullmatch(r"source-\d{3,}", name):
        raise ValueError(f"Sampled manifest lacks a stable source namespace: {name}")
    return name


def _sample_config(config: dict[str, JSONValue], mapping: dict[str, str]) -> dict[str, JSONValue]:
    result = copy.deepcopy(config)
    entries: list[JSONValue] = []
    for value in _list(result["dataset"]):
        entry = _mapping(value)
        specs: list[JSONValue] = []
        for spec in _list(entry["dataset_spec"]):
            if isinstance(spec, str):
                if spec in mapping:
                    specs.append(mapping[spec])
            else:
                reference = _mapping(spec)
                files = reference["file"]
                paths = [files] if isinstance(files, str) else _list(files)
                retained: list[JSONValue] = [mapping[_string(path)] for path in paths if _string(path) in mapping]
                if retained:
                    reference["file"] = retained[0] if isinstance(files, str) else retained
                    specs.append(reference)
        if specs:
            entry["dataset_spec"] = specs
            entries.append(entry)
    result["dataset"] = entries
    return result


def prepare(
    config: dict[str, JSONValue],
    output: Path,
    *,
    mode: str,
    seed: int,
    targets: tuple[int, ...] = HOURS,
    languages: frozenset[str] | None = None,
    hindi_spec: Path | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> None:
    """Write nested manifests and configs; refuse overwrites and fail explicitly on insufficient hours."""
    if mode not in {"hindi", "tier1", "tier1_per_language"}:
        raise ValueError(f"Unsupported ELIFT sampling mode: {mode}")
    if not targets or any(target not in HOURS for target in targets):
        raise ValueError(f"ELIFT targets must be drawn from {HOURS}")
    if languages is not None and (mode != "tier1_per_language" or not languages or not languages <= LANGUAGES):
        raise ValueError("Explicit Tier1 languages require per-language sampling and valid language codes")
    if mode == "hindi" and count_tokens is None:
        raise ValueError("Hindi effective-hour filtering requires a tokenizer")
    components = _components(config, mode=mode, seed=seed, hindi_spec=hindi_spec, count_tokens=count_tokens or len)
    groups = sorted(languages or LANGUAGES) if mode == "tier1_per_language" else [mode]
    if mode == "tier1_per_language":
        missing = set(groups) - {component.language for component in components if component.chunks}
        if missing:
            raise ValueError(f"No eligible Tier1 data for {', '.join(sorted(missing))}")
    output.mkdir(parents=True, exist_ok=False)
    incomplete = []
    for group in groups:
        selected_components = (
            [c for c in components if c.language == group] if mode == "tier1_per_language" else components
        )
        order = _nested_order(selected_components)
        summary: dict[str, JSONValue] = {"mode": mode, "seed": seed, "targets": {}}
        group_output = output / group if mode == "tier1_per_language" else output
        for target in sorted(set(targets)):
            chosen: dict[int, list[_Chunk]] = {}
            seconds = 0.0
            for component_index, chunk in order:
                if seconds >= target * 3600:
                    break
                chosen.setdefault(component_index, []).append(chunk)
                seconds += chunk.seconds
            target_root = group_output / f"{target // 1000}k"
            specs: dict[str, dict[str, JSONValue]] = {}
            mapping = {}
            for component_index, chunks in chosen.items():
                component = selected_components[component_index]
                stem = (
                    Path(component.spec_path).stem
                    + "-"
                    + hashlib.sha256(component.spec_path.encode()).hexdigest()[:10]
                )
                manifest = copy.deepcopy(component.manifest)
                names = {_string(chunk.descriptor["name"]) for chunk in chunks}
                manifest["fileInfo"] = [
                    c.descriptor for c in component.chunks if _string(c.descriptor["name"]) in names
                ]
                manifest_path = target_root / "manifests" / stem / f"source-{component.source_index:03d}.json"
                _write_json(manifest_path, manifest)
                source = copy.deepcopy(component.source)
                source["manifest_file"] = str(manifest_path)
                if component.spec_path not in specs:
                    specs[component.spec_path] = {**copy.deepcopy(component.spec), "data_sources": []}
                _list(specs[component.spec_path]["data_sources"]).append(source)
                mapping[component.spec_path] = str(target_root / "dataspecs" / f"{stem}.json")
            for path, spec in specs.items():
                _write_json(Path(mapping[path]), spec)
            source_mode = "tier1" if mode == "tier1_per_language" else mode
            _write_yaml(
                target_root / "configs" / f"data-v3.5-{source_mode}-{target // 1000}k.yaml",
                _sample_config(config, mapping),
            )
            complete = seconds >= target * 3600
            _mapping(summary["targets"])[str(target)] = {
                "requested_hours": target,
                "selected_hours": seconds / 3600,
                "complete": complete,
            }
            LOGGER.info("%s %dh: %.3fh selected, complete=%s", group, target, seconds / 3600, complete)
            if not complete:
                incomplete.append(f"{group}/{target}")
        _write_json(group_output / "summary.json", summary)
    if incomplete:
        raise ValueError(f"Insufficient hours; retained partial evidence for {', '.join(incomplete)}")


def publish(
    source: Path, output: Path, *, mount_root: str, mode: str, language: str | None, variant_suffix: str
) -> None:
    """Publish all four sizes with immutable base manifests and model-specific, source-namespaced answers."""
    if mode not in {"hindi", "tier1", "tier1_per_language"} or not Path(mount_root).is_absolute():
        raise ValueError("Publication requires a valid sampling mode and an absolute mounted data root")
    if mode == "tier1_per_language" and language not in LANGUAGES:
        raise ValueError("Per-language publication requires a Tier1 language")
    if language is not None and mode != "tier1_per_language":
        raise ValueError("A publication language requires tier1_per_language mode")
    if not re.fullmatch(r"[a-zA-Z0-9_-]*", variant_suffix):
        raise ValueError("Unsafe answer variant suffix")
    summary = _read_json(source / "summary.json")
    for hours in HOURS:
        target = _mapping(_mapping(summary["targets"])[str(hours)])
        if target["complete"] is not True:
            raise ValueError(f"Cannot publish incomplete {hours}h sample")
    if not variant_suffix:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("Publish base manifests before adding a model-specific answer variant")
    mounted = Path(mount_root)
    source_mode = "tier1" if mode == "tier1_per_language" else mode
    for hours in HOURS:
        size = f"{hours // 1000}k"
        config = _load_config(source / size / "configs" / f"data-v3.5-{source_mode}-{size}.yaml")
        original_paths = sorted((source / size / "dataspecs").glob("*.json"))
        if not original_paths:
            raise ValueError(f"No sampled dataspecs for {size}")
        base_mapping: dict[str, str] = {}
        variant_mappings: dict[str, dict[str, str]] = {}
        merged: dict[str, list[JSONValue]] = {}
        chunk_types_for_merge: list[JSONValue] | None = None
        for original in original_paths:
            stem = f"{original.stem}-{size}"
            destination = output / "configs" / f"{stem}.json"
            spec = _read_json(destination) if variant_suffix else _read_json(original)
            if language:
                chunk_types = _list(spec["chunk_type"])
                if chunk_types_for_merge is not None and chunk_types != chunk_types_for_merge:
                    raise ValueError("Cannot merge per-language dataspecs with different chunk_type values")
                chunk_types_for_merge = list(chunk_types)
            if not variant_suffix:
                for value in _list(spec["data_sources"]):
                    item = _mapping(value)
                    manifest = _read_json(Path(_string(item["manifest_file"])))
                    relative = Path("manifests") / stem / f"{_source_namespace(item)}.json"
                    _write_json(output / relative, manifest)
                    item["manifest_file"] = str(mounted / relative)
                _write_json(destination, spec)
            base_mapping[str(original)] = str(mounted / "configs" / f"{stem}.json")
            variants = ["base", "asr_nothink", "asr_continuation"]
            for variant in variants:
                clone = copy.deepcopy(spec)
                if variant != "base":
                    variant = f"{variant}_{variant_suffix}" if variant_suffix else variant
                    chunk_types = _list(clone["chunk_type"])
                    if "answer" not in chunk_types:
                        chunk_types.append("answer")
                    for value in _list(clone["data_sources"]):
                        item = _mapping(value)
                        additional = _mapping(item.get("additional_data_path") or item.get("additional_path") or {})
                        additional["answer"] = str(
                            mounted / variant / f"{original.stem}-20k" / _source_namespace(item)
                        )
                        item["additional_data_path"] = additional
                    _write_json(output / variant / "configs" / f"{stem}.json", clone)
                    variant_mappings.setdefault(variant, {})[str(original)] = str(
                        mounted / variant / "configs" / f"{stem}.json"
                    )
                if language:
                    spec_language = clone.get("language")
                    if (
                        spec_language is not None
                        and re.split(r"[-_]", _string(spec_language).lower(), maxsplit=1)[0] != language
                    ):
                        raise ValueError(f"Cannot merge {spec_language} dataspecs into {language}")
                    merged.setdefault(variant, []).extend(_list(clone["data_sources"]))
        for variant, sources in merged.items():
            prefix = Path() if variant == "base" else Path(variant)
            merged_spec = {**spec, "language": language, "data_sources": sources}
            if variant != "base":
                merged_spec["chunk_type"] = [
                    "answer",
                    *[item for item in _list(spec["chunk_type"]) if item != "answer"],
                ]
            if variant != "base" or not variant_suffix:
                _write_json(output / prefix / "configs" / f"asr_chunk_tier1_{language}_{size}.json", merged_spec)
        if not variant_suffix:
            _write_yaml(output / "configs" / f"data-v3.5-{mode}-{size}.yaml", _mapping(_replace(config, base_mapping)))
        for variant, mapping in variant_mappings.items():
            _write_yaml(
                output / variant / "configs" / f"data-v3.5-{mode}-{variant}-{size}.yaml",
                _mapping(_replace(config, mapping)),
            )
            with (output / variant / f"specs-{size}.txt").open("x", encoding="utf-8") as handle:
                handle.write("\n".join(base_mapping.values()) + "\n")
    receipt: dict[str, JSONValue] = {
        "sampling_summary": summary,
        "mount_root": mount_root,
        "mode": mode,
        "language": language,
        "variant_suffix": variant_suffix,
    }
    _write_json(output / (f"summary-{variant_suffix}.json" if variant_suffix else "summary.json"), receipt)


def main() -> None:
    """Dispatch data-only sampling or publication, without training or paid generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    composition = subparsers.add_parser("config", help="Render a model-independent recipe as a standalone data YAML")
    composition.add_argument("--config-root", type=Path, required=True)
    composition.add_argument("--recipe", required=True)
    composition.add_argument("--tokenizer", required=True)
    composition.add_argument("--hindi")
    composition.add_argument("--streaming-compression", action=argparse.BooleanOptionalAction, default=None)
    composition.add_argument("--output", type=Path, required=True)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--source-config", type=Path, required=True)
    sample.add_argument("--output-root", type=Path, required=True)
    sample.add_argument("--mode", choices=["hindi", "tier1", "tier1_per_language"], required=True)
    sample.add_argument("--targets", type=int, nargs="+", choices=HOURS, default=list(HOURS))
    sample.add_argument("--seed", type=int, default=20260921)
    sample.add_argument("--languages", nargs="+", choices=sorted(LANGUAGES))
    sample.add_argument("--hindi-spec", type=Path)
    sample.add_argument("--tokenizer")
    publication = subparsers.add_parser("publish")
    publication.add_argument("--source-root", type=Path, required=True)
    publication.add_argument("--output-root", type=Path, required=True)
    publication.add_argument("--mount-root", required=True)
    publication.add_argument("--mode", choices=["hindi", "tier1", "tier1_per_language"], required=True)
    publication.add_argument("--language", choices=sorted(LANGUAGES))
    publication.add_argument("--variant-suffix", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "config":
        _write_data_config(
            args.output,
            build_data_config(
                args.config_root,
                recipe=args.recipe,
                tokenizer=args.tokenizer,
                hindi=args.hindi,
                streaming_compression=args.streaming_compression,
            ),
        )
        return
    if args.command == "publish":
        publish(
            args.source_root,
            args.output_root,
            mount_root=args.mount_root,
            mode=args.mode,
            language=args.language,
            variant_suffix=args.variant_suffix,
        )
        return
    if args.mode == "hindi" and (args.hindi_spec is None or args.tokenizer is None):
        parser.error("Hindi sampling requires --hindi-spec and --tokenizer")

    def count_tokens(text: str) -> int:
        from transformers import AutoTokenizer

        if not tokenizers:
            tokenizers.append(AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True))
        return len(tokenizers[0].encode(text, add_special_tokens=False))

    tokenizers: list[PreTrainedTokenizerBase] = []
    prepare(
        _load_config(args.source_config),
        args.output_root,
        mode=args.mode,
        seed=args.seed,
        targets=tuple(args.targets),
        languages=frozenset(args.languages) if args.languages else None,
        hindi_spec=args.hindi_spec,
        count_tokens=count_tokens,
    )


if __name__ == "__main__":
    main()
