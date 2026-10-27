"""Dependency-light regressions for the vendored storage/cache contract.

Execute the production definitions with a fake Azure transport, without importing the
model registries or requiring a GPU training environment.
"""

import ast
import bisect
import copy
import dataclasses
import io
import itertools
import json
import logging
import os
import pickle
import sys
import threading
import time
import typing
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest


_ROOT = Path(__file__).resolve().parents[4]
_PIPELINE = _ROOT / "src/megatron/bridge/data/speech_chunk_streaming"


def _definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def reader():
    namespace = {
        **vars(typing),
        "os": os,
        "io": io,
        "json": json,
        "JSONDecodeError": json.JSONDecodeError,
        "logging": logging,
        "logger": logging.getLogger(__name__),
        "Path": Path,
        "copy": copy,
        "pickle": pickle,
        "uuid": uuid,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "as_completed": as_completed,
        "backoff": SimpleNamespace(on_exception=lambda *args, **kwargs: lambda function: function, expo=None),
        "np": SimpleNamespace(
            searchsorted=lambda values, value, side="left": (
                bisect.bisect_right(values, value) if side == "right" else bisect.bisect_left(values, value)
            ),
            cumsum=lambda values: list(itertools.accumulate(values)),
            concatenate=lambda values: sum((list(value) for value in values), []),
        ),
        "dist": SimpleNamespace(get_rank=lambda: 0),
        "_get_env_int": lambda name, default: int(os.environ.get(name, default)),
    }
    namespace["KenslerPermutation"] = lambda length, seed: list(range(length))
    return _definitions(
        _PIPELINE / "blob_chunk_alm/blobchunk.py",
        [
            "DataParser",
            "DatasetSpec",
            "_ChunkIndexPermutation",
            "_BlobChunkIterableDatasetIterator",
            "_ChunkSpillError",
            "_SpilledChunkSample",
        ],
        namespace,
    )


def _spec(reader, chunk_paths=None):
    spec = object.__new__(reader["DatasetSpec"])
    spec.use_storage_api = True
    spec.path_prefix_in_orange = None
    spec.total_chunks = 1
    spec.accum_chunks = [1]
    spec.chunk_sizes = [2]
    spec.is_tsv = False
    spec.chunk_processing_pipe = None
    spec.data_sources = [
        {
            "manifest_file": "cached/manifest.json",
            "chunk_type": ["audio", "transcription", "answer"],
            "chunk_path": "/datablob/original/audio",
            "trans_path": "/datablob/original/transcriptions",
            "additional_data_path": {"answer": "https://rewrite.blob.core.windows.net/answers/set"},
            "language": "en",
            **(chunk_paths or {}),
        }
    ]
    spec.load_manifest_file = lambda source: [{"name": "chunk0", "count": 2}]
    return spec


@pytest.mark.unit
@pytest.mark.parametrize(
    "path,expected",
    [
        ("/datablob/original/audio/", "original/audio"),
        ("/datablob1/original/audio", "original/audio"),
        ("original/audio", "original/audio"),
        ("https://rewrite.blob.core.windows.net/answers/set/", "https://rewrite.blob.core.windows.net/answers/set"),
        ("az://rewrite/answers/set/", "az://rewrite/answers/set"),
    ],
)
def test_storage_paths_keep_account_container_and_relative_keys(reader, path, expected):
    spec = _spec(reader)
    assert spec._resolve_chunk_path(path) == expected
    spec.path_prefix_in_orange = "az://mounted/data"
    if spec._is_remote_url(path):
        assert spec._resolve_chunk_path(path) == expected


@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True])
def test_binary_chunks_keep_separate_audio_transcript_answer_storage(reader, streaming):
    spec = _spec(reader)
    reads = []

    def read(blob_name, chunk_type, chunk_size):
        reads.append(blob_name)
        return [f"{chunk_type}-{i}" for i in range(chunk_size)]

    spec.load_chunk_with_chunkname = read
    if streaming:
        namespace = _definitions(
            _PIPELINE / "blob_chunk_alm/streaming_blobchunk.py",
            ["DatasetSpec"],
            {**reader, "BaseDatasetSpec": reader["DatasetSpec"]},
        )
        samples, _ = namespace["DatasetSpec"]._load_binary_chunk(spec, spec.data_sources[0], 0)
    else:
        samples = spec.load_chunk(0)
    assert reads == [
        "original/audio/chunk0.audio",
        "original/transcriptions/chunk0.transcription",
        "https://rewrite.blob.core.windows.net/answers/set/chunk0.answer",
    ]
    assert [sample["answer"] for sample in samples] == ["answer-0", "answer-1"]


@pytest.mark.unit
def test_raw_audio_storage_preserves_https_target(reader):
    spec = _spec(reader)
    reads = []
    spec.az_downloader = SimpleNamespace(
        download_to_buffer=lambda path, concurrency: reads.append(path) or io.BytesIO()
    )
    reader["sf"] = SimpleNamespace(read=lambda stream: ("waveform", 16000))
    url = "https://another.blob.core.windows.net/audio/a.wav"
    assert spec.safe_read(url) == ("waveform", 16000)
    assert reads == [url]


@pytest.fixture
def downloader():
    class Container:
        def __init__(self, account_url, container_name, **kwargs):
            self.account_url = account_url
            self.container_name = container_name
            self.reads = []

        def download_blob(self, blob_name, **kwargs):
            self.reads.append((blob_name, kwargs["offset"]))
            return SimpleNamespace(chunks=lambda **kwargs: iter([b"payload"]))

    namespace = {
        **vars(typing),
        "threading": threading,
        "os": os,
        "io": io,
        "time": time,
        "LOG": logging.getLogger(__name__),
        "urlparse": urlparse,
        "unquote": unquote,
        "ContainerClient": Container,
        "ManagedIdentityCredential": lambda client_id: object(),
        "AzureDownloaderConfig": object,
        "ResourceNotFoundError": type("ResourceNotFoundError", (Exception,), {}),
    }
    cls = _definitions(_PIPELINE / "blob_chunk_alm/az_downloader.py", ["AzureBlobDownloader"], namespace)[
        "AzureBlobDownloader"
    ]
    config = SimpleNamespace(
        account_url="https://original.blob.core.windows.net",
        container_name="data",
        uai_client_id_env="TEST_STORAGE_CLIENT_ID",
        retry_total=6,
        retry_connect=6,
        retry_read=6,
        retry_status=6,
        max_single_get_size=32 * 1024 * 1024,
        max_chunk_get_size=4 * 1024 * 1024,
        max_attempts=2,
        max_concurrency=8,
        request_timeout_s=5,
        logging_enable=False,
        chunk_size=4096,
        stall_timeout_s=10,
        log_every_s=3,
    )
    return cls(config)


@pytest.mark.unit
def test_downloader_routes_and_reuses_per_container_clients(downloader, monkeypatch):
    monkeypatch.setenv("TEST_STORAGE_CLIENT_ID", "test-identity")
    url = "https://rewrite.blob.core.windows.net/answers/set/a%20b.answer"
    assert downloader._download_bytes_until_eof_resumable(url) == b"payload"
    remote, blob_name = downloader._resolve_blob(url)
    assert (remote.account_url, remote.container_name, blob_name) == (
        "https://rewrite.blob.core.windows.net",
        "answers",
        "set/a b.answer",
    )
    assert remote.reads == [("set/a b.answer", 0)]
    assert downloader._resolve_blob("az://rewrite/answers/set/other.answer")[0] is remote
    assert downloader._resolve_blob("original/audio/chunk.audio") == (
        downloader._container,
        "original/audio/chunk.audio",
    )


@pytest.mark.unit
def test_downloader_container_cache_is_thread_safe(downloader, monkeypatch):
    monkeypatch.setenv("TEST_STORAGE_CLIENT_ID", "test-identity")
    downloader._ensure_client()
    created = []
    original = downloader._create_container_client

    def create(account, container):
        time.sleep(0.01)
        created.append((account, container))
        return original(account, container)

    monkeypatch.setattr(downloader, "_create_container_client", create)
    with ThreadPoolExecutor(max_workers=4) as executor:
        clients = list(
            executor.map(lambda _: downloader._resolve_blob("https://other.blob.core.windows.net/c/x")[0], range(8))
        )
    assert all(client is clients[0] for client in clients)
    assert len(created) == 1


@pytest.mark.unit
@pytest.mark.parametrize("parallel", [False, True])
def test_disk_spill_preserves_sample_order_and_releases_consumed_slots(reader, monkeypatch, tmp_path, parallel):
    monkeypatch.setenv("BLOBCHUNK_SPILL_DIR", str(tmp_path))
    spec = _spec(reader)
    spec.chunk_sizes = [2]
    spec.load_chunk = lambda index: [{"id": 0, "payload": b"a" * 4096}, {"id": 1, "payload": b"b" * 4096}]
    iterator = reader["_BlobChunkIterableDatasetIterator"](spec, 1, 0, 0, 1, 0, 0, parallel, False)
    assert next(iterator)["id"] == 0
    assert iterator.example_buffer[0] is None
    spilled = iterator.example_buffer[1]
    assert isinstance(spilled, reader["_SpilledChunkSample"])
    assert next(iterator)["id"] == 1
    assert iterator.example_buffer == [None, None]
    assert list(tmp_path.iterdir()) == []
    spilled.stream.close()


@pytest.mark.unit
def test_resume_releases_already_consumed_spilled_samples(reader, monkeypatch, tmp_path):
    monkeypatch.setenv("BLOBCHUNK_SPILL_DIR", str(tmp_path))
    spec = _spec(reader)
    spec.chunk_sizes = [3]
    spec.load_chunk = lambda index: [{"id": 0}, {"id": 1}, {"id": 2}]
    iterator = reader["_BlobChunkIterableDatasetIterator"](spec, 1, 0, 1, 1, 0, 0, True, False)
    assert next(iterator) == {"id": 1}
    assert iterator.example_buffer[:2] == [None, None]
    assert next(iterator) == {"id": 2}


@pytest.mark.unit
def test_disk_cache_failure_surfaces_instead_of_skipping_data(reader, monkeypatch, tmp_path):
    monkeypatch.setenv("BLOBCHUNK_SPILL_DIR", str(tmp_path))
    spec = _spec(reader)
    spec.load_chunk = lambda index: [{"id": 0}, {"id": 1}]
    iterator = reader["_BlobChunkIterableDatasetIterator"](spec, 1, 0, 0, 1, 0, 0, True, False)
    monkeypatch.setattr(iterator, "_spill_chunk", lambda examples: (_ for _ in ()).throw(OSError("cache full")))
    with pytest.raises(reader["_ChunkSpillError"], match="configured cache directory"):
        next(iterator)


@pytest.mark.unit
def test_replenishing_releases_previous_buffer_before_parallel_read(reader, monkeypatch):
    monkeypatch.delenv("BLOBCHUNK_SPILL_DIR", raising=False)
    spec = _spec(reader)
    iterator = reader["_BlobChunkIterableDatasetIterator"](spec, 1, 0, 0, 1, 0, 0, True, False)
    iterator.example_buffer = [{"previous": b"retained sample"}]

    def load(index):
        assert iterator.example_buffer == []
        return [{"id": 0}, {"id": 1}]

    spec.load_chunk = load
    assert iterator._replenish_example_buffer() is False


@pytest.mark.unit
def test_disk_spill_matches_real_kensler_shuffle_and_resume(reader, monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    permutation = _definitions(
        _PIPELINE / "blob_chunk_vlm/blobchunk.py",
        ["KenslerPermutation"],
        {"uint64": np.uint64, "random": np.random, "warnings": warnings},
    )["KenslerPermutation"]
    reader["np"] = np
    reader["KenslerPermutation"] = permutation
    spec = _spec(reader)
    spec.total_chunks = 4
    spec.accum_chunks = [4]
    spec.chunk_sizes = [2, 3, 4, 5]
    spec.load_chunk = lambda index: [{"chunk": index, "id": i} for i in range(spec.chunk_sizes[index])]

    def iterate(spill, resume=0):
        if spill:
            monkeypatch.setenv("BLOBCHUNK_SPILL_DIR", str(tmp_path))
        else:
            monkeypatch.delenv("BLOBCHUNK_SPILL_DIR", raising=False)
        return reader["_BlobChunkIterableDatasetIterator"](spec, 2, 3, resume, 2, 1, 17, True, False)

    memory = iterate(False)
    count = int(memory.worker_total_num_examples) * 2
    expected = [next(memory) for _ in range(count)]
    cached = iterate(True)
    assert [next(cached) for _ in range(count)] == expected
    resumed = iterate(True, resume=3)
    assert [next(resumed) for _ in range(count - 3)] == expected[3:]


@pytest.mark.unit
def test_builder_preserves_storage_api_unless_local_reading_is_explicit():
    config = SimpleNamespace(from_dict=lambda entry: entry)
    provider = SimpleNamespace(from_config=lambda entry: SimpleNamespace(get_train_dataset=lambda: entry, weight=1))

    class Concat:
        def __init__(self, datasets, weights):
            self.datasets = datasets

        def set_rank_and_world_size(self, rank, world_size):
            self.rank = rank
            self.world_size = world_size

    namespace = _definitions(
        _PIPELINE / "builder.py",
        ["_resolve_provider", "_normalize_entry", "_build_provider", "build_speech_chunk_dataset"],
        {
            **vars(typing),
            "copy": copy,
            "DATASET_PROVIDER_REGISTRY": {"blob_chunk_alm": (provider, config)},
            "_DEFAULT_PROVIDER": "blob_chunk_alm",
            "WeightedConcatIterableDataset": Concat,
        },
    )
    entries = [{"use_storage_api": True, "multi_threaded_download": True}]
    build = namespace["build_speech_chunk_dataset"]
    assert build(entries, rank=1, world_size=2).datasets[0]["use_storage_api"] is True
    assert build(entries, force_local=True).datasets[0]["use_storage_api"] is False
    assert entries[0]["use_storage_api"] is True


@pytest.fixture
def streaming_provider():
    class BaseProvider:
        pin_memory = True
        persistent_workers = True

    namespace = _definitions(
        _ROOT / "src/megatron/bridge/data/speech_chunk/streaming_provider.py",
        ["SpeechChunkStreamingDatasetProvider"],
        {
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "Any": typing.Any,
            "DatasetProvider": BaseProvider,
            "DatasetBuildContext": object,
            "LOGGER": logging.getLogger(__name__),
        },
    )
    return namespace["SpeechChunkStreamingDatasetProvider"]


@pytest.mark.unit
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("storage_override,expected_storage", [(None, False), (True, True), (False, False)])
def test_external_loader_forwards_storage_and_dp_rank_with_valid_worker_options(
    streaming_provider, monkeypatch, caplog, workers, storage_override, expected_storage
):
    entry = {
        "tokenizer": {"pretrained_model_name_or_path": "local/tokenizer"},
        "use_storage_api": False,
        "dataset_spec": ["original.json", "overlay.json"],
        "filter_options": {"max_utterance_frames": 4000},
        "weight": 0.3,
    }
    cfg = {"dataset": [entry], "dataset_collator": {}}
    dp = object()
    context = SimpleNamespace(pg_collection=SimpleNamespace(tp=None, dp=dp))
    dist = SimpleNamespace(get_rank=lambda group: 3, get_world_size=lambda group: 8)
    collator = object()
    calls = []
    native_dataset = object()
    wrapped_dataset = object()
    preparations = []

    def dataset(entries, **kwargs):
        calls.append((entries, kwargs))
        return native_dataset

    def prepare(self, dataset):
        preparations.append(dataset)
        return wrapped_dataset

    monkeypatch.setattr(streaming_provider, "_prepare_dataset", prepare)

    class Loader:
        def __init__(self, data, **kwargs):
            assert data is wrapped_dataset
            self.kwargs = kwargs

        def __iter__(self):
            return iter([self.kwargs])

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=dist))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setitem(sys.modules, "torch.utils.data", SimpleNamespace(DataLoader=Loader))
    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        SimpleNamespace(
            OmegaConf=SimpleNamespace(load=lambda path: cfg, to_container=lambda cfg, resolve: copy.deepcopy(cfg))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())),
    )
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.data.speech_chunk_streaming",
        SimpleNamespace(
            build_speech_chunk_dataset=dataset, build_speech_chunk_collator=lambda cfg, tokenizer: collator
        ),
    )
    provider = streaming_provider(
        seq_length=8192,
        speech_config_yaml="cached/data.yaml",
        num_workers=workers,
        prefetch_factor=8,
        use_storage_api=storage_override,
    )
    with caplog.at_level(logging.INFO):
        iterator, validation, test = provider.build_datasets(context)
    kwargs = next(iterator)
    assert validation is test is None
    assert preparations == [native_dataset]
    assert calls == [
        ([{**entry, "use_storage_api": expected_storage}], {"rank": 3, "world_size": 8, "force_local": False})
    ]
    assert cfg["dataset"][0]["use_storage_api"] is False
    assert f"enabled_entries={int(expected_storage)}/1" in caplog.text
    assert kwargs["batch_size"] == 1
    assert kwargs["collate_fn"] is collator
    assert kwargs["pin_memory"] is True
    if workers:
        assert kwargs["prefetch_factor"] == 8 and kwargs["persistent_workers"] is True
    else:
        assert "prefetch_factor" not in kwargs and "persistent_workers" not in kwargs


@pytest.mark.unit
def test_only_tp_source_rank_opens_storage(streaming_provider, monkeypatch):
    dist = SimpleNamespace(get_rank=lambda group: 1, get_world_size=lambda group: 2)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=dist))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    context = SimpleNamespace(pg_collection=SimpleNamespace(tp=object()))
    provider = streaming_provider(seq_length=8192, speech_config_yaml="must-not-open.yaml")
    iterator, _, _ = provider.build_datasets(context)
    assert list(iterator) == []


@pytest.mark.unit
def test_default_startup_adapter_preserves_native_dataset(streaming_provider):
    provider = streaming_provider(seq_length=8192, speech_config_yaml="unused.yaml")
    dataset = object()
    assert provider._prepare_dataset(dataset) is dataset


@pytest.mark.unit
def test_workerless_loader_cannot_have_worker_timeout(streaming_provider):
    provider = streaming_provider(seq_length=8192, speech_config_yaml="must-not-open.yaml", num_workers=0, timeout=1)
    with pytest.raises(ValueError, match="requires worker processes"):
        provider.build_datasets(SimpleNamespace())


@pytest.mark.unit
def test_storage_transport_override_cannot_be_silently_disabled(streaming_provider):
    provider = streaming_provider(
        seq_length=8192, speech_config_yaml="must-not-open.yaml", force_local=True, use_storage_api=True
    )
    with pytest.raises(ValueError, match="requires force_local=False"):
        provider.build_datasets(SimpleNamespace())


@pytest.mark.unit
def test_storage_transport_override_requires_an_explicit_boolean(streaming_provider):
    provider = streaming_provider(seq_length=8192, speech_config_yaml="must-not-open.yaml", use_storage_api="true")
    with pytest.raises(TypeError, match="bool or None"):
        provider.build_datasets(SimpleNamespace())


@pytest.mark.unit
def test_both_speech_recipes_forward_optional_transport_override():
    tree = ast.parse((_ROOT / "src/megatron/bridge/recipes/speech_cascades/speech_cascades.py").read_text())
    recipes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"speech_cascades_finetune_config", "speech_cascades_s2_kd_config"}
    }
    assert len(recipes) == 2
    for name, recipe in recipes.items():
        defaults = dict(
            zip((argument.arg for argument in recipe.args.kwonlyargs), recipe.args.kw_defaults, strict=True)
        )
        assert isinstance(defaults["streaming_use_storage_api"], ast.Constant)
        assert defaults["streaming_use_storage_api"].value is None
        target = (
            "SpeechChunkStreamingDatasetProvider"
            if name == "speech_cascades_finetune_config"
            else "speech_cascades_finetune_config"
        )
        keyword = "use_storage_api" if name == "speech_cascades_finetune_config" else "streaming_use_storage_api"
        calls = [
            node
            for node in ast.walk(recipe)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == target
        ]
        assert len(calls) == 1
        forwarded = {argument.arg: argument.value for argument in calls[0].keywords}
        assert isinstance(forwarded[keyword], ast.Name) and forwarded[keyword].id == "streaming_use_storage_api"


@pytest.mark.unit
def test_finetune_cli_forwards_optional_storage_api_to_both_recipe_paths():
    tree = ast.parse((_ROOT / "examples/models/speech_cascades/finetune.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("speech_cascades_finetune_config", "speech_cascades_s2_kd_config")
    ]
    assert len(calls) == 2
    for call in calls:
        keyword = next(item.value for item in call.keywords if item.arg == "streaming_use_storage_api")
        assert isinstance(keyword, ast.Attribute) and keyword.attr == "streaming_use_storage_api"
    options = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--streaming-use-storage-api"
    ]
    assert len(options) == 1
    defaults = {item.arg: item.value for item in options[0].keywords}
    assert isinstance(defaults["default"], ast.Constant) and defaults["default"].value is None
    assert isinstance(defaults["action"], ast.Attribute) and defaults["action"].attr == "BooleanOptionalAction"
