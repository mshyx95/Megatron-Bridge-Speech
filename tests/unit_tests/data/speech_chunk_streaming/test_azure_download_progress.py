"""Preserve successful slow SDK reads and resume only after transport failures."""

from types import SimpleNamespace

import pytest
from azure.core.exceptions import ServiceResponseError

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm import az_downloader


pytestmark = pytest.mark.unit


@pytest.fixture
def transport(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(az_downloader, "time", SimpleNamespace(monotonic=lambda: clock.now))
    downloader = az_downloader.AzureBlobDownloader(
        az_downloader.AzureDownloaderConfig(
            account_url="https://example.blob.core.windows.net",
            container_name="data",
            max_attempts=2,
        )
    )
    calls = []

    def configure(frames):
        def download(blob_name, **kwargs):
            calls.append((blob_name, kwargs))
            return SimpleNamespace(chunks=lambda **unused: frames(kwargs["offset"], clock))

        monkeypatch.setattr(
            downloader,
            "_resolve_blob",
            lambda blob: (SimpleNamespace(download_blob=download), blob),
        )

    return downloader, calls, configure


@pytest.mark.parametrize("buffered", [False, True])
def test_slow_successful_chunks_are_not_discarded_or_retried(transport, buffered):
    downloader, calls, configure = transport

    def frames(offset, clock):
        assert offset == 0
        for chunk in (b"abc", b"def"):
            clock.now += 1.0
            yield chunk

    configure(frames)
    if buffered:
        result = downloader._download_buffer_until_eof_resumable("audio/chunk")
        assert result.tell() == 0
        assert result.read() == b"abcdef"
    else:
        assert downloader._download_bytes_until_eof_resumable("audio/chunk") == b"abcdef"
    assert len(calls) == 1


def test_sdk_read_error_resumes_after_last_successfully_appended_byte(transport):
    downloader, calls, configure = transport

    def frames(offset, clock):
        clock.now += 1.0
        if offset == 0:
            yield b"abc"
            raise ServiceResponseError("socket read timed out")
        assert offset == 3
        yield b"def"

    configure(frames)
    assert downloader._download_bytes_until_eof_resumable("audio/chunk") == b"abcdef"
    assert [kwargs["offset"] for _, kwargs in calls] == [0, 3]
    for _, kwargs in calls:
        assert kwargs["read_timeout"] == kwargs["connection_timeout"] == kwargs["timeout"] == 30.0


def test_reference_transport_defaults_bound_requests_without_false_stalls():
    config = az_downloader.AzureDownloaderConfig(
        account_url="https://example.blob.core.windows.net",
        container_name="data",
    )
    assert config.max_single_get_size == config.max_chunk_get_size == config.chunk_size == 4 * 1024 * 1024
    assert config.request_timeout_s == 30.0


def test_nonretryable_reader_errors_are_not_hidden(transport):
    downloader, calls, configure = transport

    def frames(offset, clock):
        yield b"abc"
        raise ValueError("invalid transport payload")

    configure(frames)
    with pytest.raises(ValueError, match="invalid transport payload"):
        downloader._download_bytes_until_eof_resumable("audio/chunk")
    assert len(calls) == 1
