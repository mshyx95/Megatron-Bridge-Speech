# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# From Cascades by Abdel Heba's efforts

import io
import os
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable
from urllib.parse import unquote, urlparse

import backoff
import requests

from azure.identity import ManagedIdentityCredential
from azure.storage.blob import ContainerClient
from azure.core.exceptions import (
    AzureError,
    ServiceRequestError,
    ServiceResponseError,
    IncompleteReadError,
    ResourceNotFoundError,
)

LOG = logging.getLogger(__name__)

# Quiet the verbose HTTP request/response dumps
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# Optional: quiet the broader Azure SDK loggers too
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)


@dataclass
class AzureDownloaderConfig:
    # Storage
    account_url: str
    container_name: str
    uai_client_id_env: str = "UAI_CLIENT_ID"

    # Client retry knobs (applies to SDK pipeline retries)
    retry_total: int = 6
    retry_connect: int = 6
    retry_read: int = 6
    retry_status: int = 6

    # Transfer tuning
    max_single_get_size: int = 4 * 1024 * 1024
    max_chunk_get_size: int = 4 * 1024 * 1024

    # Download behavior
    chunk_size: int = 4 * 1024 * 1024
    stall_timeout_s: float = 0.3  # Legacy setting; socket stalls use request_timeout_s.
    request_timeout_s: float = 30.0
    max_concurrency: int = 8
    max_attempts: int = 16
    log_every_s: float = 3.0
    logging_enable: bool = False  # Azure SDK wire logging (usually False)

    # Backoff (your outer retry loop)
    backoff_max_tries: int = 4
    backoff_max_time_s: Optional[float] = None  # e.g., 300.0


class AzureBlobDownloader:
    """
    Best practice:
      - Keep ONE ContainerClient (auth + pipeline) per process.
      - For each blob operation, create a BlobClient via container.get_blob_client(blob_name).
      - Use blob.get_blob_properties() to check existence (fast, direct).
      - Download streaming until EOF and resume using offset on stall/errors.
    """

    def __init__(self, cfg: AzureDownloaderConfig, *, lazy: bool = True):
        self.cfg = cfg
        self._cred = None
        self._container: Optional[ContainerClient] = None
        self._containers = {}
        self._containers_lock = threading.Lock()

        # Optional hook called on each retry (index starts at 1)
        self.on_retry: Optional[Callable[[int, Exception, int], None]] = None

        if not lazy:
            self._ensure_client()

    def _ensure_client(self) -> None:
        if self._container is not None:
            return

        client_id = os.environ.get(self.cfg.uai_client_id_env)
        if not client_id:
            raise RuntimeError(
                f"Missing env var {self.cfg.uai_client_id_env}. "
                f"Set it to your User Assigned Managed Identity client id."
            )

        self._cred = ManagedIdentityCredential(client_id=client_id)

        self._container = self._create_container_client(self.cfg.account_url, self.cfg.container_name)
        self._containers[(self.cfg.account_url.rstrip("/"), self.cfg.container_name)] = self._container

    def _create_container_client(self, account_url: str, container_name: str) -> ContainerClient:
        return ContainerClient(
            account_url=account_url,
            container_name=container_name,
            credential=self._cred,
            retry_total=self.cfg.retry_total,
            retry_connect=self.cfg.retry_connect,
            retry_read=self.cfg.retry_read,
            retry_status=self.cfg.retry_status,
            max_single_get_size=self.cfg.max_single_get_size,
            max_chunk_get_size=self.cfg.max_chunk_get_size,
        )

    @staticmethod
    def _split_blob_url(blob_name: str) -> tuple[str, str, str] | None:
        parsed = urlparse(blob_name)
        if parsed.scheme not in {"http", "https", "az"} or not parsed.netloc:
            return None
        path_parts = parsed.path.lstrip("/").split("/", 1)
        if len(path_parts) != 2 or not all(path_parts):
            raise ValueError("Azure blob URL must include a container and blob path")
        account_url = (
            f"https://{parsed.netloc}.blob.core.windows.net"
            if parsed.scheme == "az"
            else f"{parsed.scheme}://{parsed.netloc}"
        )
        return account_url, unquote(path_parts[0]), unquote(path_parts[1])

    def _resolve_blob(self, blob_name: str) -> tuple[ContainerClient, str]:
        self._ensure_client()
        split_url = self._split_blob_url(blob_name)
        if split_url is None:
            return self._container, blob_name

        account_url, container_name, relative_blob_name = split_url
        key = (account_url.rstrip("/"), container_name)
        with self._containers_lock:
            container = self._containers.get(key)
            if container is None:
                container = self._create_container_client(account_url, container_name)
                self._containers[key] = container
        return container, relative_blob_name

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        # Not-found should NOT be retried here (surface it cleanly)
        if isinstance(exc, (FileNotFoundError, ResourceNotFoundError)):
            return False

        return isinstance(
            exc,
            (
                TimeoutError,
                ServiceRequestError,
                ServiceResponseError,
                AzureError,
                IncompleteReadError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ),
        )

    def _giveup(self, exc: Exception) -> bool:
        return not self._is_retryable_exception(exc)

    def _on_backoff(self, details: dict) -> None:
        # details: tries, wait, target, args, kwargs, value, elapsed
        tries = int(details.get("tries", 0))
        wait_s = float(details.get("wait", 0.0))
        exc = details.get("value")
        offset = -1
        try:
            # kwargs passed to _download_until_eof_resumable include offset
            offset = int(details.get("kwargs", {}).get("_offset_for_log", -1))
        except Exception:
            pass

        LOG.warning("Azure download retry #%d in %.2fs due to %r (offset=%s)", tries, wait_s, exc, offset)
        if self.on_retry is not None and exc is not None:
            try:
                self.on_retry(tries, exc, offset)
            except Exception:
                pass

    def download(self, blob_name: str) -> bytes:
        """
        Download full blob bytes.
        - Pre-check existence with get_blob_properties() -> raises FileNotFoundError if missing
        - Resumable download until EOF (no size required)
        """
        self._ensure_client()
        assert self._container is not None

        decorator = backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=self.cfg.backoff_max_tries,
            max_time=self.cfg.backoff_max_time_s,
            jitter=backoff.full_jitter,
            on_backoff=self._on_backoff,
            giveup=self._giveup,
        )
        return decorator(self._download_bytes_until_eof_resumable)(blob_name)

    def download_to_buffer(self, blob_name: str, concurrency: int) -> io.BytesIO:
        """
        Download full blob into a seekable in-memory buffer.
        This avoids the extra `bytearray -> bytes -> BytesIO` copy chain at the caller.
        """
        self._ensure_client()
        assert self._container is not None

        decorator = backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=self.cfg.backoff_max_tries,
            max_time=self.cfg.backoff_max_time_s,
            jitter=backoff.full_jitter,
            on_backoff=self._on_backoff,
            giveup=self._giveup,
        )
        return decorator(self._download_buffer_until_eof_resumable)(blob_name, concurrency=concurrency)

    def _stream_blob_until_eof(self, blob_name: str, sink, append_chunk, finalize, *, _offset_for_log: int = 0, concurrency: int = None):
        """
        Streams until EOF and resumes on stall/errors using offset.
        Done condition: iterator ends cleanly.
        """
        offset = 0
        last_log = time.monotonic()
        attempt = 0
        last_exc = None
        container, relative_blob_name = self._resolve_blob(blob_name)

        while True:
            attempt += 1
            if attempt > self.cfg.max_attempts:
                raise TimeoutError(f"Exceeded max_attempts={self.cfg.max_attempts} at offset={offset}")

            try:
                # This may raise before yielding (initial request). That’s OK; outer backoff will handle it.
                downloader = container.download_blob(
                    relative_blob_name,
                    offset=offset,
                    length=None,  # until EOF
                    max_concurrency=concurrency or self.cfg.max_concurrency,
                    timeout=self.cfg.request_timeout_s,              # service-side op timeout
                    connection_timeout=self.cfg.request_timeout_s,   # socket connect
                    read_timeout=self.cfg.request_timeout_s,         # socket read
                    logging_enable=self.cfg.logging_enable,
                )

                # Some SDK versions support chunks(chunk_size=...), others don't.
                try:
                    it = downloader.chunks(chunk_size=self.cfg.chunk_size)
                except TypeError:
                    it = downloader.chunks()

                for ch in it:
                    now = time.monotonic()
                    # A yielded chunk is progress, even after SDK retries or a slow transfer.
                    # Socket read_timeout bounds inactivity while waiting inside the iterator.
                    if ch:
                        append_chunk(sink, ch)
                        offset += len(ch)

                    # Rate-limited progress logging
                    if (now - last_log) > self.cfg.log_every_s and attempt > 4:
                        LOG.info("Downloading %s: received %.1f MB (attempt=%d)", blob_name, offset / 1e6, attempt)
                        last_log = now

                # Iterator ended cleanly => EOF
                return finalize(sink)

            except ResourceNotFoundError as e:
                # Blob deleted mid-download or missing
                last_exc = e
                break

            except Exception as e:
                # For retryable issues, just reopen stream from same offset.
                if not self._is_retryable_exception(e):
                    raise

                last_exc = e
                LOG.warning("Reopening stream for %s at offset=%d due to %r", blob_name, offset, e)
                # Let outer backoff see the error too on the next attempt if initial request fails again.
                # But here we can keep a tight local loop to resume quickly.
                _offset_for_log = offset
                continue

        raise FileNotFoundError(f"Blob not found: {blob_name}") from last_exc

    def _download_bytes_until_eof_resumable(self, blob_name, *, _offset_for_log: int = 0, concurrency: int = None) -> bytes:
        return self._stream_blob_until_eof(
            blob_name,
            bytearray(),
            lambda sink, chunk: sink.extend(chunk),
            bytes,
            _offset_for_log=_offset_for_log,
            concurrency=concurrency,
        )

    def _download_buffer_until_eof_resumable(self, blob_name, *, _offset_for_log: int = 0, concurrency: int = None) -> io.BytesIO:
        def finalize(buffer: io.BytesIO) -> io.BytesIO:
            buffer.seek(0)
            return buffer

        return self._stream_blob_until_eof(
            blob_name,
            io.BytesIO(),
            lambda sink, chunk: sink.write(chunk),
            finalize,
            _offset_for_log=_offset_for_log,
            concurrency=concurrency,
        )
