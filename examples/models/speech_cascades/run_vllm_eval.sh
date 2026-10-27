#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

entry_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$entry_dir/../../.." && pwd)
: "${MODEL:?Set MODEL to the exported serving checkpoint}"
: "${SPEECH_EVAL_LOG_DIR:?Set SPEECH_EVAL_LOG_DIR to a separate results directory}"
if (($# == 0)); then
    printf 'Usage: %s bash ./run_opensr_eval.sh <original arguments>\n' "$0" >&2
    exit 2
fi

mkdir -p "$SPEECH_EVAL_LOG_DIR"
SPEECH_EVAL_LOG_DIR=$(cd "$SPEECH_EVAL_LOG_DIR" && pwd)
export SPEECH_EVAL_LOG_DIR
# Unix sockets have a 107-byte path limit and need a local, short scratch path.
work=$(mktemp -d /tmp/speech-eval.XXXXXXXX)
export TMPDIR="$work"
server_pids=()
health_pid=
cleanup() {
    status=$?
    trap - EXIT
    if [[ -n "$health_pid" ]]; then
        kill "$health_pid" 2>/dev/null || true
        wait "$health_pid" 2>/dev/null || true
    fi
    for server_pid in "${server_pids[@]}"; do
        kill -- "-$server_pid" 2>/dev/null || kill "$server_pid" 2>/dev/null || true
    done
    for server_pid in "${server_pids[@]}"; do
        wait "$server_pid" 2>/dev/null || true
    done
    rm -rf -- "$work"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$work/scripts" "$work/bin"
cp -a "$entry_dir/speech_eval/scripts/audio_eval" "$work/scripts/"
native_python=${SPEECH_EVAL_PYTHON:-/usr/bin/python3}
ln -s "$native_python" "$work/bin/python"
export PATH="$work/bin:$PATH"
export PYTHONPATH="$entry_dir:$repo_root/src/megatron/bridge/models/speech_cascades"
unset VIRTUAL_ENV PYTHONHOME
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-speech-cascades}"
export PYTHON="$native_python"
evaluation_args=(--evaluation-mode)
case "${SPEECH_EVAL_NATIVE_TOOLS:-0}" in
    0) ;;
    1) evaluation_args+=(--native-tools) ;;
    *) printf 'SPEECH_EVAL_NATIVE_TOOLS must be 0 or 1\n' >&2; exit 2 ;;
esac

while IFS= read -r variable; do
    case "$variable" in
        OMPI_*|PMI_*|PMIX_*|TORCHELASTIC_*|RANK|WORLD_SIZE|LOCAL_RANK|LOCAL_WORLD_SIZE|GROUP_RANK|ROLE_RANK|ROLE_WORLD_SIZE|MASTER_ADDR|MASTER_PORT)
            unset "$variable"
            ;;
    esac
done < <(compgen -e)

if [[ -z "${SPEECH_BASE_URL:-}" && -z "${SPEECH_BASE_URLS:-}" ]]; then
    if compgen -G "$SPEECH_EVAL_LOG_DIR/server*.log" > /dev/null; then
        printf 'Refusing to overwrite server logs; choose a fresh SPEECH_EVAL_LOG_DIR\n' >&2
        exit 2
    fi
    server_count=${SPEECH_NUM_SERVERS:-1}
    [[ "$server_count" =~ ^[1-8]$ ]] || { printf 'SPEECH_NUM_SERVERS must be 1..8\n' >&2; exit 2; }
    if ((server_count > 1)); then
        IFS=, read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
        ((${#devices[@]} == server_count)) || { printf 'Expected one visible GPU per server\n' >&2; exit 2; }
        declare -A selected_devices=()
        for device in "${devices[@]}"; do
            [[ -n "$device" && "$device" != -1 && -z "${selected_devices[$device]:-}" ]] || exit 2
            selected_devices[$device]=1
        done
    fi
    export SPEECH_SOURCE_MODEL="$MODEL" SPEECH_MODEL_VIEW="$work/model" SPEECH_ENTRY_DIR="$entry_dir"
    "$native_python" - <<'PY'
import os
import runpy
from pathlib import Path
from speech_eval.tokenizer import prepare_tokenizer_view

source = Path(os.environ["SPEECH_SOURCE_MODEL"]).resolve(strict=True)
view = Path(os.environ["SPEECH_MODEL_VIEW"])
view.mkdir()
for path in source.iterdir():
    if path.is_file() and path.name not in {"config.json", "tokenizer_config.json"}:
        (view / path.name).symlink_to(path)
(view / "config.json").write_bytes((source / "config.json").read_bytes())
exporter = runpy.run_path(str(Path(os.environ["SPEECH_ENTRY_DIR"]) / "export_hf.py"))
exporter["prepare_vllm_config"](view)
prepare_tokenizer_view(source, view)
PY
    vllm_root=${VLLM_ROOT:-/opt/vllm-dev}
    base_port=${PORT:-8000}
    endpoints='['
    for ((server_index=0; server_index<server_count; server_index++)); do
        port=$((base_port + server_index))
        ((server_index == 0)) || endpoints+=','
        endpoints+="\"http://127.0.0.1:$port/v1\""
        log_file="$SPEECH_EVAL_LOG_DIR/server.log"
        ((server_count == 1)) || log_file="$SPEECH_EVAL_LOG_DIR/server-$server_index.log"
        (
            export PORT="$port"
            if ((server_count > 1)); then
                export CUDA_VISIBLE_DEVICES="${devices[$server_index]}"
            fi
            exec setsid "$native_python" -m speech_eval.runtime launch "$SPEECH_MODEL_VIEW" \
                --python "${VLLM_PYTHON:-/opt/vllm/bin/python}" --vllm-root "$vllm_root" \
                "${evaluation_args[@]}" --mode "${SPEECH_MODE:-standard}" --proposer stable \
                --served-model-name "$SERVED_MODEL_NAME" --host 127.0.0.1 --port "$port" \
                --tensor-parallel-size 1 --max-model-len "${MAX_MODEL_LEN:-8192}" \
                --max-num-seqs "${MAX_NUM_SEQS:-8}" \
                --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
        ) > >(tee "$log_file" | sed -u "s/^/[vllm rank=${SPEECH_EVAL_RANK:-0} server=$server_index] /") 2>&1 &
        server_pids+=("$!")
        printf '[eval rank=%s] SERVER_START gpu=%s endpoint=http://127.0.0.1:%s/v1 model=%s log=%s\n' \
            "${SPEECH_EVAL_RANK:-0}" "${CUDA_VISIBLE_DEVICES:-scheduler-default}" "$port" "$SERVED_MODEL_NAME" "$log_file"
    done
    export SPEECH_BASE_URL="http://127.0.0.1:$base_port/v1"
    export SPEECH_BASE_URLS="$endpoints]"
    export SPEECH_SERVER_PID="${server_pids[0]}" SPEECH_SERVER_PIDS="${server_pids[*]}"
fi

export SPEECH_SERVER_READY_MARKER="$work/server-ready"
"$native_python" - <<'PY' &
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s [server-check] %(message)s")
logger = logging.getLogger(__name__)
endpoints = json.loads(os.environ["SPEECH_BASE_URLS"]) if os.environ.get("SPEECH_BASE_URLS") else [os.environ["SPEECH_BASE_URL"]]
expected = int(os.environ.get("SPEECH_NUM_SERVERS", "1"))
if len(endpoints) != expected or len(set(endpoints)) != expected:
    raise ValueError("Expected distinct endpoints for every requested GPU")
startup_timeout = int(os.environ.get("SPEECH_SERVER_START_TIMEOUT", "900"))
if startup_timeout < 1:
    raise ValueError("SPEECH_SERVER_START_TIMEOUT must be positive")
deadline = time.monotonic() + startup_timeout
for endpoint in endpoints:
    logger.info("SERVER_WAIT rank=%s endpoint=%s logs=%s", os.environ.get("SPEECH_EVAL_RANK", "0"),
                endpoint, os.environ["SPEECH_EVAL_LOG_DIR"])
    base = endpoint.removesuffix("/v1")
    while True:
        for pid in os.environ.get("SPEECH_SERVER_PIDS", "").split():
            os.kill(int(pid), 0)
        try:
            with urlopen(base + "/health", timeout=5) as response:
                if response.status == 200:
                    break
        except (URLError, socket.timeout):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"vLLM endpoint {endpoint} did not become healthy within {startup_timeout} seconds")
        threading.Event().wait(1)
    with urlopen(endpoint + "/models", timeout=10) as response:
        models = json.load(response)["data"]
    if os.environ["SERVED_MODEL_NAME"] not in {model["id"] for model in models}:
        raise ValueError(f"Endpoint {endpoint} does not serve the requested model")
    logger.info("SERVER_READY rank=%s endpoint=%s model=%s", os.environ.get("SPEECH_EVAL_RANK", "0"),
                endpoint, os.environ["SERVED_MODEL_NAME"])
Path(os.environ["SPEECH_SERVER_READY_MARKER"]).write_text("ready\n")
PY
health_pid=$!
set +e
wait -n "$health_pid" "${server_pids[@]}"
health_status=$?
set -e
if [[ ! -f "$SPEECH_SERVER_READY_MARKER" ]]; then
    printf '[eval rank=%s] SERVER_START_FAILED status=%s logs=%s\n' \
        "${SPEECH_EVAL_RANK:-0}" "$health_status" "$SPEECH_EVAL_LOG_DIR" >&2
    exit "$((health_status == 0 ? 1 : health_status))"
fi
wait "$health_pid"
health_pid=

export CUDA_VISIBLE_DEVICES=""
printf '[eval rank=%s] CLIENT_START output=%s logs=%s\n' "${SPEECH_EVAL_RANK:-0}" \
    "${OUTPUT_DIR:-$SPEECH_EVAL_LOG_DIR}" "$SPEECH_EVAL_LOG_DIR"
cd "$work/scripts/audio_eval"
"$@"