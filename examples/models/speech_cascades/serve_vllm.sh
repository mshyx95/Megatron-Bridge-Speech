#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

entry_dir=${SPEECH_EVAL_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
: "${MODEL:?Set MODEL to a validated vLLM export}"
if [[ -d "$entry_dir/speech_eval" ]]; then
    export PYTHONPATH="$entry_dir${PYTHONPATH:+:$PYTHONPATH}"
    runtime=("${SPEECH_EVAL_PYTHON:-/usr/bin/python3}" -m speech_eval.runtime)
elif [[ -n "${SPEECH_EVAL_SOURCE_DIR:-}" ]]; then
    printf 'SPEECH_EVAL_SOURCE_DIR must contain the repository speech_eval package\n' >&2
    exit 2
else
    # Docker copies this script outside the repository; preserve that image's entrypoint.
    command -v speech-runtime > /dev/null || {
        printf 'Set SPEECH_EVAL_SOURCE_DIR or use the prepared speech inference image\n' >&2
        exit 2
    }
    runtime=(speech-runtime)
fi
exec "${runtime[@]}" launch "$MODEL" \
    --python "${VLLM_PYTHON:-${VLLM_PY:-/opt/vllm/bin/python}}" \
    --vllm-root "${VLLM_ROOT:-/opt/vllm-dev}" \
    --mode "${SPEECH_MODE:-standard}" \
    --proposer stable \
    --host "${HOST:-127.0.0.1}" \
    --port "${PORT:-8000}" \
    --served-model-name "${SERVED_MODEL_NAME:-speech-cascades}" \
    --evaluation-mode \
    "$@"