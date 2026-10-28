#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${SOURCE_HF:?Set the prepared speech HF directory}"
: "${SOURCE_CHECKPOINT:?Set the completed Megatron iteration directory}"
: "${EXPORT_ROOT:?Set a separate export project directory}"
: "${EXPORT_PROFILE:?Set qwen35_9b or qwen36_35b_a3b}"
: "${CKPT_STEP:?Set the exact checkpoint iteration}"
[[ "$CKPT_STEP" =~ ^[1-9][0-9]*$ ]] || { echo "CKPT_STEP must be positive" >&2; exit 2; }
printf -v iteration 'iter_%07d' "$CKPT_STEP"
[[ "$(basename "$SOURCE_CHECKPOINT")" == "$iteration" ]] || {
    echo "SOURCE_CHECKPOINT does not match CKPT_STEP" >&2; exit 2;
}

export PYTHONPATH="$PWD/examples/models/speech_cascades:$PWD/src:$PWD/3rdparty/Megatron-LM"
export PYTHONUNBUFFERED=1 UV_OFFLINE=1 NCCL_RAS_ENABLE=0
python=${EXPORT_PYTHON:-/opt/venv/bin/python}
final="$EXPORT_ROOT/hf/$iteration-vllm"
work="$final.partial"
mkdir -p "$EXPORT_ROOT/hf"
exec 9>"$EXPORT_ROOT/hf/export.lock"
flock -n 9

receipt_args=(--source-hf "$SOURCE_HF" --checkpoint "$SOURCE_CHECKPOINT"
    --profile "$EXPORT_PROFILE" --step "$CKPT_STEP")
if [[ -e "$final" ]]; then
    "$python" -m speech_eval.export_receipt verify "$final" "${receipt_args[@]}"
    echo "Verified existing export: $final"
    exit 0
fi
# Interrupted exports are retained for inspection, never overwritten or resumed implicitly.
[[ ! -e "$work" ]] || { echo "Partial export already exists: $work" >&2; exit 2; }
"$python" -m speech_eval.export_receipt preflight "$work" "${receipt_args[@]}"
"$python" examples/models/speech_cascades/export_hf.py \
    --hf-dir "$SOURCE_HF" --checkpoint "$SOURCE_CHECKPOINT" --vllm --output "$work"
"$python" -m speech_eval.export_receipt write "$work" "${receipt_args[@]}"
mv -T "$work" "$final"
echo "Published verified export: $final"
