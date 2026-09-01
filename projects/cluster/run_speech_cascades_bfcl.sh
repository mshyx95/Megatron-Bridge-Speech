#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# BFCL V3 function-call eval of a native speech-cascades checkpoint on Singularity.
# Mirrors the production run_bfcl_v3.sh pipeline: generate per-row function-call responses from audio
# (sharded across the data-parallel ranks, load-once model), then score function-call MICRO accuracy
# against the dataset's possible_answer/ ground truth on rank 0.
set -e -o xtrace

cd "$(dirname "$0")/../.."

if [[ -d /opt/bridge-overlay ]]; then
  OVERLAY=/opt/bridge-overlay
else
  OVERLAY=${OVERLAY_DIR:-$HOME/bridge-overlay}
fi

export PYTHONPATH=${OVERLAY}:$(pwd)/src:$(pwd)/3rdparty/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1
# Eager decode: torch.compile/dynamo recompiles per generation length and desyncs ranks past the
# NCCL init timeout on the larger categories (same rationale as the OpenASR launcher).
export TORCHDYNAMO_DISABLE=1

# soundfile is required to read the staged BFCL 16 kHz mono wavs.
python -m pip install --quiet --no-input soundfile 2>/dev/null || \
  python -m pip install --target "${OVERLAY}" --quiet --no-input soundfile

OUTPUT=${OUTPUT:-/modelblob/projects/yuxuanhu_s2s_qwen/bridge-results/en-asr-s1-v2}
CHECKPOINT_ITER=${CHECKPOINT_ITER:-10000}
# BFCL speech data dir: BFCL_v3_<category>.json + possible_answer/BFCL_v3_<category>.json.
DATA_DIR=${DATA_DIR:-/datablob1/users/ruchaofan/Evaluation/berkeley-function-call-leaderboard/speech_data}
TOKENIZER=${TOKENIZER:-/datablob1/users/ruchaofan/Phi-omni-checkpoints/language_ckpt/Qwen3.5-4B}
CATEGORIES=${CATEGORIES:-"simple multiple parallel parallel_multiple irrelevance"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
LIMIT_PER_CATEGORY=${LIMIT_PER_CATEGORY:-0}

STAGE1_HF=${STAGE1_HF:-${OUTPUT}/stage1_hf}
CKPT_DIR=$(printf "%s/run/checkpoints/iter_%07d" "${OUTPUT}" "${CHECKPOINT_ITER}")
RESULT_DIR=${OUTPUT}/eval_output/bfcl_iter${CHECKPOINT_ITER}
mkdir -p "${RESULT_DIR}"
RESPONSES=${RESULT_DIR}/responses.json

# One process loads the 4.3B model once and decodes every category (avoids per-category model reload).
python examples/models/speech_cascades/eval_bfcl.py \
  --hf-path "${STAGE1_HF}" \
  --checkpoint "${CKPT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --tokenizer "${TOKENIZER}" \
  --categories ${CATEGORIES} \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --limit-per-category "${LIMIT_PER_CATEGORY}" \
  --output "${RESPONSES}"

# Score function-call micro accuracy once on rank 0 (report mode needs no GPU).
if [[ "${LOCAL_RANK:-0}" -eq 0 ]]; then
  python examples/models/speech_cascades/eval_bfcl.py \
    --report \
    --responses "${RESPONSES}" \
    --data-dir "${DATA_DIR}" \
    --categories ${CATEGORIES} \
    --output-dir "${RESULT_DIR}"
fi
