#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runner="$repo_root/examples/models/speech_cascades/run_vllm_eval.sh"
suite=${1:?Pass exactly one evaluation suite}
case "$suite" in openasr|openasr_ml|openasr_long|bbh|voicebench|bfcl|fleurs) ;; *) exit 2 ;; esac
: "${MODEL:?Set MODEL to the HF candidate}"
: "${TOKENIZER:?Set TOKENIZER to the corresponding tokenizer}"
: "${EXP_ID:?Set EXP_ID as in the reference runners}"
: "${AML_EXP_NAME:?Set AML_EXP_NAME as in the reference runners}"
: "${CKPT_STEP:?Set CKPT_STEP to the concrete checkpoint step}"
: "${AMLT_OUTPUT_DIR:?Set a writable results directory}"
export PYTHONPATH="$repo_root/examples/models/speech_cascades:$repo_root/src/megatron/bridge/models/speech_cascades"
export PYTHONDONTWRITEBYTECODE=1
if [[ -z "${SPEECH_EVAL_RANK:-}" ]]; then
    exec "${SPEECH_EVAL_PYTHON:-/usr/bin/python3}" -u -m speech_eval.mpi --suite "$suite" \
        -- bash "$0" "$suite"
fi
[[ "${SPEECH_EVAL_WORLD_SIZE:-}" == 8 && "$SPEECH_EVAL_RANK" =~ ^[0-7]$ ]] || exit 2
: "${SPEECH_EVAL_RUN_KEY:?Missing coordinated MPI run identity}"
: "${SPEECH_EVAL_LOG_DIR:?Missing rank-specific output directory}"
: "${OUTPUT_DIR:?Missing rank-specific prediction directory}"
: "${PORT:?Missing rank-specific server port}"
export SPEECH_NUM_SERVERS=1
export SPEECH_EVAL_NATIVE_TOOLS=0
export SPEECH_EVAL_CONCURRENCY="${SPEECH_EVAL_CONCURRENCY:-2}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
export SERVED_MODEL_NAME="${MODEL_LABEL:-b1}-$CKPT_STEP-$suite"
export EVAL_TAG="${EVAL_TAG:-vllm-strong}"
export CHAT_TEMPLATE=qwen3_5_audio_nothink_instruct
export MODEL_PATH="$MODEL"
export RUN_JUDGE="${RUN_JUDGE:-0}"

case "$suite" in
    openasr|openasr_ml|openasr_long)
        exec bash "$runner" bash ./run_opensr_eval.sh \
            "$EXP_ID" "$AML_EXP_NAME" "$CKPT_STEP" "$TOKENIZER" "$suite" \
            1 0 false "$CHAT_TEMPLATE" true "$EVAL_TAG" "" "${ASR_BATCH_SIZE:-2}" \
            svad ter /datablob1 lfg3
        ;;
    bbh|voicebench)
        export BATCH_SIZE="${BENCH_BATCH_SIZE:-2}" MAX_NEW_TOKENS=2048 POSTPROCESS=0
        if [[ "$suite" == bbh ]]; then
            export TASK=bigbench DATASET_PATH="${BBH_DATASET:?Set BBH_DATASET}"
        else
            export TASK=voicebench DATASET_PATH="${VB_DATASET:?Set VB_DATASET}"
            export RUN_EVAL=0 RUN_JUDGE=0 POSTPROCESS=0
        fi
        export RUN_EVAL="${RUN_EVAL:-0}"
        exec bash "$runner" bash ./run_s2t_eval.sh
        ;;
    bfcl)
        export MODEL_FAMILY=qwen35-audio PREPARE_BFCL=1 AUDIO_MODE=reference
        export SOURCE_DIR="${BFCL_SOURCE_DIR:?Set BFCL_SOURCE_DIR}"
        export DATA_DIR="$SPEECH_EVAL_LOG_DIR/data"
        export CATEGORIES="simple multiple parallel parallel_multiple irrelevance"
        export LIMIT_PER_CATEGORY=0 NUM_WORKERS=1 BATCH_SIZE="${BENCH_BATCH_SIZE:-2}" MAX_NEW_TOKENS=512
        exec bash "$runner" bash ./run_bfcl_v3.sh
        ;;
    fleurs)
        exec bash "$runner" bash ./run_st_eval.sh \
            "$EXP_ID" "$AML_EXP_NAME" "$CKPT_STEP" "$TOKENIZER" fleurs false \
            "$CHAT_TEMPLATE" true "$EVAL_TAG" "${ST_BATCH_SIZE:-2}" /datablob1
        ;;
    *) printf 'Unknown evaluation suite: %s\n' "$suite" >&2; exit 2 ;;
esac