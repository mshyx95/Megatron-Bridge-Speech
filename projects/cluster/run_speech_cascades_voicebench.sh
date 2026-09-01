#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# VoiceBench (9-task macro) eval of a native speech-cascades checkpoint on Singularity.
# Mirrors the production S2T VoiceBench flow: one process loads the 4.3B model once and greedily
# decodes every subset (audio-only, empty text instruction) into per-subset <subset>_results.jsonl,
# sharded across the data-parallel ranks. Rank 0 then runs the official VoiceBench evaluators (via
# the ported evaluate_voicebench pipeline, if RUN_EVAL=1) and prints the 9-task macro.
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
# Eager decode: torch.compile/dynamo recompiles per audio length during variable-length generation,
# which dominated runtime and desynced ranks past the NCCL init timeout on the large subsets.
export TORCHDYNAMO_DISABLE=1

OUTPUT=${OUTPUT:-/modelblob/projects/yuxuanhu_s2s_qwen/bridge-results/en-asr-s1-v2}
CHECKPOINT_ITER=${CHECKPOINT_ITER:-4000}
MANIFEST=${MANIFEST:-/datablob1/users/yuxuanhu/eval/voicebench/voicebench.jsonl}
AUDIO_ROOT=${AUDIO_ROOT:-}
TOKENIZER=${TOKENIZER:-/datablob1/users/ruchaofan/Phi-omni-checkpoints/language_ckpt/Qwen3.5-4B}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

STAGE1_HF=${STAGE1_HF:-${OUTPUT}/stage1_hf}
CKPT_DIR=$(printf "%s/run/checkpoints/iter_%07d" "${OUTPUT}" "${CHECKPOINT_ITER}")
RESULT_DIR=${OUTPUT}/eval_output/voicebench_iter${CHECKPOINT_ITER}
mkdir -p "${RESULT_DIR}"

audio_args=()
if [[ -n "${AUDIO_ROOT}" ]]; then
  audio_args+=(--audio-root "${AUDIO_ROOT}")
fi

# One process loads the 4.3B model once and decodes every subset (audio -> greedy answer), writing
# ${RESULT_DIR}/<subset>_results.jsonl for the official evaluators.
python examples/models/speech_cascades/eval_voicebench.py generate \
  --hf-path "${STAGE1_HF}" \
  --checkpoint "${CKPT_DIR}" \
  --manifest "${MANIFEST}" \
  --tokenizer "${TOKENIZER}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${RESULT_DIR}" \
  "${audio_args[@]}"

# Score with the official VoiceBench evaluators (GPT-4.1 judge for the open/qa subsets) and print the
# 9-task macro. RUN_EVAL=0 stops after generation (e.g. when the judge credentials are unavailable).
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${RANK:-0}}}
if [[ "${rank}" == "0" && "${RUN_EVAL:-0}" == "1" ]]; then
  eval_args=(--results-dir "${RESULT_DIR}" --output-dir "${RESULT_DIR}/evaluation")
  if [[ -n "${VOICEBENCH_OFFICIAL_REPO:-}" ]]; then
    eval_args+=(--official-repo "${VOICEBENCH_OFFICIAL_REPO}")
  fi
  if [[ "${RUN_JUDGE:-0}" == "1" ]]; then
    eval_args+=(--run-judge)
  fi
  python scripts/audio_eval/evaluate_voicebench_s2t.py "${eval_args[@]}"
  python examples/models/speech_cascades/eval_voicebench.py macro \
    --metrics-dir "${RESULT_DIR}/evaluation/metrics" \
    --output-json "${RESULT_DIR}/voicebench_macro.json"
fi
