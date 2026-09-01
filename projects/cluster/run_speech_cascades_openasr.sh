#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# HuggingFace Open-ASR-Leaderboard WER eval of a native speech-cascades checkpoint on Singularity.
# Mirrors the production run_opensr_eval.sh "openasr" path: iterates the ESB English subsets
# (librispeech-clean/other, earnings22, spgispeech, and the min-1s "clean" ami/gigaspeech), each
# sharded across the data-parallel ranks. Prints a per-subset WER table on rank 0.
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

# editdistance + kaldialign (official Open ASR Leaderboard metric) + regex (vendored normalizer dep)
python -m pip install --quiet --no-input editdistance kaldialign regex 2>/dev/null || \
  python -m pip install --target "${OVERLAY}" --quiet --no-input editdistance kaldialign regex

OUTPUT=${OUTPUT:-/modelblob/projects/yuxuanhu_s2s_qwen/bridge-results/en-asr-s1-v2}
CHECKPOINT_ITER=${CHECKPOINT_ITER:-10000}
DATA_DIR=${DATA_DIR:-/datablob1/users/ruchaofan/Evaluation/OpenASR/esb-datasets-test-only-sorted}
TOKENIZER=${TOKENIZER:-/datablob1/users/ruchaofan/Phi-omni-checkpoints/language_ckpt/Qwen3.5-4B}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

# Model to evaluate: set MODEL_HF + MODEL_CKPT to eval any model; they default to the OUTPUT+ITER layout.
MODEL_HF=${MODEL_HF:-${OUTPUT}/stage1_hf}
MODEL_CKPT=${MODEL_CKPT:-$(printf "%s/run/checkpoints/iter_%07d" "${OUTPUT}" "${CHECKPOINT_ITER}")}
RESULT_DIR=${RESULT_DIR:-${OUTPUT}/eval_output/openasr_iter${CHECKPOINT_ITER}}
mkdir -p "${RESULT_DIR}"

# One process loads the 4.3B model once and evaluates every subset (avoids the per-subset model
# reload + NCCL re-init that tripped the init timeout). ami/gigaspeech get a 1s floor internally.
python examples/models/speech_cascades/eval_openasr.py \
  --hf-path "${MODEL_HF}" \
  --checkpoint "${MODEL_CKPT}" \
  --data-dir "${DATA_DIR}" \
  --subsets librispeech-clean librispeech-other earnings22 spgispeech ami gigaspeech \
  --tokenizer "${TOKENIZER}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${RESULT_DIR}"
