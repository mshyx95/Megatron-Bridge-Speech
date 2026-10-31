#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fleurs speech-translation (S2T) BLEU eval of a native speech-cascades checkpoint on Singularity.
# Mirrors the production run_fleurs_cascade_qwen35.py metric path, but replaces the two-stage
# Phi-4MM-ASR -> Qwen-text cascade with the native model doing audio -> greedy translation directly.
# Iterates every src!=tgt direction over the requested languages, each sharded across the
# data-parallel ranks, and prints a per-direction BLEU table (plus a macro) on rank 0.
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

# sacrebleu is the reference translation-BLEU library (matches run_fleurs_cascade_qwen35.py).
python -m pip install --quiet --no-input sacrebleu 2>/dev/null || \
  python -m pip install --target "${OVERLAY}" --quiet --no-input sacrebleu

OUTPUT=${OUTPUT:-/modelblob/projects/yuxuanhu_s2s_qwen/bridge-results/en-asr-s1-v2}
CHECKPOINT_ITER=${CHECKPOINT_ITER:-10000}
DATA_ROOT=${DATA_ROOT:-/datablob1/users/ruchaofan/Evaluation/Fleurs/fleurs_s2t}
TOKENIZER=${TOKENIZER:-/datablob1/users/ruchaofan/Phi-omni-checkpoints/language_ckpt/Qwen3.5-4B}
LANGUAGES=${LANGUAGES:-"EN DE ES FR IT JA PT ZH"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}

STAGE1_HF=${STAGE1_HF:-${OUTPUT}/stage1_hf}
CKPT_DIR=$(printf "%s/run/checkpoints/iter_%07d" "${OUTPUT}" "${CHECKPOINT_ITER}")
RESULT_DIR=${OUTPUT}/eval_output/fleurs_iter${CHECKPOINT_ITER}
mkdir -p "${RESULT_DIR}"

# One process loads the 4.3B model once and evaluates every direction (avoids the per-direction model
# reload + NCCL re-init that tripped the init timeout).
python examples/models/speech_cascades/eval_fleurs.py \
  --hf-path "${STAGE1_HF}" \
  --checkpoint "${CKPT_DIR}" \
  --data-root "${DATA_ROOT}" \
  --tokenizer "${TOKENIZER}" \
  --languages ${LANGUAGES} \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${RESULT_DIR}"
