#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export EXPORT_PROFILE=qwen35_9b CKPT_STEP="${CKPT_STEP:-4000}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_speech_cascades_vllm_export.sh"
