# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export a native SpeechCascadesModel back to a HuggingFace-format directory.

Usage:
    uv run python -m torch.distributed.run --nproc_per_node=1 \
        examples/models/speech_cascades/export_hf.py \
        --hf-dir /path/to/converted_hf_dir \
        --output /path/to/exported_hf_dir
"""

import argparse

import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

import megatron.bridge.models.speech_cascades.bridge  # noqa: F401  (registers the bridge)
from megatron.bridge import AutoBridge


def main() -> None:
    """Build the native model from the converted HF directory and export it back to HF format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True, help="Converted HF directory to build the model from.")
    parser.add_argument("--output", required=True, help="Destination directory for the exported HF checkpoint.")
    args = parser.parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    ps.initialize_model_parallel(1, 1)
    torch.cuda.set_device(0)
    model_parallel_cuda_manual_seed(0)

    bridge = AutoBridge.from_hf_pretrained(args.hf_dir)
    model = bridge.to_megatron_model(load_weights=True, wrap_with_ddp=False)
    bridge.save_hf_pretrained(model, args.output)
    if ps.get_data_parallel_rank() == 0:
        print(f"Exported HF checkpoint to {args.output}")


if __name__ == "__main__":
    main()
