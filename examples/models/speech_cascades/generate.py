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

"""Greedy ASR decoding with a native SpeechCascadesModel from a captured ALM batch.

The ``--batch`` file is a saved ALM collator batch (``dict`` with ``input_ids``, ``inputs_embeds``,
``audio_attention_mask``, ``audio_embed_sizes``, ``audio_frames``). Generation continues from the
audio+text prompt and prints the decoded continuation token ids.

Usage:
    uv run python -m torch.distributed.run --nproc_per_node=1 \
        examples/models/speech_cascades/generate.py \
        --hf-dir /path/to/converted_hf_dir \
        --batch /path/to/m1_capture/batch.pt \
        --max-new-tokens 16
"""

import argparse

import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

import megatron.bridge.models.speech_cascades.bridge  # noqa: F401  (registers the bridge)
from megatron.bridge import AutoBridge
from megatron.bridge.data.speech_chunk import ALMDataCollator
from megatron.bridge.models.speech_cascades.generate import greedy_generate


def main() -> None:
    """Load the native model and greedily decode a continuation from the captured audio batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True, help="Converted HF directory to build the model from.")
    parser.add_argument("--batch", required=True, help="Saved ALM batch (.pt) with audio features.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    ps.initialize_model_parallel(1, 1)
    torch.cuda.set_device(0)
    model_parallel_cuda_manual_seed(0)

    bridge = AutoBridge.from_hf_pretrained(args.hf_dir)
    model = bridge.to_megatron_model(load_weights=True, wrap_with_ddp=False)[0]
    model.eval()
    inner = model.module if hasattr(model, "module") else model

    ref = torch.load(args.batch, map_location="cpu", weights_only=False)
    frames = ref["audio_frames"]
    features = [ref["inputs_embeds"][i, : int(frames[i])].clone() for i in range(ref["inputs_embeds"].shape[0])]
    example = {
        "input_ids": ref["input_ids"],
        "labels": ref["labels"],
        "input_embeds": features,
        "audio_embed_sizes": ref["audio_embed_sizes"],
        "audio_frames": frames,
        "is_text_sample": False,
    }
    batch = ALMDataCollator(pad_token_id=int(ref["input_ids"].max()) + 1, audio_key="inputs_embeds")([example])

    out = greedy_generate(
        inner,
        ref["input_ids"].cuda(),
        audio_input_features=batch["inputs_embeds"].to("cuda", torch.bfloat16),
        audio_attention_mask=batch["audio_attention_mask"].cuda(),
        audio_embed_sizes=batch["audio_embed_sizes"].cuda(),
        max_new_tokens=args.max_new_tokens,
    )
    generated = out[0, ref["input_ids"].shape[1] :].tolist()
    if ps.get_data_parallel_rank() == 0:
        print(f"Generated {len(generated)} tokens: {generated}")


if __name__ == "__main__":
    main()
