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

"""BigBench-Audio (BBH) speech-to-text eval for a native SpeechCascadesModel.

Mirrors the load-once + greedy-decode infra of ``eval_openasr.py``: one process initializes the
process group (30-min timeout) and TP=PP=1 model-parallel state, loads the 4.3B speech-cascades model
once, then iterates the BigBench-Audio manifest (a JSONL of ``{UUID, CorpusName, Transcription
(=ground-truth answer), WavPath, Question}``). For each utterance the spoken question audio is decoded
with soundfile (``Audio(decode=False)`` semantics, no torchcodec), log-mel features are extracted with
the vendored ``LogFbankProcessor``, the QA prompt is built with the vendored conversation builder
(``sqa`` formatter, a fixed instruction since the question is in the audio, placeholders = ``-1`` ×
``embed_size``), and the model greedily decodes an answer. Samples are sharded across data-parallel
ranks and gathered on rank 0, which writes both ``predictions.json`` (``audio_ids`` /
``generated_texts``) and ``judge_input.json`` (the GPT-4.1 judge schema from ``bbh_gpt_judge.py``).

Scoring is a separate CPU step: run ``bbh_gpt_judge.py --input <out>/judge_input.json`` to get the
CORRECT/total accuracy. Launch under the platform MPI env or ``torch.distributed.run``
(TP=PP=1, world size == DP)::

    TORCHDYNAMO_DISABLE=1 python examples/models/speech_cascades/eval_bbh.py \
        --hf-path /path/stage1_hf --checkpoint /path/checkpoints/iter_0012000 \
        --manifest /datablob1/.../bigbench_audio/manifest.jsonl --audio-root /datablob1/.../audio \
        --tokenizer /path/Qwen3.5-4B --output-dir /out/eval_output/bbh
"""

import argparse
import json
import os


def _patch_transformers_compat() -> None:
    """Materialize transformers.utils symbols some vendored modeling files import.

    Importing the bridge pulls in unrelated modeling files (e.g. ``bailing``) that do
    ``from transformers.utils import is_flash_attn_greater_or_equal_2_10``. Under the lazy
    module layout of newer transformers this name is not always exported at import time,
    so bind it eagerly (no-op when it already resolves).
    """
    try:
        import transformers.utils as tu
    except ImportError:
        return
    if not hasattr(tu, "is_flash_attn_greater_or_equal_2_10"):
        if hasattr(tu, "is_flash_attn_greater_or_equal"):
            tu.is_flash_attn_greater_or_equal_2_10 = lambda: tu.is_flash_attn_greater_or_equal("2.1.0")
        else:
            tu.is_flash_attn_greater_or_equal_2_10 = lambda: False
    if not hasattr(tu, "is_flash_attn_2_available"):
        tu.is_flash_attn_2_available = lambda: False


_patch_transformers_compat()

import numpy as np  # noqa: E402
import soundfile  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from bbh_gpt_judge import format_judge_inputs, load_manifest  # noqa: E402  (colocated judge harness)
from megatron.core import parallel_state as ps  # noqa: E402
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed  # noqa: E402

import megatron.bridge.models.speech_cascades.bridge  # noqa: E402, F401  (registers the bridge)
from megatron.bridge import AutoBridge  # noqa: E402
from megatron.bridge.models.speech_cascades.generate import greedy_generate  # noqa: E402


IGNORE_INDEX = -100
# The audio already speaks the question, so the text turn is just a fixed answer instruction.
DEFAULT_PROMPT = "Answer the question."


def _load_model(hf_path: str, checkpoint: str):
    from megatron.bridge.models.speech_cascades import dcp_compat  # noqa: F401  (torch>=2.6 no_dist shim)
    from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model = bridge.to_megatron_model(load_weights=False, wrap_with_ddp=False)
    model = model if isinstance(model, list) else [model]
    _load_model_weights_from_checkpoint(checkpoint, model, strict=False)
    for m in model:
        m.eval()
    return model[0].module if hasattr(model[0], "module") else model[0]


def _resolve_wav_path(record: dict, audio_root: str | None) -> str:
    wav = record.get("WavPath") or record.get("wav_path") or record.get("audio")
    if not wav:
        raise ValueError(f"Manifest record {record.get('UUID')} has no WavPath")
    if audio_root and not os.path.isabs(wav):
        return os.path.join(audio_root, wav)
    return wav


def _decode_audio(path: str) -> tuple[np.ndarray, int]:
    """Return (waveform 1-D float32, sampling_rate); mono-downmix if needed (no torchcodec)."""
    array, fs = soundfile.read(path)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array, int(fs)


def _embed_size(num_frames: int, compression: int = 8) -> int:
    """Placeholder count = encoder-downsampled frames (streaming compression)."""
    return (num_frames - 7) // compression


def _build_prompt(tokenizer, prompt_text: str, embed_size: int):
    """Build (input_ids, labels) with the vendored ``sqa`` QA template (audio placeholders=-1)."""
    from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.conversation_utils import (
        conv_templates,
    )
    from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.utils import (
        add_audio_text_in_conversation,
    )

    conv = conv_templates["qwen3_audio_nothink_instruct"].copy()
    conv.tokenizer = tokenizer
    # Empty target text: eval only needs the prompt tokens up to the assistant answer start.
    conv = add_audio_text_in_conversation(
        [[""]], [prompt_text], ["en"], [None], conv, "sqa", True, False, False
    )
    _, input_id, label, _ = conv.get_prompt(
        is_train=True, audio_input_id=-1, only_add_first_audio_occurrence=False, num_audio_tokens=[embed_size]
    )
    return input_id, label


def _eos_ids(tokenizer) -> list[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid not in ids:
            ids.append(tid)
    return ids


def _decode_manifest(manifest, model, processor, tokenizer, eos, prompt_text, args, dp_rank, dp_world):
    """Greedily answer each BBH utterance on this DP shard; return [(uuid, answer)] pairs."""
    pairs: list[tuple[str, str]] = []
    for idx in range(dp_rank, len(manifest), dp_world):
        if len(pairs) >= args.max_samples:
            break
        record = manifest[idx]
        uuid = record["UUID"]
        array, fs = _decode_audio(_resolve_wav_path(record, args.audio_root))

        features = processor.extract_features(array, fs)  # T x 80 float32
        num_frames = features.shape[0]
        embed_size = _embed_size(num_frames)
        if embed_size <= 0:
            pairs.append((uuid, ""))
            continue

        input_id, label = _build_prompt(tokenizer, prompt_text, embed_size)
        labels = torch.tensor(label)
        answer_mask = labels != IGNORE_INDEX
        if not bool(answer_mask.any()):
            pairs.append((uuid, ""))
            continue
        answer_start = int(answer_mask.nonzero()[0].item())
        prompt_ids = torch.tensor(input_id[:answer_start], dtype=torch.long).unsqueeze(0).cuda()

        feats = torch.from_numpy(features).unsqueeze(0).to("cuda", torch.bfloat16)  # [1, T, 80]
        attn = torch.ones(1, num_frames, dtype=torch.long, device="cuda")
        sizes = torch.tensor([embed_size], dtype=torch.long, device="cuda")
        gen = greedy_generate(
            model,
            prompt_ids,
            audio_input_features=feats,
            audio_attention_mask=attn,
            audio_embed_sizes=sizes,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos,
        )
        hyp_ids = gen[0, answer_start:].tolist()
        answer = tokenizer.decode(hyp_ids, skip_special_tokens=True)
        pairs.append((uuid, answer))
    return pairs


def main() -> None:
    """Load the model once, decode BBH answers across DP ranks, and write judge inputs on rank 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Converted stage-1 HF dir (model config).")
    parser.add_argument("--checkpoint", required=True, help="Megatron dist checkpoint dir (iter_*).")
    parser.add_argument("--manifest", required=True, help="BigBench-Audio JSONL manifest.")
    parser.add_argument("--audio-root", default=None, help="Prefix for relative WavPath entries.")
    parser.add_argument("--tokenizer", required=True, help="Qwen3.5-4B tokenizer path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Fixed QA instruction text turn.")
    parser.add_argument("--max-samples", type=int, default=100000, help="Cap examples per rank.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", required=True, help="Dir for predictions.json + judge_input.json.")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        # Long timeout: the 4.3B dist-checkpoint load + first-forward compile can exceed the 600s
        # default before all ranks reach the next collective, which otherwise trips a NCCL timeout.
        from datetime import timedelta

        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)
    ps.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(0)
    dp_rank = ps.get_data_parallel_rank()
    dp_world = ps.get_data_parallel_world_size()

    from transformers import AutoTokenizer

    from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.audio_processors.logfbank_processor import (
        LogFbankProcessor,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = _load_model(args.hf_path, args.checkpoint)
    processor = LogFbankProcessor()
    eos = _eos_ids(tokenizer)
    manifest = load_manifest(args.manifest)

    pairs = _decode_manifest(manifest, model, processor, tokenizer, eos, args.prompt, args, dp_rank, dp_world)

    # Gather all (uuid, answer) shards to rank 0 so predictions cover the whole manifest.
    gathered: list[list[tuple[str, str]]] = [None] * dp_world  # type: ignore[list-item]
    dist.all_gather_object(gathered, pairs, group=ps.get_data_parallel_group())

    if dp_rank == 0:
        answers = {uuid: ans for shard in gathered for uuid, ans in shard}
        predictions = [{"audio_ids": [uuid], "generated_texts": [answers[uuid]]} for uuid in answers]
        judge_items = format_judge_inputs(manifest, answers)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "judge_input.json"), "w", encoding="utf-8") as f:
            json.dump(judge_items, f, ensure_ascii=False, indent=2)
        print(f"[bbh] decoded={len(answers)} ranks={dp_world} | wrote predictions.json + judge_input.json "
              f"to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
