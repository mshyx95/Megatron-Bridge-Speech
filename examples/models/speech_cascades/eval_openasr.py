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

"""HuggingFace Open-ASR-Leaderboard WER eval for a native SpeechCascadesModel.

Mirrors the production ``run_opensr_eval.sh`` ``openasr`` path: loads the ESB test parquet datasets
(librispeech-clean/other, earnings22, spgispeech, ami, gigaspeech) via ``datasets.load_dataset``,
extracts cascades log-mel features with the vendored ``LogFbankProcessor`` (16 kHz, 80-mel, 10 ms
hop), builds the exact training ASR prompt with the vendored conversation builder (audio placeholders
= ``-1`` × ``embed_size`` where ``embed_size = (T-7)//compression``), greedily decodes, and reports
corpus WER using the **production-parity** scorer: the vendored custom ``EnglishTextNormalizer``
(acronym/name/compound normalization) plus ``kaldialign.batch_error_rate(merge_compounds=True)`` —
the official Open ASR Leaderboard metric (word edit distance fallback if kaldialign is absent).
Data is sharded across data-parallel ranks and scored once on rank 0.

Launch under the platform MPI env or ``torch.distributed.run`` (TP=PP=1, world size == DP):

    python examples/models/speech_cascades/eval_openasr.py \
        --hf-path /path/stage1_hf --checkpoint /path/checkpoints/iter_0010000 \
        --data-dir /datablob1/users/ruchaofan/Evaluation/OpenASR/esb-datasets-test-only-sorted \
        --subsets librispeech-clean librispeech-other --tokenizer /path/Qwen3.5-4B --output-dir /out
"""

import argparse
import io
import json
import logging
import os
import time


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
from megatron.core import parallel_state as ps  # noqa: E402
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed  # noqa: E402

import megatron.bridge.models.speech_cascades.bridge  # noqa: E402, F401  (registers the bridge)
from megatron.bridge import AutoBridge  # noqa: E402
from megatron.bridge.models.speech_cascades.generate import greedy_generate  # noqa: E402


IGNORE_INDEX = -100
logger = logging.getLogger(__name__)
# Datasets that use the lexical-format ASR prompt on the leaderboard.
_LEXICAL = {"librispeech-clean", "librispeech-other", "gigaspeech"}


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


def _get_text(sample: dict) -> str | None:
    for key in ("text", "sentence", "normalized_text", "transcript", "transcription"):
        if key in sample and sample[key]:
            return sample[key]
    return None


def _decode_audio(audio_field) -> tuple[np.ndarray, int]:
    """Return (waveform 1-D float, sampling_rate) from a HF datasets audio field."""
    if isinstance(audio_field, dict):
        if audio_field.get("array") is not None:
            return np.asarray(audio_field["array"], dtype=np.float32), int(audio_field["sampling_rate"])
        array, fs = soundfile.read(io.BytesIO(audio_field["bytes"]))
        return array.astype(np.float32), int(fs)
    array, fs = soundfile.read(io.BytesIO(audio_field))
    return array.astype(np.float32), int(fs)


def _embed_size(num_frames: int, compression: int = 8) -> int:
    """Placeholder count = encoder-downsampled frames (streaming compression)."""
    return (num_frames - 7) // compression


def _build_prompt(tokenizer, reference: str, embed_size: int, subset: str):
    """Build (input_ids, labels) with the vendored ASR conversation template (audio placeholders=-1)."""
    from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.conversation_utils import (
        conv_templates,
    )
    from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.utils import (
        add_audio_text_in_conversation,
    )

    prompt_formatter = "asr-lexical" if subset in _LEXICAL else "asr"
    conv = conv_templates["qwen3_audio_nothink_instruct"].copy()
    conv.tokenizer = tokenizer
    conv = add_audio_text_in_conversation(
        [[reference]], [None], ["en"], [None], conv, prompt_formatter, True, False, False
    )
    _, input_id, label, _ = conv.get_prompt(
        is_train=True, audio_input_id=-1, only_add_first_audio_occurrence=False, num_audio_tokens=[embed_size]
    )
    return input_id, label


def _build_scorer(language: str):
    """Return a production-parity WER scorer (custom TN + kaldialign for English)."""
    from megatron.bridge.models.speech_cascades.eval_metrics import HFWerScorer

    lang = language.lower()
    sub_task = "hfwer_en" if lang in ("en", "eng") else f"hfwer_{lang}"
    return HFWerScorer(sub_task=sub_task)


def _eos_ids(tokenizer) -> list[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid not in ids:
            ids.append(tid)
    return ids


def _eval_one_subset(subset, min_duration, model, processor, tokenizer, eos, language, args, dp_rank, dp_world):
    """Decode one ESB subset across DP ranks and print/write its corpus WER (scored on rank 0)."""
    from datasets import Audio, load_dataset

    scorer = _build_scorer(language)
    report_progress = ps.get_tensor_model_parallel_rank() == 0
    started = time.monotonic()
    capped = 0
    if report_progress:
        logger.info("[openasr] starting subset=%s dp_rank=%s/%s", subset, dp_rank, dp_world)

    if subset == "librispeech-clean":
        dataset = load_dataset(args.data_dir, "librispeech", split="test.clean")
    elif subset == "librispeech-other":
        dataset = load_dataset(args.data_dir, "librispeech", split="test.other")
    else:
        dataset = load_dataset(args.data_dir, subset, split="test")

    # Disable datasets' Audio auto-decode (datasets>=4 routes it through torchcodec, which is not
    # in this image); _decode_audio reads the raw bytes with soundfile instead.
    if "audio" in dataset.features:
        dataset = dataset.cast_column("audio", Audio(decode=False))

    # (normalized reference, raw hypothesis) pairs; scoring runs once on rank 0 for exact parity.
    pairs: list[tuple[str, str]] = []
    for idx in range(dp_rank, len(dataset), dp_world):
        if len(pairs) >= args.max_samples:
            break
        sample_started = time.monotonic()
        sample = dataset[idx]
        reference = _get_text(sample)
        if not reference:
            continue
        array, fs = _decode_audio(sample["audio"])
        if min_duration and len(array) / fs < min_duration:
            continue

        features = processor.extract_features(array, fs)  # T x 80 float32
        num_frames = features.shape[0]
        embed_size = _embed_size(num_frames)
        if embed_size <= 0:
            continue

        input_id, label = _build_prompt(tokenizer, reference, embed_size, subset)
        labels = torch.tensor(label)
        answer_mask = labels != IGNORE_INDEX
        if not bool(answer_mask.any()):
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
        hypothesis = tokenizer.decode(hyp_ids, skip_special_tokens=True)
        # Reference is normalized once here; hypothesis is normalized inside the scorer (prod parity).
        pairs.append((scorer.normalize_reference(reference), hypothesis))
        capped += int(len(hyp_ids) >= args.max_new_tokens and (not hyp_ids or hyp_ids[-1] not in eos))
        if report_progress and (len(pairs) == 1 or len(pairs) % 25 == 0):
            logger.info(
                "[openasr] subset=%s dp_rank=%s completed=%s dataset_index=%s/%s "
                "elapsed_s=%.1f sample_s=%.2f tokens=%s capped=%s",
                subset, dp_rank, len(pairs), idx, len(dataset), time.monotonic() - started,
                time.monotonic() - sample_started, len(hyp_ids), capped,
            )

    if report_progress:
        logger.info(
            "[openasr] decode_done subset=%s dp_rank=%s completed=%s elapsed_s=%.1f; gathering scores",
            subset, dp_rank, len(pairs), time.monotonic() - started,
        )

    # Gather all pairs to rank 0 so kaldialign batch scoring runs over the whole corpus (parity).
    gathered: list[list[tuple[str, str]]] = [None] * dp_world  # type: ignore[list-item]
    dist.all_gather_object(gathered, pairs, group=ps.get_data_parallel_group())

    if dp_rank == 0 and ps.get_tensor_model_parallel_rank() == 0:
        corpus = [p for shard in gathered for p in shard]
        for ref_norm, raw_hyp in corpus:
            scorer.add_string(ref_norm, raw_hyp)
        wer = scorer.score() / 100.0
        print(f"[openasr] subset={subset} ranks={dp_world} | utts={len(corpus)} | "
              f"ref_words={scorer.ref_length} | WER={wer:.4f}", flush=True)
        if args.output_dir:
            records = [{"reference": r, "hypothesis": h} for r, h in corpus]
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, f"wer_{subset}.json"), "w") as f:
                json.dump({"subset": subset, "wer": wer, "utts": len(corpus),
                           "ref_words": scorer.ref_length, "records": records}, f)


def main() -> None:
    """Load the model once and decode each ESB subset, reporting corpus WER per subset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Converted stage-1 HF dir (model config).")
    parser.add_argument("--checkpoint", required=True, help="Megatron dist checkpoint dir (iter_*).")
    parser.add_argument("--data-dir", required=True, help="ESB esb-datasets-test-only-sorted dir.")
    parser.add_argument("--subsets", required=True, nargs="+", help="One or more ESB subsets to eval.")
    parser.add_argument("--tokenizer", required=True, help="Qwen3.5-4B tokenizer path.")
    parser.add_argument("--max-samples", type=int, default=100000, help="Cap examples per rank.")
    parser.add_argument("--min-duration", type=float, default=0.0, help="Global min-clip floor; ami/gigaspeech default to 1.0.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default=None, help="Directory for per-subset wer_<subset>.json.")
    parser.add_argument("--lora", action="store_true", help="Restore Speech Cascades LoRA and audio trainables.")
    parser.add_argument("--base-checkpoint", help="Original frozen-base Megatron iteration directory for LoRA.")
    parser.add_argument("--expert-scaling", choices=("normalized", "legacy"))
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if args.lora and args.expert_scaling is None:
        parser.error("--lora requires explicit --expert-scaling")
    if not args.lora and (args.base_checkpoint or args.tensor_model_parallel_size != 1):
        parser.error("Base checkpoint and TP overrides currently require --lora")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        # Long timeout: the 4.3B dist-checkpoint load + first-forward compile can exceed the 600s
        # default before all ranks reach the next collective, which otherwise trips a NCCL timeout.
        from datetime import timedelta

        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)
    ps.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=args.tensor_model_parallel_size,
    )
    model_parallel_cuda_manual_seed(0)
    dp_rank = ps.get_data_parallel_rank()
    dp_world = ps.get_data_parallel_world_size()

    from transformers import AutoTokenizer

    from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.audio_processors.logfbank_processor import (
        LogFbankProcessor,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.lora:
        from lora_eval import build_lora_model

        model = build_lora_model(
            args.hf_path,
            args.checkpoint,
            base_checkpoint=args.base_checkpoint,
            expert_scaling=args.expert_scaling,
            tensor_parallel_size=args.tensor_model_parallel_size,
        )
    else:
        model = _load_model(args.hf_path, args.checkpoint)
    processor = LogFbankProcessor()
    eos = _eos_ids(tokenizer)

    for subset in args.subsets:
        # Leaderboard uses a 1s floor for the far-field/noisy ami and gigaspeech "clean" subsets.
        min_dur = args.min_duration or (1.0 if subset in {"ami", "gigaspeech"} else 0.0)
        _eval_one_subset(subset, min_dur, model, processor, tokenizer, eos, args.language, args, dp_rank, dp_world)


if __name__ == "__main__":
    main()
