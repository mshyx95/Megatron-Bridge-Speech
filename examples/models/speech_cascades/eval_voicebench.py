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

"""VoiceBench (9-task macro) eval for a native SpeechCascadesModel.

Ports the production S2T VoiceBench pipeline (``run_voicebench_cascade_text_qwen35.py`` +
``convert_voice_bench.py`` generation, ``evaluate_voicebench_s2t.py`` scoring) onto the native
audio path. VoiceBench is fully audio-only: each item is one spoken question (the MCQ subsets speak
the options and a "select A/B/C/D" instruction) and the model sees only the audio with an empty text
instruction, matching the official audio modality. This runner loads the 4.3B checkpoint **once**
and greedily decodes every subset, mirroring ``eval_openasr.py`` (30-min process-group timeout,
``TORCHDYNAMO_DISABLE=1`` set by the launcher, ``datasets.Audio(decode=False)`` + soundfile), then
writes the per-subset ``<subset>_results.jsonl`` files consumed by the official VoiceBench evaluators.

The **nine** VoiceBench subsets and their official evaluators:

    alpacaeval / commoneval / wildvoice  -> open  (GPT-4.1 judge, mean rating on a 1-5 scale)
    sd-qa                                -> qa    (GPT-4.1 yes/no majority vote, % correct)
    mmsu / openbookqa                    -> mcq   (letter-extraction accuracy, %)
    ifeval                               -> ifeval(strict/loose instruction following, mean in [0,1])
    advbench                             -> harm  (refusal-string rate, mean in [0,1])
    bbh                                  -> bbh   (answer-extraction accuracy, %)

The **9-task macro** is the unweighted average across the nine subsets after normalizing every
subset headline to a percentage: the three open GPT ratings are scaled x20 (1-5 -> 0-100), ifeval's
``final`` and harm's ``refusal_rate`` are scaled x100 (0-1 -> 0-100), and mcq/bbh accuracy and the
sd-qa GPT percentage are already percentages. This matches ``S2T_COMPARE_RESULTS.md`` ("unweighted
average across the nine subsets after normalizing the three 1-5 scores to percentages").

Two modes:

    # 1. Generate per-subset responses (loads the model once; run under MPI / torch.distributed.run).
    python examples/models/speech_cascades/eval_voicebench.py generate \
        --hf-path /path/stage1_hf --checkpoint /path/checkpoints/iter_0004000 \
        --manifest /datablob1/users/.../voicebench/voicebench.jsonl \
        --tokenizer /path/Qwen3.5-4B --output-dir /out/voicebench

    # 2. Aggregate the official per-subset metric JSONs into the 9-task macro (CPU, no model).
    python examples/models/speech_cascades/eval_voicebench.py macro --metrics-dir /out/voicebench/metrics
"""

import argparse
import io
import json
import os
from collections.abc import Mapping
from typing import Any


IGNORE_INDEX = -100

# The nine VoiceBench subsets and the official evaluator used to score each (mirrors the reference
# ``SUBSET_EVALUATORS`` in ``evaluate_voicebench_s2t.py``).
SUBSET_EVALUATORS: dict[str, str] = {
    "alpacaeval": "open",
    "commoneval": "open",
    "wildvoice": "open",
    "sd-qa": "qa",
    "mmsu": "mcq",
    "openbookqa": "mcq",
    "ifeval": "ifeval",
    "advbench": "harm",
    "bbh": "bbh",
}

# Manifest ``CorpusName`` -> subset key (``convert_voice_bench.py`` uses ``test_name.lower()``).
CORPUS_TO_SUBSET: dict[str, str] = {
    "AlpacaEval": "alpacaeval",
    "CommonEval": "commoneval",
    "SD-QA": "sd-qa",
    "MMSU": "mmsu",
    "OpenBookQA": "openbookqa",
    "IFEval": "ifeval",
    "AdvBench": "advbench",
    "bbh": "bbh",
    "WildVoice": "wildvoice",
}

# Official example counts per subset (mirrors ``prepare_voicebench_latest.py`` EXPECTED_COUNTS, keyed
# by subset). Used only to warn about an incomplete manifest, never to reject one.
EXPECTED_COUNTS: dict[str, int] = {
    "alpacaeval": 636,
    "commoneval": 200,
    "sd-qa": 553,
    "mmsu": 3074,
    "openbookqa": 455,
    "ifeval": 345,
    "advbench": 520,
    "bbh": 1000,
    "wildvoice": 1000,
}

# Per-subset headline metric extracted from each official evaluator's return dict, and the factor
# that maps it onto the 0-100 percentage scale the macro averages:
#   open   -> {'gpt': mean rating in [1, 5]}          -> x20
#   qa     -> {'gpt': % correct in [0, 100], ...}     -> x1  (already a percentage)
#   mcq    -> {'acc': % in [0, 100], 'fail': ...}     -> x1
#   ifeval -> {..., 'final': mean in [0, 1]}          -> x100
#   harm   -> {'refusal_rate': mean in [0, 1]}        -> x100
#   bbh    -> {'acc': % in [0, 100]}                  -> x1
SUBSET_HEADLINE: dict[str, tuple[str, float]] = {
    "alpacaeval": ("gpt", 20.0),
    "commoneval": ("gpt", 20.0),
    "wildvoice": ("gpt", 20.0),
    "sd-qa": ("gpt", 1.0),
    "mmsu": ("acc", 1.0),
    "openbookqa": ("acc", 1.0),
    "ifeval": ("final", 100.0),
    "advbench": ("refusal_rate", 100.0),
    "bbh": ("acc", 1.0),
}


# --------------------------------------------------------------------------------------------------
# Scoring / macro aggregation (pure, CPU-only, unit-tested).
# --------------------------------------------------------------------------------------------------
def subset_headline_metric(subset: str, metrics: Mapping[str, Any]) -> float:
    """Extract one subset's headline percentage from its official evaluator's metric dict.

    Args:
        subset: One of the nine VoiceBench subset keys (see :data:`SUBSET_EVALUATORS`).
        metrics: The dict returned by the subset's official evaluator (e.g. ``{'gpt': 4.51}``,
            ``{'acc': 59.27}``, ``{'final': 0.61}``, ``{'refusal_rate': 0.996}``).

    Returns:
        The subset headline score normalized to the 0-100 percentage scale.
    """
    if subset not in SUBSET_HEADLINE:
        raise KeyError(f"Unknown VoiceBench subset: {subset!r}")
    key, scale = SUBSET_HEADLINE[subset]
    if key not in metrics:
        raise KeyError(f"Subset {subset!r} metrics missing headline key {key!r}: {dict(metrics)}")
    return float(metrics[key]) * scale


def compute_macro(per_subset_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the VoiceBench 9-task macro from the nine official per-subset metric dicts.

    The macro is the unweighted mean of the nine subset headline percentages (open GPT ratings
    scaled x20, ifeval/harm scaled x100, mcq/bbh/sd-qa already percentages).

    Args:
        per_subset_metrics: Mapping of subset key -> that subset's official evaluator metric dict.
            Must contain exactly the nine :data:`SUBSET_EVALUATORS` keys.

    Returns:
        ``{"macro": float, "per_subset": {subset: headline_percentage}, "num_subsets": int}``.
    """
    missing = set(SUBSET_EVALUATORS) - set(per_subset_metrics)
    extra = set(per_subset_metrics) - set(SUBSET_EVALUATORS)
    if missing or extra:
        raise ValueError(
            f"VoiceBench macro requires exactly the nine subsets; missing={sorted(missing)}, "
            f"unexpected={sorted(extra)}"
        )
    per_subset = {subset: subset_headline_metric(subset, per_subset_metrics[subset]) for subset in SUBSET_EVALUATORS}
    macro = sum(per_subset.values()) / len(per_subset)
    return {"macro": macro, "per_subset": per_subset, "num_subsets": len(per_subset)}


def score_metrics_dir(metrics_dir: str) -> dict[str, Any]:
    """Read the official ``<subset>_metrics.json`` files from a directory and compute the macro."""
    per_subset_metrics: dict[str, Mapping[str, Any]] = {}
    for subset in SUBSET_EVALUATORS:
        path = os.path.join(metrics_dir, f"{subset}_metrics.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing metrics for subset {subset!r}: {path}")
        with open(path, encoding="utf-8") as stream:
            per_subset_metrics[subset] = json.load(stream)
    return compute_macro(per_subset_metrics)


# --------------------------------------------------------------------------------------------------
# Manifest loading (mirrors prepare_voicebench_latest.py conventions).
# --------------------------------------------------------------------------------------------------
def load_manifest(manifest_path: str) -> list[dict[str, Any]]:
    """Load the VoiceBench manifest (one JSON object per line) and validate its UUIDs."""
    with open(manifest_path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    uuids = [row["UUID"] for row in rows]
    if len(uuids) != len(set(uuids)):
        raise ValueError("VoiceBench manifest contains duplicate UUIDs")
    return rows


def group_by_subset(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group manifest rows by subset key, warning if a subset row count differs from the official one."""
    grouped: dict[str, list[dict[str, Any]]] = {subset: [] for subset in SUBSET_EVALUATORS}
    for row in rows:
        corpus = row.get("CorpusName")
        if corpus not in CORPUS_TO_SUBSET:
            raise ValueError(f"VoiceBench manifest row has unknown CorpusName {corpus!r}")
        grouped[CORPUS_TO_SUBSET[corpus]].append(row)
    for subset, subset_rows in grouped.items():
        expected = EXPECTED_COUNTS[subset]
        if subset_rows and len(subset_rows) != expected:
            print(f"[voicebench] warning: subset={subset} has {len(subset_rows)} rows (official={expected})", flush=True)
    return grouped


# --------------------------------------------------------------------------------------------------
# Native audio generation (loads the model once). Heavy imports are deferred so `--help`, the macro
# path, and unit tests import this module without torch / megatron / a GPU.
# --------------------------------------------------------------------------------------------------
def _patch_transformers_compat() -> None:
    """Materialize transformers.utils symbols some vendored modeling files import (see eval_openasr)."""
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


def _load_model(hf_path: str, checkpoint: str):
    _patch_transformers_compat()
    import megatron.bridge.models.speech_cascades.bridge  # noqa: F401  (registers the bridge)
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.speech_cascades import dcp_compat  # noqa: F401  (torch>=2.6 no_dist shim)
    from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model = bridge.to_megatron_model(load_weights=False, wrap_with_ddp=False)
    model = model if isinstance(model, list) else [model]
    _load_model_weights_from_checkpoint(checkpoint, model, strict=False)
    for m in model:
        m.eval()
    return model[0].module if hasattr(model[0], "module") else model[0]


def _decode_audio(audio_field) -> tuple[Any, int]:
    """Return (waveform 1-D float32, sampling_rate) from a raw path/bytes/HF-datasets audio field."""
    import numpy as np
    import soundfile

    if isinstance(audio_field, str):
        array, fs = soundfile.read(audio_field)
        return array.astype(np.float32), int(fs)
    if isinstance(audio_field, dict):
        if audio_field.get("array") is not None:
            return np.asarray(audio_field["array"], dtype=np.float32), int(audio_field["sampling_rate"])
        array, fs = soundfile.read(io.BytesIO(audio_field["bytes"]))
        return array.astype(np.float32), int(fs)
    array, fs = soundfile.read(io.BytesIO(audio_field))
    return array.astype(np.float32), int(fs)


def _embed_size(num_frames: int, compression: int = 8) -> int:
    """Placeholder count = encoder-downsampled frames (streaming compression), mirroring eval_openasr."""
    return (num_frames - 7) // compression


def _build_audio_prompt(tokenizer, embed_size: int):
    """Build (input_ids, labels) for an audio-only VoiceBench turn (empty text instruction).

    Uses the vendored ``qwen3_audio_nothink_instruct`` template with a raw ``<audio>``-only user turn
    and a one-token dummy answer so the answer boundary can be located; generation slices the prompt
    at that boundary (which includes the assistant generation prompt).
    """
    from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.conversation_utils import (
        conv_templates,
    )
    from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.utils import (
        add_audio_text_in_conversation,
    )

    conv = conv_templates["qwen3_audio_nothink_instruct"].copy()
    conv.tokenizer = tokenizer
    # A two-message (user + assistant) conversation takes the raw-message path; the "sqa" formatter
    # applies no ASR/LID/lexical rewriting, so the audio-only user turn is preserved verbatim.
    messages = [[{"role": "user", "content": "<audio>\n"}, {"role": "assistant", "content": "x"}]]
    conv = add_audio_text_in_conversation(messages, [None], [None], [None], conv, "sqa", True, True, False)
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


def _resolve_wav_path(row: dict[str, Any], audio_root: str | None) -> str:
    """Resolve a manifest ``WavPath`` against an optional root that replaces the manifest prefix."""
    wav_path = row["WavPath"]
    if audio_root:
        return os.path.join(audio_root, os.path.basename(wav_path))
    return wav_path


def _utt_id(row: dict[str, Any], subset: str, index: int) -> str:
    for key in ("UUID", "key", "question_id"):
        if row.get(key) is not None:
            return str(row[key])
    return f"{subset}_{index}"


def _atomic_write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _generate_subset(subset, rows, model, processor, tokenizer, eos, args, dp_rank, dp_world):
    """Greedily decode one VoiceBench subset across DP ranks; rank 0 writes ``<subset>_results.jsonl``."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps

    from megatron.bridge.models.speech_cascades.generate import greedy_generate

    local_records: list[dict[str, Any]] = []
    for index in range(dp_rank, len(rows), dp_world):
        if len(local_records) >= args.max_samples:
            break
        row = rows[index]
        array, fs = _decode_audio(_resolve_wav_path(row, args.audio_root))
        features = processor.extract_features(array, fs)  # T x 80 float32
        num_frames = features.shape[0]
        embed_size = _embed_size(num_frames)
        if embed_size <= 0:
            continue

        input_id, label = _build_audio_prompt(tokenizer, embed_size)
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
        response = tokenizer.decode(hyp_ids, skip_special_tokens=True).strip()
        # Pass every manifest field through (minus embedded audio) so the official evaluators can read
        # their subset-specific keys (reference / id / kwargs / prompt); mirrors convert_voice_bench.
        record = {key: value for key, value in row.items() if key != "audio"}
        record["utt_id"] = _utt_id(row, subset, index)
        record["response"] = response
        local_records.append(record)

    gathered: list[list[dict[str, Any]]] = [None] * dp_world  # type: ignore[list-item]
    dist.all_gather_object(gathered, local_records, group=ps.get_data_parallel_group())
    if dp_rank == 0:
        corpus = [record for shard in gathered for record in shard]
        out_path = os.path.join(args.output_dir, f"{subset}_results.jsonl")
        _atomic_write_jsonl(out_path, corpus)
        print(f"[voicebench] subset={subset} ranks={dp_world} | samples={len(corpus)} -> {out_path}", flush=True)


def run_generate(args: argparse.Namespace) -> None:
    """Load the model once and decode every requested VoiceBench subset."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        # Long timeout: the 4.3B dist-checkpoint load + first-forward compile can exceed the 600s
        # default before all ranks reach the next collective, tripping a NCCL timeout otherwise.
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

    grouped = group_by_subset(load_manifest(args.manifest))
    requested = args.subsets or list(SUBSET_EVALUATORS)
    if dp_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    for subset in requested:
        rows = grouped[subset]
        if not rows:
            print(f"[voicebench] subset={subset} has no manifest rows; skipping", flush=True)
            continue
        _generate_subset(subset, rows, model, processor, tokenizer, eos, args, dp_rank, dp_world)


def run_macro(args: argparse.Namespace) -> None:
    """Aggregate the official per-subset metric JSONs into the 9-task macro (CPU, no model)."""
    summary = score_metrics_dir(args.metrics_dir)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build the ``generate`` (model decode) and ``macro`` (metric aggregation) subcommand parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Decode VoiceBench responses with the native audio model.")
    gen.add_argument("--hf-path", required=True, help="Converted stage-1 HF dir (model config).")
    gen.add_argument("--checkpoint", required=True, help="Megatron dist checkpoint dir (iter_*).")
    gen.add_argument("--manifest", required=True, help="VoiceBench manifest jsonl (CorpusName/UUID/WavPath/...).")
    gen.add_argument("--tokenizer", required=True, help="Qwen3.5-4B tokenizer path.")
    gen.add_argument("--output-dir", required=True, help="Directory for per-subset <subset>_results.jsonl.")
    gen.add_argument("--audio-root", default=None, help="Override the manifest WavPath dir (basename join).")
    gen.add_argument("--subsets", nargs="+", choices=list(SUBSET_EVALUATORS), help="Subset subset (default: all nine).")
    gen.add_argument("--max-samples", type=int, default=100000, help="Cap examples per rank.")
    gen.add_argument("--max-new-tokens", type=int, default=2048, help="Greedy decode budget per item.")
    gen.set_defaults(func=run_generate)

    macro = sub.add_parser("macro", help="Aggregate official per-subset metrics into the 9-task macro.")
    macro.add_argument("--metrics-dir", required=True, help="Dir of official <subset>_metrics.json files.")
    macro.add_argument("--output-json", default=None, help="Optional path to write the macro summary JSON.")
    macro.set_defaults(func=run_macro)
    return parser


def main() -> None:
    """Dispatch to the ``generate`` or ``macro`` subcommand."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
