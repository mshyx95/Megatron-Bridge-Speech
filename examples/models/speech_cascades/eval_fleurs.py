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

"""Fleurs speech-translation (S2T) BLEU eval for a native SpeechCascadesModel.

Mirrors the production ``run_fleurs_cascade_qwen35.py`` metric path, but replaces the two-stage
Phi-4MM-ASR -> Qwen-text cascade with the native speech-cascades model doing audio -> greedy
translation directly. For each ``(source, target)`` language direction it reads the Fleurs TSV
(``<data-root>/<SRC>/<TGT>/test.tsv``: ``row_id``, a one-element list of audio paths, and a list of
``{"messages": [user, assistant]}`` turns where the assistant content is the target-language
reference translation), extracts cascades log-mel features with the vendored ``LogFbankProcessor``
(16 kHz, 80-mel, 10 ms hop), builds the AST prompt with the vendored conversation builder
(``prompt_formatter="ast"``, ``languages=[[src, tgt]]``; audio placeholders = ``-1`` × ``embed_size``
where ``embed_size = (T-7)//compression``), greedily decodes, and scores **corpus BLEU + chrF** with
the same library and tokenization as the reference (``sacrebleu``: ``zh`` -> ``"zh"``, ``ja`` ->
``"ja-mecab"``, otherwise ``"13a"``). Directions are sharded across the data-parallel ranks and
scored once on rank 0.

Launch under the platform MPI env or ``torch.distributed.run`` (TP=PP=1, world size == DP):

    python examples/models/speech_cascades/eval_fleurs.py \
        --hf-path /path/stage1_hf --checkpoint /path/checkpoints/iter_0010000 \
        --data-root /datablob1/.../fleurs_s2t --tokenizer /path/Qwen3.5-4B \
        --languages EN DE ES FR IT JA PT ZH --output-dir /out
"""

import argparse
import ast
import csv
import json
import os
from pathlib import Path


IGNORE_INDEX = -100

# Fleurs directions used by the reference (uppercase dir names -> lowercase prompt-builder locales).
_DEFAULT_LANGUAGES = ("EN", "DE", "ES", "FR", "IT", "JA", "PT", "ZH")
_LOCALE = {
    "EN": "en", "DE": "de", "ES": "es", "FR": "fr", "IT": "it",
    "JA": "ja", "PT": "pt", "ZH": "zh", "KO": "ko",
}


class FleursBleuScorer:
    """Corpus BLEU + chrF scorer matching the reference ``SacrebleuScorer``.

    Uses ``sacrebleu`` with the reference tokenization: target ``zh`` -> ``"zh"``, ``ja`` ->
    ``"ja-mecab"``, otherwise ``"13a"``. ``sacrebleu`` is imported lazily so this module (and the
    ``--help`` output) still load when the library is absent from the image; ``score`` then raises a
    clear ImportError.
    """

    def __init__(self, target_language: str):
        lang = target_language[:2].lower()
        if lang == "zh":
            self.tokenize = "zh"
        elif lang == "ja":
            self.tokenize = "ja-mecab"
        else:
            self.tokenize = "13a"
        self.refs: list[str] = []
        self.preds: list[str] = []

    def add_string(self, ref: str, pred: str) -> None:
        self.refs.append(ref)
        self.preds.append(pred)

    def score(self) -> tuple[float, float]:
        """Return ``(bleu, chrf)`` corpus scores (0-100)."""
        try:
            import sacrebleu
        except ImportError as exc:  # pragma: no cover - image-dependent
            raise ImportError(
                "Please install sacrebleu (`pip install sacrebleu`) to compute Fleurs translation "
                "BLEU; this matches the reference run_fleurs_cascade_qwen35.py metric."
            ) from exc

        bleu = sacrebleu.BLEU(tokenize=self.tokenize).corpus_score(self.preds, [self.refs]).score
        chrf = sacrebleu.CHRF().corpus_score(self.preds, [self.refs]).score
        return bleu, chrf


def _parse_tsv(path: Path, source_language: str, target_language: str) -> list[dict]:
    """Parse one Fleurs ``test.tsv`` into ``{audio_id, audio_path, source/target, label}`` rows."""
    examples: list[dict] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row_index, row in enumerate(csv.reader(stream, delimiter="\t")):
            if len(row) != 3:
                raise ValueError(f"{path}:{row_index + 1}: expected 3 TSV columns")
            row_id, serialized_paths, serialized_messages = row
            audio_paths = ast.literal_eval(serialized_paths)
            message_groups = ast.literal_eval(serialized_messages)
            if not isinstance(audio_paths, list) or len(audio_paths) != 1:
                raise ValueError(f"{path}:{row_index + 1}: expected one audio path")
            for turn_index, group in enumerate(message_groups):
                messages = group["messages"]
                if len(messages) != 2:
                    raise ValueError(f"{path}:{row_index + 1}: expected user/assistant messages")
                examples.append(
                    {
                        "audio_id": f"{row_id}_{turn_index}",
                        "audio_path": audio_paths[0],
                        "source_language": source_language,
                        "target_language": target_language,
                        "label": messages[1]["content"],
                    }
                )
    return examples


def _load_directions(args) -> dict[tuple[str, str], list[dict]]:
    """Load every ``src != tgt`` direction TSV under ``--data-root`` for the requested languages."""
    directions: dict[tuple[str, str], list[dict]] = {}
    for source_language in args.languages:
        for target_language in args.languages:
            if source_language == target_language:
                continue
            path = args.data_root / source_language / target_language / "test.tsv"
            if not path.is_file():
                continue
            examples = _parse_tsv(path, source_language, target_language)
            if args.limit_per_direction:
                examples = examples[: args.limit_per_direction]
            directions[(source_language, target_language)] = examples
    return directions


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


def _decode_audio(audio_path: str):
    """Return ``(waveform 1-D float32, sampling_rate)`` for a Fleurs audio file path."""
    import numpy as np
    import soundfile

    array, fs = soundfile.read(audio_path)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array.astype(np.float32), int(fs)


def _embed_size(num_frames: int, compression: int = 8) -> int:
    """Placeholder count = encoder-downsampled frames (streaming compression)."""
    return (num_frames - 7) // compression


def _build_prompt(tokenizer, reference: str, embed_size: int, src_lang: str, tgt_lang: str):
    """Build (input_ids, labels) with the vendored AST conversation template (audio placeholders=-1)."""
    from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.conversation_utils import (
        conv_templates,
    )
    from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.utils import (
        add_audio_text_in_conversation,
    )

    conv = conv_templates["qwen3_audio_nothink_instruct"].copy()
    conv.tokenizer = tokenizer
    conv = add_audio_text_in_conversation(
        [[reference]], [None], [[src_lang, tgt_lang]], [None], conv, "ast", True, False, False
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


def _eval_one_direction(source, target, examples, model, processor, tokenizer, eos, args, dp_rank, dp_world):
    """Decode one Fleurs direction across DP ranks and print/write its corpus BLEU (scored on rank 0)."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps

    from megatron.bridge.models.speech_cascades.generate import greedy_generate

    src_locale = _LOCALE.get(source, source.lower())
    tgt_locale = _LOCALE.get(target, target.lower())

    # (reference translation, hypothesis) pairs; scoring runs once on rank 0 for exact parity.
    pairs: list[tuple[str, str]] = []
    for idx in range(dp_rank, len(examples), dp_world):
        if len(pairs) >= args.max_samples:
            break
        sample = examples[idx]
        reference = sample["label"]
        if not reference:
            continue
        audio_path = sample["audio_path"]
        if args.audio_root:
            audio_path = os.path.join(args.audio_root, audio_path)
        array, fs = _decode_audio(audio_path)

        features = processor.extract_features(array, fs)  # T x 80 float32
        num_frames = features.shape[0]
        embed_size = _embed_size(num_frames)
        if embed_size <= 0:
            continue

        input_id, label = _build_prompt(tokenizer, reference, embed_size, src_locale, tgt_locale)
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
        pairs.append((reference, hypothesis))

    # Gather all pairs to rank 0 so corpus BLEU is scored over the whole direction (parity).
    gathered: list[list[tuple[str, str]]] = [None] * dp_world  # type: ignore[list-item]
    dist.all_gather_object(gathered, pairs, group=ps.get_data_parallel_group())

    if dp_rank == 0:
        corpus = [p for shard in gathered for p in shard]
        scorer = FleursBleuScorer(tgt_locale)
        for ref, hyp in corpus:
            scorer.add_string(ref, hyp)
        bleu, chrf = scorer.score()
        print(f"[fleurs] direction={source}->{target} ranks={dp_world} | utts={len(corpus)} | "
              f"BLEU={bleu:.2f} | chrF={chrf:.2f}", flush=True)
        if args.output_dir:
            records = [{"reference": r, "hypothesis": h} for r, h in corpus]
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, f"bleu_{source}_{target}.json"), "w") as f:
                json.dump({"source": source, "target": target, "bleu": bleu, "chrf": chrf,
                           "utts": len(corpus), "records": records}, f)
        return {"source": source, "target": target, "bleu": bleu, "chrf": chrf, "utts": len(corpus)}
    return None


def main() -> None:
    """Load the model once and decode each Fleurs direction, reporting corpus BLEU per direction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Converted stage-1 HF dir (model config).")
    parser.add_argument("--checkpoint", required=True, help="Megatron dist checkpoint dir (iter_*).")
    parser.add_argument("--data-root", required=True, type=Path, help="Fleurs S2T root (<SRC>/<TGT>/test.tsv).")
    parser.add_argument("--tokenizer", required=True, help="Qwen3.5-4B tokenizer path.")
    parser.add_argument("--languages", nargs="+", default=list(_DEFAULT_LANGUAGES),
                        help="Uppercase Fleurs language codes; all src!=tgt directions are evaluated.")
    parser.add_argument("--audio-root", default=None, help="Optional prefix for relative audio paths in the TSV.")
    parser.add_argument("--limit-per-direction", type=int, default=0, help="Cap examples per direction (0=all).")
    parser.add_argument("--max-samples", type=int, default=100000, help="Cap examples per rank per direction.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default=None, help="Directory for per-direction bleu_<src>_<tgt>.json.")
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    _patch_transformers_compat()

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

    directions = _load_directions(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = _load_model(args.hf_path, args.checkpoint)
    processor = LogFbankProcessor()
    eos = _eos_ids(tokenizer)

    summary = []
    for (source, target), examples in directions.items():
        result = _eval_one_direction(
            source, target, examples, model, processor, tokenizer, eos, args, dp_rank, dp_world
        )
        if result is not None:
            summary.append(result)

    if dp_rank == 0 and summary:
        macro = sum(r["bleu"] for r in summary) / len(summary)
        print(f"[fleurs] directions={len(summary)} | macro_BLEU={macro:.2f}", flush=True)
        if args.output_dir:
            with open(os.path.join(args.output_dir, "bleu_summary.json"), "w") as f:
                json.dump({"macro_bleu": macro, "directions": summary}, f)


if __name__ == "__main__":
    main()
