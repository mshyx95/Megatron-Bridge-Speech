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

"""BFCL V3 function-call eval (micro accuracy) for a native SpeechCascadesModel.

Ports the production ``run_bfcl_v3.sh`` pipeline (``eval_bfcl.py`` -> ``report_bfcl.py``) into a
single, self-contained module with two modes:

* **generate** (default): loads the 4.3B speech-cascades model once (matching ``eval_openasr.py``:
  30-min process-group timeout, greedy decode, ``TORCHDYNAMO_DISABLE=1`` set by the launcher), reads
  each ``BFCL_v3_<category>.json`` row's 16 kHz mono audio, builds the tool-aware chat prompt with an
  ``<audio>`` placeholder (audio tokens = ``-1`` x ``embed_size``), greedily decodes, and parses the
  raw ``<tool_call>`` output back into the canonical BFCL call string ``[fn(a=1, b='x'), ...]``. One
  JSON record per row is written, byte-compatible with the reference ``responses.json``.
* **report** (``--report``): scores an existing responses file against the dataset's
  ``possible_answer/BFCL_v3_<category>.json`` ground truth and prints **function-call micro accuracy**
  (pooled correct / pooled examples across categories) plus a per-category breakdown.

The scorer is a self-contained, stdlib-only subset of the official Berkeley Function-Calling
Leaderboard AST checker (name match + per-parameter value match against the possible-answer set, with
BFCL's string standardization, and order-independent multi-call matching). The full upstream ``bfcl``
harness is an *optional* authoritative alternative; see ``score_micro_accuracy`` docstring. Import and
``--help`` never touch the GPU or torch: all heavy imports are lazy inside the generation runtime.

Launch generation under ``torch.distributed.run`` (TP=PP=1, world size == DP), then report on rank 0:

    python examples/models/speech_cascades/eval_bfcl.py \
        --hf-path /path/stage1_hf --checkpoint /path/checkpoints/iter_0010000 \
        --data-dir /path/bfcl/speech_data --tokenizer /path/Qwen3.5-4B \
        --categories simple multiple parallel parallel_multiple irrelevance --output /out/responses.json
    python examples/models/speech_cascades/eval_bfcl.py --report \
        --responses /out/responses.json --data-dir /path/bfcl/speech_data --output-dir /out/report
"""

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CATEGORIES = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)

# BFCL schema-type aliases -> JSON-schema types (mirrors the reference eval_bfcl.py TYPE_MAPPING).
TYPE_MAPPING = {
    "any": "string",
    "arraylist": "array",
    "bool": "boolean",
    "dict": "object",
    "float": "number",
    "hashtable": "object",
    "integer": "integer",
    "list": "array",
    "long": "integer",
    "number": "number",
    "object": "object",
    "string": "string",
    "tuple": "array",
}

SYSTEM_PROMPT = (
    "You are an expert at selecting and calling tools. Listen to the user's spoken request and call "
    "every relevant tool with all required arguments. If no provided tool is relevant, do not call a "
    "tool."
)


# --------------------------------------------------------------------------------------------------
# Tool schema formatting (ported verbatim from the reference eval_bfcl.py; pure stdlib).
# --------------------------------------------------------------------------------------------------
def normalize_schema(value: Any) -> Any:
    """Recursively lower-case and canonicalize BFCL ``type`` fields to JSON-schema types."""
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: normalize_schema(item) for key, item in value.items()}
    schema_type = normalized.get("type")
    if isinstance(schema_type, str):
        normalized["type"] = TYPE_MAPPING.get(schema_type.lower(), schema_type.lower())
    return normalized


def make_tool_name(name: str, used_names: set[str]) -> str:
    """Sanitize a BFCL function name to a chat-template-safe, unique tool name."""
    candidate = re.sub(r"\W", "__", name)
    if not candidate or candidate[0].isdigit():
        candidate = f"tool__{candidate}"
    unique_name = candidate
    suffix = 2
    while unique_name in used_names:
        unique_name = f"{candidate}__{suffix}"
        suffix += 1
    used_names.add(unique_name)
    return unique_name


def make_tools(functions: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Build OpenAI-style tool specs from BFCL ``function`` entries.

    Returns the tool list and a mapping from sanitized tool name back to the original BFCL name so
    the decoded call string uses the ground-truth function name.
    """
    tools = []
    original_names: dict[str, str] = {}
    used_names: set[str] = set()
    for function in functions:
        original_name = function["name"]
        tool_name = make_tool_name(original_name, used_names)
        original_names[tool_name] = original_name
        parameters = normalize_schema(function.get("parameters", {}))
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters.setdefault("required", [])
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": function.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return tools, original_names


def format_argument(value: Any) -> str:
    """Render a Python value as BFCL call-string source (``repr`` for scalars, recursive for containers)."""
    if isinstance(value, str):
        return repr(value)
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    if isinstance(value, list):
        return "[" + ", ".join(format_argument(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key!r}: {format_argument(item)}" for key, item in value.items()) + "}"
    return repr(value)


def parameter_schema(tools: list[dict], function_name: str, parameter_name: str) -> dict | None:
    """Look up the JSON-schema for one parameter of one tool (for value-type coercion)."""
    for tool in tools:
        function = tool.get("function", tool)
        if function.get("name") == function_name:
            return (function.get("parameters", {}).get("properties") or {}).get(parameter_name)
    return None


def parse_parameter(raw_value: str, schema: dict | None) -> Any:
    """Coerce a raw ``<parameter>`` string to a Python value using its schema type."""
    schema_types: set[str] = set()
    if schema:
        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            schema_types.add(schema_type)
        elif isinstance(schema_type, list):
            schema_types.update(schema_type)
        for alternative in schema.get("anyOf", []) + schema.get("oneOf", []):
            if isinstance(alternative.get("type"), str):
                schema_types.add(alternative["type"])
    if not schema_types or "string" in schema_types:
        return raw_value
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def qwen_response_to_bfcl(raw_response: str, tools: list[dict], original_names: dict[str, str]) -> str:
    """Convert a Qwen ``<tool_call>`` raw generation into the canonical BFCL call string.

    Mirrors the reference: extracts ``<function=...>``/``<parameter=...>`` blocks into
    ``[name(arg=value, ...), ...]``; if no tool call was emitted, returns the model's plain text with
    ``<think>`` spans stripped (the "no function call" signal used by the irrelevance category).
    """
    call_pattern = re.compile(
        r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )
    parameter_pattern = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
    calls = []
    for call_match in call_pattern.finditer(raw_response):
        generated_name = call_match.group(1)
        arguments: dict[str, Any] = {}
        for parameter_match in parameter_pattern.finditer(call_match.group(2)):
            parameter_name = parameter_match.group(1)
            arguments[parameter_name] = parse_parameter(
                parameter_match.group(2).strip(),
                parameter_schema(tools, generated_name, parameter_name),
            )
        function_name = original_names.get(generated_name, generated_name)
        argument_text = ", ".join(f"{name}={format_argument(value)}" for name, value in arguments.items())
        calls.append(f"{function_name}({argument_text})")
    if calls:
        return "[" + ", ".join(calls) + "]"
    content = call_pattern.sub("", raw_response)
    return re.sub(r"<(?:think|thinking)>.*?</(?:think|thinking)>", "", content, flags=re.DOTALL).strip()


# --------------------------------------------------------------------------------------------------
# Self-contained micro-accuracy scorer (stdlib-only subset of the BFCL AST checker).
# --------------------------------------------------------------------------------------------------
def _func_name(node: ast.expr) -> str:
    """Render a call target (``Name`` or dotted ``Attribute``) as a plain function name string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_func_name(node.value)}.{node.attr}"
    raise ValueError("unsupported call target")


def decode_calls(text: str) -> list[dict[str, dict[str, Any]]]:
    """Decode a BFCL call string into ``[{name: {param: value}}, ...]``.

    Mirrors BFCL ``default_decode_ast_prompting``: strip code fences, wrap in ``[...]`` if needed, and
    AST-parse. Only keyword arguments are captured (the generation path always emits ``key=value``).
    Returns ``[]`` for any non-function-call output (plain text, parse failure, empty list) so the
    irrelevance category is scored correctly.
    """
    stripped = text.strip().strip("`\n ")
    if not stripped:
        return []
    if not stripped.startswith("["):
        stripped = "[" + stripped
    if not stripped.endswith("]"):
        stripped = stripped + "]"
    try:
        parsed = ast.parse(stripped, mode="eval").body
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (ast.List, ast.Tuple)):
        return []
    calls: list[dict[str, dict[str, Any]]] = []
    for element in parsed.elts:
        if not isinstance(element, ast.Call):
            return []
        try:
            name = _func_name(element.func)
            arguments: dict[str, Any] = {}
            for keyword in element.keywords:
                if keyword.arg is None:
                    continue
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            return []
        calls.append({name: arguments})
    return calls


def _standardize(text: str) -> str:
    """BFCL ``standardize_string``: drop `` ,./-_*^`` punctuation, lower-case, single->double quotes."""
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", text).lower().replace("'", '"')


def _standardize_value(value: Any) -> Any:
    """Recursively standardize strings inside a value while leaving numbers/booleans intact."""
    if isinstance(value, str):
        return _standardize(value)
    if isinstance(value, list):
        return [_standardize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _standardize_value(item) for key, item in value.items()}
    return value


def _value_matches(value: Any, acceptable: list[Any]) -> bool:
    """True if ``value`` equals one of the possible-answer entries (BFCL string standardization).

    Empty-string acceptable entries mark an optional parameter and never match a concrete value.
    Python int->float leniency is applied, matching the reference checker.
    """
    standardized_value = _standardize_value(value)
    for answer in acceptable:
        if answer == "":
            continue
        # Guard Python's ``True == 1`` / ``False == 0``: a bool only matches a bool possible answer.
        if isinstance(value, bool) != isinstance(answer, bool):
            continue
        if standardized_value == _standardize_value(answer):
            return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        for answer in acceptable:
            if isinstance(answer, float) and float(value) == answer:
                return True
    return False


def _match_single(pred_call: dict[str, dict[str, Any]], gold_call: dict[str, dict[str, list]]) -> bool:
    """True if a predicted ``{name: {param: value}}`` satisfies one gold ``{name: {param: [values]}}``.

    Mirrors BFCL ``simple_function_checker``: name must match; no unexpected parameters; every
    predicted value must be in its possible-answer set; every gold parameter must be present unless
    ``""`` (optional) is among its possible answers.
    """
    (pred_name, pred_params), = pred_call.items()
    (gold_name, gold_params), = gold_call.items()
    if pred_name != gold_name:
        return False
    for param, value in pred_params.items():
        if param not in gold_params:
            return False
        if not _value_matches(value, gold_params[param]):
            return False
    for param, acceptable in gold_params.items():
        if param not in pred_params and "" not in acceptable:
            return False
    return True


def _match_call_sets(pred_calls: list[dict], gold_calls: list[dict]) -> bool:
    """Order-independent match: a bijection where every gold call matches a distinct predicted call."""
    if len(pred_calls) != len(gold_calls):
        return False
    remaining = list(pred_calls)
    for gold_call in gold_calls:
        for index, pred_call in enumerate(remaining):
            if _match_single(pred_call, gold_call):
                del remaining[index]
                break
        else:
            return False
    return True


def score_example(prediction: str, ground_truth: list[dict] | None, category: str) -> bool:
    """Score one BFCL example. Irrelevance is correct iff no function call was produced."""
    predicted_calls = decode_calls(prediction)
    if "irrelevance" in category:
        return len(predicted_calls) == 0
    if not ground_truth:
        return False
    return _match_call_sets(predicted_calls, ground_truth)


def score_micro_accuracy(examples: list[tuple[str, list[dict] | None, str]]) -> dict[str, Any]:
    """Compute function-call **micro** accuracy over ``(prediction, ground_truth, category)`` examples.

    Micro accuracy pools every example: ``sum(correct) / sum(total)`` across all categories (as opposed
    to a per-category macro average). The full upstream Berkeley Function-Calling Leaderboard harness
    (``bfcl`` package + ``report_bfcl.py``) is the authoritative scorer for published numbers; this
    self-contained checker reproduces its single-turn AST semantics for the ``simple``/``multiple``/
    ``parallel``/``parallel_multiple``/``irrelevance`` categories with no external dependency.
    """
    per_category: dict[str, dict[str, int]] = {}
    total = 0
    correct = 0
    for prediction, ground_truth, category in examples:
        bucket = per_category.setdefault(category, {"total": 0, "correct": 0})
        is_correct = score_example(prediction, ground_truth, category)
        bucket["total"] += 1
        total += 1
        if is_correct:
            bucket["correct"] += 1
            correct += 1
    for bucket in per_category.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
    return {
        "total": total,
        "correct": correct,
        "micro_accuracy": correct / total if total else 0.0,
        "per_category": per_category,
    }


# --------------------------------------------------------------------------------------------------
# Data loading + report mode.
# --------------------------------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file (one JSON object per non-blank line)."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _category_of(example_id: str) -> str:
    """BFCL example ids are ``<category>_<index>``; the category is everything before the last ``_``."""
    return example_id.rsplit("_", 1)[0]


def load_ground_truth(data_dir: Path, categories: list[str]) -> dict[str, list[dict]]:
    """Map example id -> ground-truth call list from ``possible_answer/BFCL_v3_<category>.json``."""
    ground_truth: dict[str, list[dict]] = {}
    for category in categories:
        answer_path = data_dir / "possible_answer" / f"BFCL_v3_{category}.json"
        if not answer_path.is_file():
            continue
        for row in load_jsonl(answer_path):
            ground_truth[row["id"]] = row["ground_truth"]
    return ground_truth


def report(responses_path: Path, data_dir: Path, output_dir: Path | None) -> dict[str, Any]:
    """Score a responses file against the dataset ground truth and print micro accuracy."""
    records = json.loads(responses_path.read_text(encoding="utf-8"))
    categories = sorted({_category_of(record["audio_ids"][0]) for record in records})
    ground_truth = load_ground_truth(data_dir, categories)
    examples: list[tuple[str, list[dict] | None, str]] = []
    for record in records:
        example_id = record["audio_ids"][0]
        prediction = record["generated_texts"][0]
        examples.append((prediction, ground_truth.get(example_id), _category_of(example_id)))
    summary = score_micro_accuracy(examples)
    print(
        f"[bfcl] examples={summary['total']} correct={summary['correct']} "
        f"micro_accuracy={summary['micro_accuracy']:.4f}",
        flush=True,
    )
    for category in sorted(summary["per_category"]):
        bucket = summary["per_category"][category]
        print(
            f"[bfcl]   {category}: {bucket['correct']}/{bucket['total']} = {bucket['accuracy']:.4f}",
            flush=True,
        )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "bfcl_micro_accuracy.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


# --------------------------------------------------------------------------------------------------
# Generation mode (audio -> model function-call output). Heavy imports are lazy (no GPU on import).
# --------------------------------------------------------------------------------------------------
@dataclass
class GenerationResult:
    """One decoded BFCL response: the raw generation, its BFCL call string, and any metadata."""

    raw_response: str
    generated_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def read_audio(row: dict):
    """Load one BFCL row's 16 kHz mono waveform (``wav_path``/``audio_path``) as float32."""
    import soundfile as sf

    path = row.get("wav_path") or row.get("audio_path")
    if isinstance(path, list):
        path = path[0]
    audio, sample_rate = sf.read(Path(path), dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1:
        raise ValueError(f"{row['id']}: expected 16 kHz mono audio, got {sample_rate} Hz shape {audio.shape}")
    return audio, sample_rate


def _embed_size(num_frames: int, compression: int = 8) -> int:
    """Audio placeholder count = encoder-downsampled frame count (streaming compression)."""
    return (num_frames - 7) // compression


def _eos_ids(tokenizer) -> list[int]:
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
            ids.append(token_id)
    return ids


class SpeechCascadesRuntime:
    """Loads the native speech-cascades model once and greedily decodes BFCL rows to call strings.

    Prompt construction renders the tool-aware chat scaffold (system + ``<audio>`` user turn) with the
    tokenizer chat template, then expands the ``<audio>`` marker into ``embed_size`` audio-placeholder
    tokens (``-1``), matching how the model consumes ``audio_input_features`` in ``eval_openasr.py``.
    """

    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoTokenizer

        from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.audio_processors.logfbank_processor import (  # noqa: E501
            LogFbankProcessor,
        )

        self.torch = torch
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        self.processor = LogFbankProcessor()
        self.model = _load_model(args.hf_path, args.checkpoint)
        self.eos = _eos_ids(self.tokenizer)

    def _build_prompt_ids(self, tools: list[dict], embed_size: int) -> list[int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<audio>"},
        ]
        scaffold = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        prefix, _, suffix = scaffold.partition("<audio>")
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
        suffix_ids = self.tokenizer(suffix, add_special_tokens=False).input_ids
        return prefix_ids + [-1] * embed_size + suffix_ids

    def generate(self, row: dict) -> GenerationResult:
        from megatron.bridge.models.speech_cascades.generate import greedy_generate

        audio, sample_rate = read_audio(row)
        tools, original_names = make_tools(row["function"])
        features = self.processor.extract_features(audio, sample_rate)
        num_frames = features.shape[0]
        embed_size = _embed_size(num_frames)
        if embed_size <= 0:
            return GenerationResult("", "", {"skipped": "audio too short"})
        input_ids = self._build_prompt_ids(tools, embed_size)
        audio_start = input_ids.index(-1)
        prompt_ids = self.torch.tensor([input_ids[:audio_start]], dtype=self.torch.long).cuda()
        feats = self.torch.from_numpy(features).unsqueeze(0).to("cuda", self.torch.bfloat16)
        attn = self.torch.ones(1, num_frames, dtype=self.torch.long, device="cuda")
        sizes = self.torch.tensor([embed_size], dtype=self.torch.long, device="cuda")
        generated = greedy_generate(
            self.model,
            prompt_ids,
            audio_input_features=feats,
            audio_attention_mask=attn,
            audio_embed_sizes=sizes,
            max_new_tokens=self.args.max_new_tokens,
            eos_token_ids=self.eos,
        )
        raw = self.tokenizer.decode(generated[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=False)
        text = qwen_response_to_bfcl(raw, tools, original_names)
        return GenerationResult(raw, text, {})


def _load_model(hf_path: str, checkpoint: str):
    import megatron.bridge.models.speech_cascades.bridge  # noqa: F401  (registers the bridge)
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.speech_cascades import dcp_compat  # noqa: F401  (torch>=2.6 no_dist shim)
    from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model = bridge.to_megatron_model(load_weights=False, wrap_with_ddp=False)
    model = model if isinstance(model, list) else [model]
    _load_model_weights_from_checkpoint(checkpoint, model, strict=False)
    for module in model:
        module.eval()
    return model[0].module if hasattr(model[0], "module") else model[0]


def _generate(args: argparse.Namespace) -> None:
    """Load the model once and write one responses record per BFCL row."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        # Long timeout: the 4.3B checkpoint load + first-forward compile can exceed the 600s default.
        from datetime import timedelta

        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)
    ps.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(0)
    dp_rank = ps.get_data_parallel_rank()
    dp_world = ps.get_data_parallel_world_size()

    rows: list[dict] = []
    for category in args.categories:
        category_rows = load_jsonl(args.data_dir / f"BFCL_v3_{category}.json")
        if args.limit_per_category:
            category_rows = category_rows[: args.limit_per_category]
        rows.extend(category_rows)

    runtime = SpeechCascadesRuntime(args)
    results: list[dict] = []
    for index in range(dp_rank, len(rows), dp_world):
        row = rows[index]
        generation = runtime.generate(row)
        results.append(
            {
                "audio_ids": [row["id"]],
                "generated_texts": [generation.generated_text],
                "raw_response": generation.raw_response,
                "input_mode": "speech",
                **generation.metadata,
            }
        )

    gathered: list[list[dict]] = [None] * dp_world  # type: ignore[list-item]
    dist.all_gather_object(gathered, results, group=ps.get_data_parallel_group())
    if dp_rank == 0:
        merged = [record for shard in gathered for record in shard]
        merged.sort(key=lambda record: record["audio_ids"][0])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[bfcl] wrote {len(merged)} records to {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and validate the required set for report vs generate mode."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true", help="Score an existing responses file (no GPU).")
    parser.add_argument("--responses", type=Path, help="[report] Responses JSON to score.")
    parser.add_argument("--data-dir", type=Path, help="BFCL dir with BFCL_v3_<category>.json + possible_answer/.")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--hf-path", help="[generate] Converted stage-1 HF dir (model config).")
    parser.add_argument("--checkpoint", help="[generate] Megatron dist checkpoint dir (iter_*).")
    parser.add_argument("--tokenizer", help="[generate] Qwen3.5-4B tokenizer path.")
    parser.add_argument("--output", type=Path, help="[generate] Responses JSON output path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="[report] Dir for bfcl_micro_accuracy.json.")
    parser.add_argument("--limit-per-category", type=int, default=0, help="[generate] Cap rows per category (0=all).")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.report:
        if args.responses is None or args.data_dir is None:
            parser.error("--report requires --responses and --data-dir")
    else:
        missing = [name for name in ("hf_path", "checkpoint", "data_dir", "tokenizer", "output") if getattr(args, name) is None]
        if missing:
            parser.error("generation requires " + ", ".join("--" + name.replace("_", "-") for name in missing))
    return args


def main() -> None:
    """Dispatch to report scoring (no GPU) or model generation based on ``--report``."""
    args = parse_args()
    if args.report:
        report(args.responses, args.data_dir, args.output_dir)
    else:
        _generate(args)


if __name__ == "__main__":
    main()
