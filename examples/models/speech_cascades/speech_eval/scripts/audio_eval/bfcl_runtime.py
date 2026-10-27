#!/usr/bin/env python3

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf

from .cascade_transcripts import load_transcripts


DEFAULT_CATEGORIES = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)

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

SYSTEM_PROMPTS = {
    "lfg3": (
        "You are an expert at selecting and calling tools. Use the provided tools "
        "to complete the user's spoken request. Call every tool needed and provide "
        "all required arguments. If no provided tool is relevant, do not call a tool."
    ),
    "qwen35-audio": (
        "You are an expert at selecting and calling tools. Listen to the user's "
        "spoken request and call every relevant tool with all required arguments. "
        "If no provided tool is relevant, do not call a tool."
    ),
    "qwen35-text": (
        "You are an expert at selecting and calling tools. Read the user's request "
        "and call every relevant tool with all required arguments. If no provided "
        "tool is relevant, do not call a tool."
    ),
}


@dataclass
class GenerationResult:
    raw_response: str
    generated_text: str
    latency_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate supported speech or text models on BFCL V3.")
    parser.add_argument(
        "--model-family",
        choices=tuple(SYSTEM_PROMPTS),
        required=True,
        help="Runtime used to load the model and encode BFCL requests.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Tokenizer path; required for qwen35-audio and ignored otherwise.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--asr-transcripts",
        type=Path,
        help=(
            "JSON or JSONL ASR results keyed by id/UUID/audio_id. For qwen35-text, "
            "replace the single spoken user message with its hypothesis."
        ),
    )
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn", default="flash_attention_2")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = {
            "lfg3": 1,
            "qwen35-audio": 8,
            "qwen35-text": 16,
        }[args.model_family]
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.model_family == "qwen35-audio" and args.tokenizer is None:
        parser.error("--tokenizer is required for --model-family qwen35-audio")
    if args.asr_transcripts is not None and args.model_family != "qwen35-text":
        parser.error("--asr-transcripts is only supported for qwen35-text")
    return args


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_rows(args: argparse.Namespace) -> list[dict]:
    rows = []
    for category in args.categories:
        category_rows = load_jsonl(args.data_dir / f"BFCL_v3_{category}.json")
        if args.limit_per_category:
            category_rows = category_rows[: args.limit_per_category]
        rows.extend(category_rows)
    if args.asr_transcripts is not None:
        transcripts = load_transcripts(args.asr_transcripts)
        missing_ids = [row["id"] for row in rows if row["id"] not in transcripts]
        if missing_ids:
            raise ValueError(
                f"Missing ASR transcripts for {len(missing_ids)} BFCL rows; first missing id: {missing_ids[0]}"
            )
        for row in rows:
            row["asr_transcript"] = transcripts[row["id"]]
    return rows


def save_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def normalize_schema(value: Any) -> Any:
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
    tools = []
    original_names = {}
    used_names = set()
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


def parsed_response_to_bfcl(
    parsed_response: dict,
    original_names: dict[str, str],
) -> str:
    calls = []
    for tool_call in parsed_response.get("tool_calls") or []:
        function = tool_call.get("function", tool_call)
        name = original_names.get(function["name"], function["name"])
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise TypeError(f"Tool arguments for {name} are not an object")
        argument_text = ", ".join(f"{key}={format_argument(value)}" for key, value in arguments.items())
        calls.append(f"{name}({argument_text})")
    if calls:
        return "[" + ", ".join(calls) + "]"
    return (parsed_response.get("content") or "").strip()


def parameter_schema(tools: list[dict], function_name: str, parameter_name: str):
    for tool in tools:
        function = tool.get("function", tool)
        if function.get("name") == function_name:
            return (function.get("parameters", {}).get("properties") or {}).get(parameter_name)
    return None


def parse_parameter(raw_value: str, schema: dict | None) -> Any:
    schema_types = set()
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


def qwen_response_to_bfcl(
    raw_response: str,
    tools: list[dict],
    original_names: dict[str, str],
) -> str:
    call_pattern = re.compile(
        r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )
    parameter_pattern = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
    calls = []
    for call_match in call_pattern.finditer(raw_response):
        generated_name = call_match.group(1)
        arguments = {}
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
    return re.sub(
        r"<(?:think|thinking)>.*?</(?:think|thinking)>",
        "",
        content,
        flags=re.DOTALL,
    ).strip()


def read_audio(row: dict):
    audio, sample_rate = sf.read(Path(row["wav_path"][0]), dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1:
        raise ValueError(f"{row['id']}: expected 16 kHz mono audio, got {sample_rate} Hz with shape {audio.shape}")
    return audio, sample_rate


def system_prompt(row: dict, model_family: str) -> str:
    parts = [SYSTEM_PROMPTS[model_family]]
    for message in row["question"][0]:
        if message["role"] in ("system", "developer"):
            parts.append(message["content"])
    return "\n\n".join(parts)


def text_messages(row: dict) -> list[dict]:
    messages = [dict(message) for message in row["question"][0] if message["role"] not in ("system", "developer")]
    transcript = row.get("asr_transcript")
    if transcript is None:
        return messages
    user_messages = [message for message in messages if message["role"] == "user"]
    if len(user_messages) != 1:
        raise ValueError(f"{row['id']}: cascade BFCL expects exactly one user message, found {len(user_messages)}")
    user_messages[0]["content"] = transcript
    return messages


def make_qwen_prompt(tokenizer, row: dict, tools: list[dict], args) -> str:
    if args.model_family == "qwen35-audio":
        messages = [
            {"role": "system", "content": system_prompt(row, args.model_family)},
            {"role": "user", "content": "<audio>"},
        ]
    else:
        messages = text_messages(row)
        messages.insert(
            0,
            {"role": "system", "content": system_prompt(row, args.model_family)},
        )
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )


def load_runtime(args):
    raise RuntimeError("Use the repository eval_vllm.py entrypoint; local model loading is disabled")


def main() -> None:
    args = parse_args()
    rows = load_rows(args)
    results = []
    if args.output.is_file() and not args.no_resume:
        results = json.loads(args.output.read_text(encoding="utf-8"))
    completed_ids = {result["audio_ids"][0] for result in results}
    pending_rows = [row for row in rows if row["id"] not in completed_ids]
    print(
        f"Loaded {len(rows)} examples; {len(completed_ids)} complete and {len(pending_rows)} pending.",
        flush=True,
    )
    if not pending_rows:
        return

    runtime = load_runtime(args)
    for batch_start in range(0, len(pending_rows), args.batch_size):
        batch_rows = pending_rows[batch_start : batch_start + args.batch_size]
        generated = runtime.generate(batch_rows)
        if len(generated) != len(batch_rows):
            raise RuntimeError(f"Runtime returned {len(generated)} results for {len(batch_rows)} rows")
        for row, generation in zip(batch_rows, generated):
            results.append(
                {
                    "audio_ids": [row["id"]],
                    "label": [row.get("label")],
                    "generated_texts": [generation.generated_text],
                    "raw_response": generation.raw_response,
                    "model_family": args.model_family,
                    "input_mode": (
                        "cascade"
                        if args.asr_transcripts is not None
                        else "text"
                        if args.model_family == "qwen35-text"
                        else "speech"
                    ),
                    "enable_thinking": args.enable_thinking,
                    "latency_seconds": round(generation.latency_seconds, 4),
                    "batch_size": len(batch_rows),
                    **generation.metadata,
                }
            )
        save_results(args.output, results)
        completed = min(batch_start + len(batch_rows), len(pending_rows))
        print(
            f"[{completed}/{len(pending_rows)}] batch={len(batch_rows)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
