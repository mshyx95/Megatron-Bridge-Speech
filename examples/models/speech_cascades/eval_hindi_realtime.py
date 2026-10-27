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

"""Evaluate pinned Hindi text and VoiceAgentBench sets via the GA Realtime WebSocket API."""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import numpy as np
import soundfile as sf
import soxr
import websockets
from eval_hindi_text import DATASETS, extract_answer, prompt, score
from eval_voiceagentbench_hindi import SUBSETS, messages


LOGGER = logging.getLogger(__name__)
MAX_OUTPUT_TOKENS = 2048
PROTOCOL = "realtime-text-output-v1"


class RealtimeTransientError(RuntimeError):
    """The endpoint explicitly marked a request as retryable."""


def load_endpoint(config: Path, credential_file: Path) -> tuple[str, str, Path]:
    """Resolve a configured Realtime deployment without embedding credentials in output."""
    endpoint = json.loads(config.read_text(encoding="utf-8"))
    url = endpoint["url"]
    parsed = urlsplit(url)
    if parsed.scheme != "wss" or not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("Expected a wss endpoint with no query or embedded credentials")
    if endpoint["protocol"] != "openai_realtime" or not endpoint["model"]:
        raise ValueError("Expected a named GA OpenAI Realtime deployment")
    if not credential_file.is_file():
        raise FileNotFoundError(credential_file)
    return url + "?" + urlencode({"model": endpoint["model"]}), endpoint["model"], credential_file


def audio_content(root: Path, item: dict) -> dict:
    """Convert the benchmark's WAV (including float WAV) to Realtime's 24 kHz mono PCM16."""
    raw = base64.b64decode(item["input_audio"]["data"], validate=True)
    with sf.SoundFile(io.BytesIO(raw)) as source:
        if source.channels not in (1, 2) or source.frames == 0:
            raise ValueError("Expected a nonempty mono or stereo benchmark WAV")
        samples = source.read(dtype="float32")
        rate = source.samplerate
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if rate != 24000:
        samples = soxr.resample(samples, rate, 24000)
    pcm = np.clip(np.rint(samples * 32767), -32768, 32767).astype("<i2").tobytes()
    return {"type": "input_audio", "audio": base64.b64encode(pcm).decode("ascii")}


def voice_items(root: Path, row: dict) -> tuple[str | None, list[dict]]:
    """Preserve the existing benchmark message order, roles and audio-only input boundary."""
    rendered = messages(root, row)
    instructions = None
    items = []
    for turn in rendered:
        if turn["role"] == "system":
            instructions = turn["content"]
            continue
        content = turn["content"]
        if turn["role"] == "assistant":
            items.append(
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}
            )
        elif turn["role"] == "user":
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": part["text"]}
                        if part["type"] == "text"
                        else audio_content(root, part)
                        for part in content
                    ],
                }
            )
        else:
            raise ValueError(f"Unexpected benchmark role: {turn['role']}")
    if not items or not any(part["type"] == "input_audio" for item in items for part in item["content"]):
        raise ValueError(f"No audio input in VoiceAgentBench row {row['id']}")
    return instructions, items


def score_text(rows: list[dict], predictions: list[dict]) -> dict:
    """Report strict prompt adherence and a separate gold-blind final-line answer score."""
    if len(rows) != len(predictions):
        raise ValueError("Text source and prediction counts differ")
    metrics = score(predictions)
    final = []
    for row, result in zip(rows, predictions):
        if result["gold"] is None:
            continue
        lines = result["response"].strip().splitlines()
        final.append((extract_answer(lines[-1] if lines else "", len(row["choices"])), result["gold"]))
    final_correct = sum(answer == gold for answer, gold in final)
    return {
        **metrics,
        "final_line_correct": final_correct,
        "final_line_accuracy": final_correct / len(final) if final else None,
        "final_line_unparsed": sum(answer is None for answer, _ in final),
    }


async def complete(
    endpoint: str,
    model: str,
    credential_file: Path,
    items: list[dict],
    instructions: str | None,
    *,
    reasoning_effort: str | None = None,
) -> str:
    """Use a fresh conversation for every item and require an actual completed text response."""
    for attempt in range(5):
        try:
            return await _complete_once(endpoint, model, credential_file, items, instructions, reasoning_effort)
        except RealtimeTransientError as error:
            if attempt == 4:
                raise
            LOGGER.warning("Retry Realtime request after %s (attempt %d/5)", error, attempt + 1)
        except websockets.exceptions.InvalidStatus as error:
            if error.response.status_code not in (429, 502, 503, 504) or attempt == 4:
                raise
            LOGGER.warning("Retry Realtime HTTP %d (attempt %d/5)", error.response.status_code, attempt + 1)
        except websockets.exceptions.ConnectionClosedError as error:
            if attempt == 4:
                raise
            LOGGER.warning("Retry closed Realtime connection %d (attempt %d/5)", error.code, attempt + 1)
        except TimeoutError:
            if attempt == 4:
                raise
            LOGGER.warning("Retry Realtime response timeout (attempt %d/5)", attempt + 1)
        await asyncio.sleep(2**attempt)
    raise RuntimeError("Realtime retry budget exhausted")


async def _complete_once(
    endpoint: str,
    model: str,
    credential_file: Path,
    items: list[dict],
    instructions: str | None,
    reasoning_effort: str | None,
) -> str:
    token = credential_file.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("Empty Realtime credential file")
    async with websockets.connect(
        endpoint,
        additional_headers={"Authorization": "Bearer " + token},
        open_timeout=30,
        max_size=32 * 1024 * 1024,
        ping_interval=None,
    ) as socket:
        created = json.loads(await asyncio.wait_for(socket.recv(), 30))
        if created["type"] != "session.created":
            raise ValueError(f"Unexpected Realtime handshake: {created['type']}")
        session = {
            "type": "realtime",
            "output_modalities": ["text"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": None,
                    "transcription": None,
                }
            },
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        if instructions is not None:
            session["instructions"] = instructions
        if reasoning_effort is not None:
            session["reasoning"] = {"effort": reasoning_effort}
        await socket.send(json.dumps({"type": "session.update", "session": session}))
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), 60))
            if event["type"] == "error":
                code = event["error"].get("code")
                if code == "rate_limit_exceeded":
                    raise RealtimeTransientError(code)
                raise RuntimeError(f"Realtime session error: {code}")
            if event["type"] == "session.updated":
                accepted = event["session"]
                if accepted["output_modalities"] != ["text"] or accepted["max_output_tokens"] != MAX_OUTPUT_TOKENS:
                    raise ValueError("Realtime session did not accept requested output contract")
                if reasoning_effort is not None and accepted.get("reasoning", {}).get("effort") != reasoning_effort:
                    raise ValueError("Realtime session did not accept requested reasoning effort")
                break
        for item in items:
            await socket.send(json.dumps({"type": "conversation.item.create", "item": item}))
        await socket.send(json.dumps({"type": "response.create"}))
        timeout = 300 if any(part["type"] == "input_audio" for item in items for part in item["content"]) else 120
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout))
            if event["type"] == "error":
                code = event["error"].get("code")
                if code == "rate_limit_exceeded":
                    raise RealtimeTransientError(code)
                raise RuntimeError(f"Realtime generation error: {code}")
            if event["type"] != "response.done":
                continue
            response = event["response"]
            if response["status"] != "completed":
                details = response["status_details"]
                if response["status"] == "failed" and details["error"].get("code") in {
                    "output_timeout",
                    "first_output_timeout",
                    "rate_limit_exceeded",
                }:
                    raise RealtimeTransientError(details["error"]["code"])
                raise RuntimeError(f"Realtime response not completed: {response['status_details']}")
            outputs = response["output"]
            if any(item["type"] != "message" or item["role"] != "assistant" for item in outputs):
                raise ValueError(f"Expected text messages from {model}, got {[part['type'] for part in outputs]}")
            parts = [part for item in outputs for part in item["content"]]
            if any(part["type"] != "output_text" for part in parts):
                raise ValueError(f"Expected output_text parts from {model}")
            return "\n".join(part["text"] for part in parts)


def run(
    root: Path,
    output: Path,
    mirror: Path | None,
    config: Path,
    credential_file: Path,
    dataset: str,
    *,
    official_repo: Path | None,
    limit: int,
    resume: bool,
    workers: int,
    text_effort: str = "default",
) -> dict:
    """Generate ordered resumable predictions for one text or voice subset."""
    if (
        not 1 <= workers <= 8
        or limit < 0
        or dataset not in (*DATASETS, *SUBSETS)
        or text_effort not in ("default", "low")
    ):
        raise ValueError("Expected a known dataset, 1-8 workers and nonnegative limit")
    endpoint, model, credential_file = load_endpoint(config, credential_file)
    voice = dataset in SUBSETS
    if voice and text_effort != "default":
        raise ValueError("The VoiceAgentBench comparison uses the provider-default reasoning setting")
    protocol = PROTOCOL if text_effort == "default" else PROTOCOL + "-low"
    if voice:
        if official_repo is None:
            raise ValueError("VoiceAgentBench requires a pinned official evaluator checkout")
        sys.path.insert(0, str(official_repo / "voice_agent_bench"))
        from utils.parser import parse_response

        folder, evaluator = SUBSETS[dataset]
        source = root / "voiceagentbench" / "data" / folder / "hindi" / f"{dataset}_hindi.json"
        rows = json.loads(source.read_text(encoding="utf-8"))
    else:
        source = root / dataset / "hindi_test.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if limit:
        rows = rows[:limit]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected unique nonempty source IDs: {source}")
    if output.exists() and not resume:
        raise FileExistsError(f"Use --resume or a new output path: {output}")
    results = (
        json.loads(output.read_text(encoding="utf-8"))
        if voice and output.exists()
        else [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        if output.exists()
        else []
    )
    if len(results) > len(rows):
        raise ValueError("Existing output has more predictions than the source")
    for index, result in enumerate(results):
        row = rows[index]
        digest = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        expected = digest if voice else hashlib.sha256(prompt(row).encode()).hexdigest()
        if (
            result["id"] != row["id"]
            or result["run_model"] != model
            or result["protocol"] != protocol
            or result["source_sha256" if voice else "prompt_sha256"] != expected
            or (not voice and result["gold"] != row["answer_index"])
        ):
            raise ValueError(f"Prediction/source/protocol mismatch at offset {index}: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if mirror is not None:
        mirror.parent.mkdir(parents=True, exist_ok=True)

    def infer(row: dict) -> dict:
        if voice:
            instructions, items = voice_items(root / "voiceagentbench", row)
        else:
            instructions = None
            items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt(row)}]}]
        text = asyncio.run(
            complete(
                endpoint,
                model,
                credential_file,
                items,
                instructions,
                reasoning_effort="low" if text_effort == "low" else None,
            )
        )
        if voice:
            parsed = parse_response(text, evaluator)
            if evaluator in {"single", "multiple", "multiturn"}:
                parsed = None if parsed is None else [part for part in parsed if part is not None]
            return {
                **row,
                "generated_text": text,
                "response": parsed,
                "run_model": model,
                "protocol": protocol,
                "source_sha256": hashlib.sha256(
                    json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
            }
        return {
            "id": row["id"],
            "gold": row["answer_index"],
            "prediction": extract_answer(text, len(row["choices"])),
            "response": text,
            "model": model,
            "run_model": model,
            "protocol": protocol,
            "prompt_sha256": hashlib.sha256(prompt(row).encode()).hexdigest(),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        if voice:
            for result in pool.map(infer, rows[len(results) :]):
                results.append(result)
                output.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
                if mirror is not None and (len(results) % 10 == 0 or len(results) == len(rows)):
                    shutil.copyfile(output, mirror)
                LOGGER.info("%s %d/%d", dataset, len(results), len(rows))
        else:
            with output.open("a", encoding="utf-8") as destination:
                for result in pool.map(infer, rows[len(results) :]):
                    destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                    destination.flush()
                    results.append(result)
                    if mirror is not None and (len(results) % 50 == 0 or len(results) == len(rows)):
                        shutil.copyfile(output, mirror)
                    if len(results) % 50 == 0 or len(results) == len(rows):
                        LOGGER.info("%s %d/%d", dataset, len(results), len(rows))
    if mirror is not None and not mirror.exists():
        shutil.copyfile(output, mirror)
    return (
        {"dataset": dataset, "model": model, "completed": len(results), "judge": "not_run"}
        if voice
        else {"dataset": dataset, "model": model, **score_text(rows[: len(results)], results)}
    )


def main() -> None:
    """Run one pinned Hindi split against an existing Realtime deployment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=(*DATASETS, *SUBSETS), required=True)
    parser.add_argument("--endpoint-config", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mirror", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--text-effort", choices=("default", "low"), default="default")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = run(
        args.root,
        args.output,
        args.mirror,
        args.endpoint_config,
        args.credential_file,
        args.dataset,
        official_repo=args.official_repo,
        limit=args.limit,
        resume=args.resume,
        workers=args.workers,
        text_effort=args.text_effort,
    )
    LOGGER.info("RESULT %s", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
