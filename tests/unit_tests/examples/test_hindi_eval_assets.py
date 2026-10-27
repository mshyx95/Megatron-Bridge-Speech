"""CPU-only Hindi evaluation preparation and audio request tests."""

import asyncio
import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


pytestmark = pytest.mark.unit
SCRIPTS = Path(__file__).resolve().parents[3] / "examples" / "models" / "speech_cascades"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hindi_voice_inventory_checks_referenced_audio(tmp_path):
    prepare = load_script("prepare_hindi_eval_assets")
    root = tmp_path / "voice"
    annotation = root / "data" / "multi_turn_data" / "hindi" / "multi_turn_hindi.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps([{"id": "a", "chat_history": [{"role": "user", "path": "data/multi_turn_audios/hindi/a.wav"}]}])
    )
    with pytest.raises(ValueError, match="Missing or invalid"):
        prepare._voice_files(root)
    audio = root / "data" / "multi_turn_audios" / "hindi" / "a.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"wav")
    assert prepare._voice_files(root) == {"multi_turn_hindi": 1}


def test_voice_request_never_includes_text_transcript(tmp_path):
    runner = load_script("eval_voiceagentbench_hindi")
    audio = tmp_path / "data" / "single_tool_audios" / "hindi" / "0.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFFfake")
    row = {
        "instruction": [[{"content": "Tools: {functions}"}]],
        "functions": [{"name": "book"}],
        "path": str(audio.relative_to(tmp_path)),
        "query": "HIDDEN TRANSCRIPT",
    }
    rendered = runner.messages(tmp_path, row)
    assert "HIDDEN TRANSCRIPT" not in str(rendered)
    assert rendered[0]["content"][0]["text"] == "Tools: [{'name': 'book'}]"
    assert rendered[0]["content"][1]["input_audio"]["format"] == "wav"


def test_voice_sequential_tools_receive_explicit_user_info(tmp_path):
    runner = load_script("eval_voiceagentbench_hindi")
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFfake")
    row = {
        "instruction": [[{"content": "Tools: {functions}; user: {user_info}"}]],
        "functions": [{"name": "book"}],
        "user_info": "User ID: user_123",
        "path": audio.name,
    }
    assert runner.messages(tmp_path, row)[0]["content"][0]["text"] == (
        "Tools: [{'name': 'book'}]; user: User ID: user_123"
    )


def test_safety_instructions_are_system_message_with_audio_only_user_turn(tmp_path):
    runner = load_script("eval_voiceagentbench_hindi")
    (tmp_path / "clip.wav").write_bytes(b"RIFFfake")
    row = {
        "instruction": [[{"content": "Refuse unsafe requests. Tools: {functions}"}]],
        "functions": ["book"],
        "user_request": "HIDDEN TRANSCRIPT",
        "path": "clip.wav",
    }
    rendered = runner.messages(tmp_path, row)
    assert rendered[0] == {"role": "system", "content": "Refuse unsafe requests. Tools: ['book']"}
    assert rendered[1]["role"] == "user"
    assert len(rendered[1]["content"]) == 1
    assert rendered[1]["content"][0]["type"] == "input_audio"
    assert "HIDDEN TRANSCRIPT" not in str(rendered)


def test_voice_request_rejects_path_escape(tmp_path):
    runner = load_script("eval_voiceagentbench_hindi")
    outside = tmp_path.parent / f"{tmp_path.name}.wav"
    outside.write_bytes(b"wav")
    try:
        with pytest.raises(ValueError, match="Invalid audio reference"):
            runner._audio(tmp_path, f"../{outside.name}")
    finally:
        outside.unlink()


def test_realtime_voice_preserves_assistant_history_and_pcm_audio(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    runner = load_script("eval_hindi_realtime")
    root = tmp_path / "voice"
    root.mkdir()
    sf.write(root / "clip.wav", np.array([0.0, 0.25, -0.25] * 1000), 16000, subtype="FLOAT")
    row = {
        "id": "sample",
        "instruction": [[{"content": "Tools: {functions}"}]],
        "functions": [{"name": "book"}],
        "path": "clip.wav",
        "chat_history": [
            {"role": "user", "path": "clip.wav"},
            {"role": "assistant", "content": "Could you clarify?"},
            {"role": "user", "path": "clip.wav"},
        ],
    }
    instructions, items = runner.voice_items(root, row)
    assert instructions is None
    assert [item["role"] for item in items] == ["user", "user", "assistant", "user"]
    assert items[2]["content"] == [{"type": "output_text", "text": "Could you clarify?"}]
    audio = base64.b64decode(items[1]["content"][0]["audio"])
    assert len(audio) == 4500 * 2
    assert np.frombuffer(audio, dtype="<i2").max() > 0


def test_realtime_collects_all_assistant_messages_and_rejects_incomplete(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    runner = load_script("eval_hindi_realtime")
    token = tmp_path / "token"
    token.write_text("test-token")
    responses = [
        {"type": "session.created"},
        {"type": "session.updated", "session": {"output_modalities": ["text"], "max_output_tokens": 2048}},
        {
            "type": "response.done",
            "response": {
                "status": "completed",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Refused."}]},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Safe action."}],
                    },
                ],
            },
        },
    ]

    class FakeSocket:
        def __init__(self):
            self.events = iter(responses)
            self.sent = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def recv(self):
            return json.dumps(next(self.events))

        async def send(self, message):
            self.sent.append(json.loads(message))

    sockets = []

    def connect(*args, **kwargs):
        socket = FakeSocket()
        sockets.append(socket)
        return socket

    monkeypatch.setattr(runner.websockets, "connect", connect)
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Test"}]}]
    assert asyncio.run(runner.complete("wss://example.test", "gpt-realtime-2.1", token, items, None)) == (
        "Refused.\nSafe action."
    )
    assert sockets[0].sent[-2]["item"] == items[0]
    responses[-1]["response"] = {
        "status": "incomplete",
        "status_details": {"reason": "max_output_tokens"},
        "output": [],
    }
    with pytest.raises(RuntimeError, match="not completed"):
        asyncio.run(runner.complete("wss://example.test", "gpt-realtime-2.1", token, items, None))


def test_realtime_retries_only_explicit_output_timeout(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    runner = load_script("eval_hindi_realtime")
    token = tmp_path / "token"
    token.write_text("test-token")
    attempts = []

    class FakeSocket:
        async def __aenter__(self):
            attempts.append(1)
            self.events = iter(
                [
                    {"type": "session.created"},
                    {"type": "session.updated", "session": {"output_modalities": ["text"], "max_output_tokens": 2048}},
                    {
                        "type": "response.done",
                        "response": (
                            {
                                "status": "failed",
                                "status_details": {"error": {"type": "server_error", "code": "output_timeout"}},
                            }
                            if len(attempts) == 1
                            else {
                                "status": "completed",
                                "output": [
                                    {
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": "B"}],
                                    }
                                ],
                            }
                        ),
                    },
                ]
            )
            return self

        async def __aexit__(self, *args):
            pass

        async def recv(self):
            return json.dumps(next(self.events))

        async def send(self, message):
            pass

    async def no_wait(delay):
        pass

    monkeypatch.setattr(runner.websockets, "connect", lambda *args, **kwargs: FakeSocket())
    monkeypatch.setattr(runner.asyncio, "sleep", no_wait)
    assert asyncio.run(runner.complete("wss://example.test", "gpt-realtime-2.1", token, [], None)) == "B"
    assert len(attempts) == 2


def test_realtime_retries_socket_timeout(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    runner = load_script("eval_hindi_realtime")
    calls = []

    async def once(*args):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError
        return "A"

    async def no_wait(delay):
        pass

    monkeypatch.setattr(runner, "_complete_once", once)
    monkeypatch.setattr(runner.asyncio, "sleep", no_wait)
    assert asyncio.run(runner.complete("wss://example.test", "gpt-realtime-2.1", tmp_path, [], None)) == "A"
    assert len(calls) == 2


def test_realtime_low_reasoning_must_be_accepted(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    runner = load_script("eval_hindi_realtime")
    token = tmp_path / "token"
    token.write_text("test-token")
    accepted = {"output_modalities": ["text"], "max_output_tokens": 2048}

    class FakeSocket:
        def __init__(self):
            self.events = iter(
                [
                    {"type": "session.created"},
                    {"type": "session.updated", "session": accepted},
                    {
                        "type": "response.done",
                        "response": {
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": "C"}],
                                }
                            ],
                        },
                    },
                ]
            )
            self.sent = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def recv(self):
            return json.dumps(next(self.events))

        async def send(self, message):
            self.sent.append(json.loads(message))

    sockets = []

    def connect(*args, **kwargs):
        socket = FakeSocket()
        sockets.append(socket)
        return socket

    monkeypatch.setattr(runner.websockets, "connect", connect)
    with pytest.raises(ValueError, match="reasoning effort"):
        asyncio.run(runner.complete("wss://example.test", "gpt-realtime-2.1", token, [], None, reasoning_effort="low"))
    assert sockets[0].sent[0]["session"]["reasoning"] == {"effort": "low"}
    accepted["reasoning"] = {"effort": "low"}
    assert asyncio.run(
        runner.complete("wss://example.test", "gpt-realtime-2.1", token, [], None, reasoning_effort="low")
    ) == ("C")


def test_realtime_refuses_to_resume_default_text_as_low(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    runner = load_script("eval_hindi_realtime")
    row = {"id": "sample", "question": "प्रश्न", "choices": ["एक", "दो"], "answer_index": 1}
    dataset = tmp_path / "global_mmlu_lite"
    dataset.mkdir()
    (dataset / "hindi_test.jsonl").write_text(json.dumps(row) + "\n")
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        json.dumps(
            {
                "id": row["id"],
                "gold": 1,
                "run_model": "gpt-realtime-2.1",
                "protocol": runner.PROTOCOL,
                "prompt_sha256": hashlib.sha256(runner.prompt(row).encode()).hexdigest(),
            }
        )
        + "\n"
    )
    config = tmp_path / "endpoint.json"
    config.write_text(
        json.dumps({"url": "wss://example.test/realtime", "model": "gpt-realtime-2.1", "protocol": "openai_realtime"})
    )
    credential = tmp_path / "token"
    credential.write_text("test-token")
    with pytest.raises(ValueError, match="protocol mismatch"):
        runner.run(
            tmp_path,
            output,
            None,
            config,
            credential,
            "global_mmlu_lite",
            official_repo=None,
            limit=0,
            resume=True,
            workers=1,
            text_effort="low",
        )


def test_realtime_text_reports_strict_and_separate_final_line_scores(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    runner = load_script("eval_hindi_realtime")
    source = [
        {"choices": ["एक", "दो", "तीन"]},
        {"choices": ["एक", "दो", "तीन"]},
        {"choices": ["एक", "दो"]},
    ]
    predictions = [
        {"gold": 1, "prediction": None, "response": "मुझे लगता है कि जवाब यह है:\nB"},
        {"gold": 2, "prediction": 2, "response": "C"},
        {"gold": None, "prediction": None, "response": "A"},
    ]
    result = runner.score_text(source, predictions)
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5
    assert result["final_line_correct"] == 2
    assert result["final_line_accuracy"] == 1.0
    assert result["final_line_unparsed"] == 0
    assert result["label_conflicts"] == 1


def test_hindi_safety_judge_uses_upstream_rubric_and_rejects_ambiguous_answers(tmp_path):
    judge = load_script("judge_voiceagentbench_hindi_safety")
    rubric = tmp_path / "utils.py"
    rubric.write_text('meta_prompt_refusal = "Review <response>{generated_text}</response>. Answer: Yes or No."')
    assert judge.refusal_rubric(rubric).format(generated_text="refused") == (
        "Review <response>refused</response>. Answer: Yes or No."
    )
    assert judge.parse_verdict("Reasoning.\nAnswer: Yes") is True
    assert judge.parse_verdict("Reasoning.\nAnswer: No") is False
    with pytest.raises(ValueError, match="unambiguous"):
        judge.parse_verdict("Answer: Yes\nAnswer: No")
    with pytest.raises(ValueError, match="unambiguous"):
        judge.parse_verdict("I cannot decide")


def test_hindi_safety_judge_verifies_source_and_resumes_without_rejudging(tmp_path, monkeypatch):
    judge = load_script("judge_voiceagentbench_hindi_safety")
    source = tmp_path / "source.json"
    predictions = tmp_path / "predictions.json"
    utils = tmp_path / "utils.py"
    output = tmp_path / "judge.jsonl"
    utils.write_text('meta_prompt_refusal = "Judge <response>{generated_text}</response>. Answer: Yes or No."')
    rows = [{"id": f"safety_{index}"} for index in range(75)]
    source.write_text(json.dumps(rows))
    results = [
        {
            **row,
            "run_model": "test-model",
            "generated_text": "I cannot help." if index % 2 else "[tool_call()]",
            "source_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        }
        for index, row in enumerate(rows)
    ]
    predictions.write_text(json.dumps(results))
    calls = []
    monkeypatch.setattr(judge, "azure_token", lambda: "test-token")

    def fake_judge(text, rubric, endpoint, token):
        calls.append(text)
        assert token == "test-token"
        assert "<response>" in rubric
        return (
            text.startswith("I cannot"),
            "Answer: Yes" if text.startswith("I cannot") else "Answer: No",
            "gpt-4.1-test",
        )

    monkeypatch.setattr(judge, "judge_response", fake_judge)
    args = (source, predictions, utils, output, "https://example.cognitiveservices.azure.com", "test-model")
    report = judge.run(*args, resume=False)
    assert (report["completed"], report["refusals"], report["refusal_rate"]) == (75, 37, 37 / 75)
    assert report["official_score"] is False
    assert len(calls) == 75
    assert judge.run(*args, resume=True) == report
    assert len(calls) == 75
    results[0]["generated_text"] = "Changed response"
    predictions.write_text(json.dumps(results))
    with pytest.raises(ValueError, match="verdict/input mismatch"):
        judge.run(*args, resume=True)


def test_voiceagentbench_gpt52_judge_validates_official_reply_shapes(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    judge = load_script("judge_voiceagentbench_hindi_tools")
    judge.validate_judge_reply("The response below (contained in the <response> tags)", "Answer: Yes")
    judge.validate_judge_reply(
        "You are tasked with evaluating the correctness of a sequential call", '{"Score":[1,0]}'
    )
    judge.validate_judge_reply("You are tasked with evaluating the correctness of arguments", '{"Score":1}')
    for prompt, answer in [
        ("The response below (contained in the <response> tags)", "Answer: Yes\nAnswer: No"),
        ("You are tasked with evaluating the correctness of a sequential call", '{"Score":[0,1]}'),
        ("You are tasked with evaluating the correctness of arguments", '{"Score":true}'),
        ("You are tasked with evaluating the correctness of arguments", "invalid"),
    ]:
        with pytest.raises(ValueError):
            judge.validate_judge_reply(prompt, answer)


def test_voiceagentbench_gpt52_parallel_denominator_counts_functions(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    judge = load_script("judge_voiceagentbench_hindi_tools")
    records = [
        {"result": {"scores": [{"score": [1, 1, 1]}, {"score": [1, 0, 0]}]}},
        {"result": {"scores": [{"score": [0, 0, 0]}]}},
    ]
    report = judge.metrics(records, "multiple")
    assert report["completed_items"] == 2
    assert report["scored_functions"] == 3
    assert report["tool_selection"] == {"correct": 2, "accuracy": 2 / 3}
    assert report["tool_call_schema"] == {"correct": 1, "accuracy": 1 / 3}
    assert report["parameter_filling"] == {"correct": 1, "accuracy": 1 / 3}


def test_voice_safety_accepts_explicit_empty_refusal(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    scorer = load_script("score_voiceagentbench_hindi")
    row = {"id": "sample", "user_request": "unsafe"}
    source = tmp_path / "source.json"
    predictions = tmp_path / "predictions.json"
    source.write_text(json.dumps([row]))
    predictions.write_text(
        json.dumps(
            [
                {
                    **row,
                    "generated_text": "",
                    "run_model": "gpt-realtime-2.1",
                    "source_sha256": hashlib.sha256(
                        json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
                    ).hexdigest(),
                }
            ]
        )
    )
    result = scorer.score_subset(source, predictions, "safety", "gpt-realtime-2.1")
    assert result["completed"] == 1
    assert result["empty_responses"] == 1
    assert "unscored" in result["note"]


def test_voice_realtime_empty_completion_counts_as_incorrect(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    scorer = load_script("score_voiceagentbench_hindi")
    row = {"id": "sample", "expected_tool_call": [{"book": {"city": ["Delhi"]}}]}
    source = tmp_path / "source.json"
    predictions = tmp_path / "predictions.json"
    source.write_text(json.dumps([row]))
    predictions.write_text(
        json.dumps(
            [
                {
                    **row,
                    "generated_text": "",
                    "response": None,
                    "run_model": "gpt-realtime-2.1",
                    "protocol": "realtime-text-output-v1",
                    "source_sha256": hashlib.sha256(
                        json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
                    ).hexdigest(),
                }
            ]
        )
    )
    result = scorer.score_subset(source, predictions, "multi_turn", "gpt-realtime-2.1")
    assert result["completed"] == 1
    assert result["empty_responses"] == 1
    assert result["all_function_names_correct"] == result["strict_argument_match"] == 0


def test_mmlu_prox_export_rejects_missing_middle_option(tmp_path):
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    prepare = load_script("prepare_hindi_eval_assets")
    path = tmp_path / "test.parquet"
    row = {"question": "प्रश्न", "answer": "C", "answer_index": 2, "category": "science"}
    row.update({f"option_{index}": f"choice {index}" if index != 1 else None for index in range(10)})
    parquet.write_table(arrow.Table.from_pylist([row]), path)
    with pytest.raises(ValueError, match="Missing options"):
        prepare._export_test("mmlu_prox", [path], tmp_path / "test.jsonl")


def test_global_mmlu_export_produces_scored_choices(tmp_path):
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    prepare = load_script("prepare_hindi_eval_assets")
    path = tmp_path / "test.parquet"
    parquet.write_table(
        arrow.Table.from_pylist(
            [
                {
                    "question": "प्रश्न",
                    "option_a": "एक",
                    "option_b": "दो",
                    "option_c": "तीन",
                    "option_d": "चार",
                    "answer": "B",
                    "subject": "math",
                }
            ]
        ),
        path,
    )
    output = tmp_path / "normalized.jsonl"
    assert prepare._export_test("global_mmlu_lite", [path], output) == (1, 0)
    assert json.loads(output.read_text()) == {
        "id": "global_mmlu_lite/hi/test/0",
        "question": "प्रश्न",
        "choices": ["एक", "दो", "तीन", "चार"],
        "answer_index": 1,
        "subject": "math",
    }


def test_prox_conflicting_answers_are_not_silently_scored(tmp_path):
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    prepare = load_script("prepare_hindi_eval_assets")
    path = tmp_path / "test.parquet"
    row = {"question": "प्रश्न", "answer": "C", "answer_index": 1, "category": "science"}
    row.update({f"option_{index}": f"choice {index}" for index in range(3)})
    row.update({f"option_{index}": None for index in range(3, 10)})
    parquet.write_table(arrow.Table.from_pylist([row]), path)
    output = tmp_path / "normalized.jsonl"
    assert prepare._export_test("mmlu_prox", [path], output) == (1, 1)
    normalized = json.loads(output.read_text())
    assert normalized["answer_index"] is None
    assert normalized["label_conflict"] == {"answer": "C", "answer_index": 1}


def test_hindi_text_letter_scoring_is_strict():
    runner = load_script("eval_hindi_text")
    assert runner.extract_answer("C", 4) == 2
    assert runner.extract_answer(" C. ", 4) == 2
    assert runner.extract_answer("C, लेकिन शायद D", 4) is None
    assert runner.extract_answer("I", 4) is None
    assert runner.score(
        [
            {"gold": 2, "prediction": 2},
            {"gold": 0, "prediction": None},
            {"gold": None, "prediction": 1},
        ]
    ) == {"completed": 3, "scored": 2, "label_conflicts": 1, "correct": 1, "accuracy": 0.5, "unparsed": 1}


def test_hindi_text_resume_checks_model_and_prompt(tmp_path):
    runner = load_script("eval_hindi_text")
    row = {"id": "sample", "question": "प्रश्न", "choices": ["हाँ", "नहीं"], "answer_index": 0}
    output = tmp_path / "results.jsonl"
    output.write_text(
        json.dumps(
            {
                "id": "sample",
                "gold": 0,
                "prediction": 0,
                "model": "old",
                "prompt_sha256": "invalid",
                "response": "A",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="mismatch"):
        runner._load_existing(output, [row], "new", resume=True)
    with pytest.raises(FileExistsError):
        runner._load_existing(output, [row], "old", resume=False)


def test_voice_offline_score_checks_full_source_and_numeric_arguments(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    scorer = load_script("score_voiceagentbench_hindi")
    row = {
        "id": "sample",
        "expected_tool_call": [{"book": {"city": ["Delhi"], "count": ["3"]}}],
    }
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps([row]))
    prediction = {
        **row,
        "generated_text": "[book(city='delhi', count=3)]",
        "response": [{"book": {"city": ["delhi"], "count": [3]}}],
        "run_model": "model",
        "source_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    }
    output = tmp_path / "pred.json"
    output.write_text(json.dumps([prediction]))
    result = scorer.score_subset(gold, output, "single_tool", "model")
    assert result["completed"] == 1
    assert result["all_function_names_correct"] == 1
    assert result["strict_argument_match"] == 1
    output.write_text("[]")
    with pytest.raises(ValueError, match="Incomplete"):
        scorer.score_subset(gold, output, "single_tool", "model")
