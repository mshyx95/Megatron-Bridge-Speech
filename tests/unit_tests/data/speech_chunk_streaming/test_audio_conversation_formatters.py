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

"""Unit tests for the S2 KD conversation formatters (ASR / nothink / continuation)."""

import ast
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest
import torch

from megatron.bridge.data.speech_chunk.teacher_text import (
    build_teacher_text_example,
    derive_transcript_ids,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.audio_text_conversation import (
    AudioTextConversationTransform,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.prompt import (
    AUDIO_CONTINUATION_PROMPTS,
)
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.utils import convert_to_prompt


ROLES = ("<|user|>", "<|assistant|>")
TRANSCRIPT = "the quick brown fox"


def _assistant(messages):
    return next(text for role, text in messages if role == ROLES[1])


def _user(messages):
    return next(text for role, text in messages if role == ROLES[0])


def test_asr_formatter_still_transcribes():
    messages = convert_to_prompt(ROLES, TRANSCRIPT, "asr", fixed_prompt=True)
    assert "<audio>" in _user(messages)
    assert _assistant(messages) == TRANSCRIPT


def test_audio_response_bare_user_turn_response_is_transcript():
    # nothink: user turn is a bare audio placeholder, response is the transcript.
    messages = convert_to_prompt(ROLES, TRANSCRIPT, "audio-response", fixed_prompt=True)
    assert _user(messages) == "<audio>\n"
    assert _assistant(messages) == TRANSCRIPT


def test_audio_continuation_prompt_before_audio_response_is_transcript():
    # continuation: user turn is `{prompt}\n\n<audio>`, response is the transcript.
    messages = convert_to_prompt(ROLES, TRANSCRIPT, "audio-continuation", fixed_prompt=True)
    assert _user(messages) == f"{AUDIO_CONTINUATION_PROMPTS[0]}\n\n<audio>"
    assert _assistant(messages) == TRANSCRIPT


def test_audio_response_with_transcription_embeds_transcript():
    messages = convert_to_prompt(
        ROLES, "sure, here you go", "audio-response-with-transcription", audio_transcripts=[TRANSCRIPT]
    )
    assert _assistant(messages) == f"<think></think> <ASR> The user said: {TRANSCRIPT} </ASR> sure, here you go"


def test_with_transcription_requires_transcript():
    with pytest.raises(ValueError):
        convert_to_prompt(ROLES, "resp", "audio-continuation-with-transcription", audio_transcripts=None)


def test_unknown_formatter_still_raises():
    with pytest.raises(NotImplementedError):
        convert_to_prompt(ROLES, "resp", "totally-unknown-formatter")


def test_continuation_style_teacher_alignment():
    # continuation layout: user-prompt tokens precede the audio span; response == transcript.
    # The teacher path derives the transcript from the supervised response and substitutes it for
    # the audio placeholder run, so KD works for nothink/continuation exactly as for ASR.
    IGN = -100
    input_ids = torch.tensor([91, 92, -1, -1, -1, 500, 501])  # [prompt, prompt, <audio x3>, ans, ans]
    labels = torch.tensor([IGN, IGN, IGN, IGN, IGN, 500, 501])

    transcripts = derive_transcript_ids(input_ids, labels)
    assert len(transcripts) == 1
    assert torch.equal(transcripts[0], torch.tensor([500, 501]))

    teacher_ids, teacher_labels = build_teacher_text_example(input_ids, labels, transcripts)
    # audio placeholders replaced by the transcript tokens; the response segment stays intact.
    assert torch.equal(teacher_ids, torch.tensor([91, 92, 500, 501, 500, 501]))
    assert torch.equal(teacher_labels[-2:], torch.tensor([500, 501]))


def test_ast_cot_teacher_uses_transcript_before_sep():
    # ast-cot response = {transcript}<sep>{translation}; the teacher transcript is the pre-<sep> part.
    IGN = -100
    SEP = [98, 99]  # synthetic tokenization of "<sep>"
    # <audio x2>, transcript[10, 11], <sep>[98, 99], translation[12, 13]
    input_ids = torch.tensor([-1, -1, 10, 11, 98, 99, 12, 13])
    labels = torch.tensor([IGN, IGN, 10, 11, 98, 99, 12, 13])

    # Without the separator the whole response (incl. translation) would be taken.
    full = derive_transcript_ids(input_ids, labels)
    assert torch.equal(full[0], torch.tensor([10, 11, 98, 99, 12, 13]))

    # With the separator only the source transcript is kept.
    transcripts = derive_transcript_ids(input_ids, labels, separator_ids_variants=[SEP])
    assert torch.equal(transcripts[0], torch.tensor([10, 11]))


class _CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]

    def decode(self, ids):
        return "".join(chr(int(token) - 1) for token in ids if token > 0)


def _kd_raw_example():
    return {
        "audios": [torch.zeros(80, 80), torch.zeros(96, 80)],
        "texts": [["Paris."], ["Four."]],
        "audio_embed_sizes": [[3], [4]],
        "audio_frames": [80, 96],
        "audio_transcripts": [["What is the capital of France?"], ["What is two plus two?"]],
        "user_prompts": [None, None],
        "languages": ["en", "en"],
        "contexts": [None, None],
        "tokenizer": _CharacterTokenizer(),
    }


@pytest.mark.unit
@pytest.mark.parametrize("batch_format", ["samples_to_sequence", "samples_to_batch"])
@pytest.mark.parametrize("formatter", ["audio-response", "audio-continuation"])
def test_rewrite_teacher_uses_source_transcript_and_same_answer(batch_format, formatter):
    transform = AudioTextConversationTransform(
        {"pad_to_context_length": False}, conv_template="qwen3_audio_instruct",
        prompt_formatter=formatter, batch_format=batch_format, knowledge_distillation=True,
    )
    raw = _kd_raw_example()
    result = transform.transform(deepcopy(raw))
    teacher = raw["tokenizer"].decode(result["teacher_input_ids"].reshape(-1))
    assert "What is the capital of France?" in teacher
    assert "What is two plus two?" in teacher
    assert "Paris." in teacher and "Four." in teacher
    assert "<audio>" not in teacher
    assert (result["teacher_input_ids"] >= 0).all()
    student_labels = result["labels"].input_ids
    teacher_labels = result["teacher_labels"]
    assert torch.equal(student_labels[student_labels != -100], teacher_labels[teacher_labels != -100])
    transform.knowledge_distillation = False
    baseline = transform.transform(deepcopy(raw))
    assert "teacher_input_ids" not in baseline
    assert torch.equal(baseline["inputs"].input_ids, result["inputs"].input_ids)
    assert torch.equal(baseline["labels"].input_ids, student_labels)


@pytest.mark.unit
@pytest.mark.parametrize("transcripts", [None, [[""], ["valid"]], [["extra", "other"], ["valid"]]])
def test_rewrite_kd_rejects_missing_or_invalid_transcript(transcripts):
    raw = _kd_raw_example()
    raw["audio_transcripts"] = transcripts
    transform = AudioTextConversationTransform(
        {"pad_to_context_length": False}, conv_template="qwen3_audio_instruct",
        prompt_formatter="audio-response", knowledge_distillation=True,
    )
    with pytest.raises(ValueError, match="one non-empty transcript per audio"):
        transform.transform(raw)


@pytest.mark.unit
@pytest.mark.parametrize("formatter,text,transcript", [
    ("asr", "bonjour", "bonjour"),
    ("asr-lexical", "bonjour", "bonjour"),
    ("ast-cot", "bonjour<sep>hello", "bonjour"),
])
def test_only_transcription_tasks_can_infer_source(formatter, text, transcript):
    transform = AudioTextConversationTransform(
        {}, conv_template="qwen3_audio_instruct", prompt_formatter=formatter, knowledge_distillation=True,
    )
    assert transform._require_complete_teacher_transcripts([[text]], None, [[3]]) == [[transcript]]


@pytest.mark.unit
def test_teacher_placeholder_count_must_match():
    transform = AudioTextConversationTransform(
        {}, conv_template="qwen3_audio_instruct", prompt_formatter="audio-response", knowledge_distillation=True,
    )
    conversation = transform.conv_template.copy()
    conversation.tokenizer = _CharacterTokenizer()
    conversation.messages = [[conversation.roles[0], "<audio> then <|audio_1|>"], [conversation.roles[1], "answer"]]
    with pytest.raises(ValueError, match="Missing transcript"):
        transform._build_teacher_inputs(conversation, ["first"])
    with pytest.raises(ValueError, match="More audio transcripts"):
        transform._build_teacher_inputs(conversation, ["first", "second", "extra"])


@pytest.fixture
def reference_transform(monkeypatch):
    root = os.environ.get("SPEECH_CASCADES_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set SPEECH_CASCADES_REFERENCE_ROOT for source-checkout parity")
    directory = Path(root) / "phyagi/datasets/blob_chunk_alm/transforms"
    package_name = "_speech_kd_reference"
    package = ModuleType(package_name)
    package.__path__ = [str(directory)]
    monkeypatch.setitem(sys.modules, package_name, package)

    def load(name):
        path = directory / f"{name}.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                node.module = node.module.replace("phyagi.datasets.blob_chunk_alm.transforms", package_name)
        module = ModuleType(f"{package_name}.{name}")
        module.__package__ = package_name
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(tree, str(path), "exec"), module.__dict__)
        return module

    load("utils")
    return load("audio_text_conversation").AudioTextConversationTransform


@pytest.fixture
def parity_tokenizer():
    path = os.environ.get("SPEECH_CASCADES_PARITY_TOKENIZER")
    if not path:
        return _CharacterTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@pytest.mark.unit
@pytest.mark.parametrize("batch_format", ["samples_to_sequence", "samples_to_batch"])
@pytest.mark.parametrize("formatter", [
    "asr", "asr-lexical", "ast-cot", "audio-response", "audio-continuation",
    "audio-response-with-transcription", "audio-continuation-with-transcription",
])
def test_teacher_tensors_match_reference(reference_transform, parity_tokenizer, batch_format, formatter):
    raw = _kd_raw_example()
    raw["tokenizer"] = parity_tokenizer
    if formatter.startswith("asr"):
        raw["audio_transcripts"] = None
    elif formatter == "ast-cot":
        raw["texts"] = [["bonjour<sep>hello"], ["merci<sep>thank you"]]
        raw["languages"] = [["fr", "en"], ["fr", "en"]]
        raw["audio_transcripts"] = None
    kwargs = dict(
        conv_template="qwen3_audio_instruct", prompt_formatter=formatter,
        fixed_prompt=False, batch_format=batch_format, knowledge_distillation=True,
    )
    actual_transform = AudioTextConversationTransform({"pad_to_context_length": False}, **kwargs)
    expected_transform = reference_transform({"pad_to_context_length": False}, **kwargs)
    random.seed(1234)
    actual = actual_transform.transform(deepcopy(raw))
    random.seed(1234)
    expected = expected_transform.transform(deepcopy(raw))
    assert torch.equal(actual["inputs"].input_ids, expected["inputs"].input_ids)
    assert torch.equal(actual["labels"].input_ids, expected["labels"].input_ids)
    for key in ("teacher_input_ids", "teacher_labels", "teacher_attention_mask", "audio_embed_sizes"):
        assert torch.equal(actual[key], expected[key]), key


@pytest.mark.unit
@pytest.mark.parametrize("batch_format", ["samples_to_sequence", "samples_to_batch"])
def test_multi_audio_teacher_matches_reference(reference_transform, parity_tokenizer, batch_format):
    raw = _kd_raw_example()
    raw["tokenizer"] = parity_tokenizer
    raw["texts"] = [[
        {"role": "user", "content": "First: <|audio_0|> and second: <|audio_1|>"},
        {"role": "assistant", "content": "Both answers."},
    ]]
    raw["audio_transcripts"] = [["First source question.", "Second source question."]]
    raw["audio_embed_sizes"] = [[3, 4]]
    raw["user_prompts"] = [None]
    raw["languages"] = ["en"]
    raw["contexts"] = [None]
    kwargs = dict(
        conv_template="qwen3_audio_instruct", prompt_formatter="audio-response",
        batch_format=batch_format, knowledge_distillation=True,
    )
    actual = AudioTextConversationTransform({"pad_to_context_length": False}, **kwargs).transform(deepcopy(raw))
    expected = reference_transform({"pad_to_context_length": False}, **kwargs).transform(deepcopy(raw))
    for key in ("teacher_input_ids", "teacher_labels", "teacher_attention_mask"):
        assert torch.equal(actual[key], expected[key]), key


