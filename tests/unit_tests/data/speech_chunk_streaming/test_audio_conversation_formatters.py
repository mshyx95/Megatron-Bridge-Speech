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

import pytest
import torch

from megatron.bridge.data.speech_chunk.teacher_text import (
    build_teacher_text_example,
    derive_transcript_ids,
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


