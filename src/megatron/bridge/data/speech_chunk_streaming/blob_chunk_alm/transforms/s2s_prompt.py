# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Speech-to-speech prompt extensions.

This module intentionally reuses the existing text/audio prompt lists and only
defines the speech-output specific prompts added by the s2s pipeline.
"""

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.prompt import (
    ASR_DIARIZATION_PROMPT,
    ASR_PROMPTS,
    BIAS_PROMPTS,
    DIALOG_PROMPT,
    LID_ANSWERS,
    LID_PROMPTS,
    QA_PROMPTS,
    ST_COT_PROMPTS,
    ST_LID_COT_PROMPTS,
    ST_PROMPTS,
)

ASR_PROMPT = ASR_PROMPTS
QA_PROMPT = QA_PROMPTS

CODEC_PROMPT = [
    "",
    "",
    "",
    "Conduct a spoken conversation with the user.",
    "Carry out a voice-based conversation with the user.",
    "Engage in a spoken dialogue with the user.",
    "Interact with the user through spoken conversation.",
    "Have a verbal exchange with the user.",
    "Participate in a voice conversation with the user.",
    "Hold a spoken dialogue with the user.",
    "Communicate with the user using spoken language.",
    "Engage in a voice conversation with the user.",
    "Speak directly with the user.",
    "Have a spoken conversation with the user.",
    "Chat with the user using natural speech.",
    "Talk in real-time with the user.",
    "Start speaking with the user.",
    "Launch a talk session with the user.",
    "Start a spoken dialogue session with the user.",
    "Speak with the user using a natural voice interface.",
]

TTS_CODEC_PROMPT = [
    "Generate TTS based on text.",
]

TEXTIN_CODEC_PROMPT = [
    "Conduct a spoken conversation with the text input from user.",
]

SEMANTIC_CODEC_ASR_PROMPT = [
    "Recognize transcription based on semantic token.",
]

NTP_CODEC_PROMPT = [
    "Generate Text and Semantic token as next token predict pretraining.",
]
