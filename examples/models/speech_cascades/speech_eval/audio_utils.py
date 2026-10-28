"""utils for audio tasks"""

import soundfile as sf


def is_empty_audio(audio_path):
    """Check if audio file is empty."""
    with sf.SoundFile(audio_path) as f:
        if f.frames == 0:
            return True
    return False


def is_valid_audio(audio_path, min_seconds=0.0):
    """Check if audio file is valid."""
    frames = 0
    samplerate = 0
    try:
        with sf.SoundFile(audio_path) as f:
            frames = f.frames
            samplerate = f.samplerate
    except Exception:
        pass
    return frames > 0 and (min_seconds <= 0 or (samplerate > 0 and frames / samplerate >= min_seconds))


def apply_chat_template(user_prompt, template, audio_tag="<audio>"):
    """chat template"""
    template_mapping = {
        "phi_o": "<|user|>\n{audio_tag}\n{user_prompt}<|end|>\n<|assistant|>\n",
        "phi_3": "<|user|>\n{audio_tag}\n{user_prompt}<|end|>\n<|assistant|>\n",
        "phi_3_200k": "user:{audio_tag}\n{user_prompt}<|endofprompt|>assistant:",
        "phi_o_long": "<|user|>{audio_tag}{user_prompt}<|end|><|assistant|>",
        "qwen3_audio_instruct": "<|im_start|>user\n{audio_tag}\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n",
        "qwen3_audio_nothink_instruct": "<|im_start|>user\n{audio_tag}\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "qwen3_5_audio_instruct": "<|im_start|>user\n{audio_tag}\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n",
        "qwen3_5_audio_nothink_instruct": "<|im_start|>user\n{audio_tag}\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "qwen3_5_audio_transcribe_answer_instruct": "<|im_start|>user\n{audio_tag}\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think></think> <ASR> The user said:",
        "qwen3_5_mtp_audio_instruct": "<|im_start|>user\n{audio_tag}\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "mistral3_audio_instruct": "<s>[INST]{audio_tag}{user_prompt}[/INST]",
        "gemma3_audio_instruct": "<bos><start_of_turn>user\n{audio_tag}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n",
    }
    return template_mapping[template].format(user_prompt=user_prompt, audio_tag=audio_tag)


language_pairs = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "el": "Greek",
    "id": "Indonesian",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "ru": "Russian",
    "nb": "Norwegian",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "he": "Hebrew",
    "cs": "Czech",
    "hu": "Hungarian",
    "th": "Thai",
    "hi": "Hindi",
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "None": "None",
}

LOCALE_SCRIPT_IN_PROMPT = {
    "zh-tw": "Traditional Chinese",
}

LOCALE_SCRIPT_IN_TARGET = {
    "zh": "Mandarin",
    "zh-cn": "Mandarin",
    "zh-tw": "Taiwanese",
}
