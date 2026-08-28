# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Language related utilities"""

LANGUAGE_PAIRS = {
    "en": "English", 
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "zh-CN": "Chinese",
    "zh-CN-shaanxi": "Chinese",
    "zh-CN-anhui": "Chinese",
    "zh-CN-shandong": "Chinese",
    "zh-CN-henan": "Chinese",
    "hsn-CN": "Chinese",
    "wuu-CN": "Chinese",
    "zh-CN-tianjin": "Chinese",
    "zh-CN-sichuan": "Chinese",
    "nan-CN": "Chinese",
    "cjy-CN": "Chinese",
    "zh-CN-gansu": "Chinese",
    "zh-CN-liaoning": "Chinese",
    "zh-tw": "Chinese",
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
    "el": "Greek",
    "id": "Indonesian",
    "hi": "Hindi",
    "hi-IN": "Hindi",
    "as-IN": "Assamese",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "or-IN": "Oriya",
    "mr-IN": "Marathi",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "ur-IN": "Urdu",
    "None": "None",
}

# Mapping from locale to script/writing system.
# Only locales with ambiguous scripts need an entry here.
# If a locale is not in this dict, no script hint is needed.
LOCALE_SCRIPT_IN_PROMPT = {    
    "zh-tw": "Traditional Chinese",
}

LOCALE_SCRIPT_IN_TARGET = {
    "zh": "Mandarin",
    "zh-CN": "Mandarin",
    "zh-CN-shaanxi": "Mandarin",
    "zh-CN-anhui": "Mandarin",
    "zh-CN-shandong": "Mandarin",
    "zh-CN-henan": "Mandarin",
    "hsn-CN": "Mandarin",
    "wuu-CN": "Mandarin",
    "zh-CN-tianjin": "Mandarin",
    "zh-CN-sichuan": "Mandarin",
    "nan-CN": "Mandarin",
    "cjy-CN": "Mandarin",
    "zh-CN-gansu": "Mandarin",
    "zh-CN-liaoning": "Mandarin",
    "zh-tw": "Taiwanese",
}
