# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import random
import torch
import numpy as np

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.prompt import (
    ASR_PROMPTS,
    LID_PROMPTS,
    LID_ANSWERS,
    QA_PROMPTS,
    ST_PROMPTS,
    ST_COT_PROMPTS,
    ST_LID_COT_PROMPTS,
    ASR_DIARIZATION_PROMPT,
    BIAS_PROMPTS,
    SCRIPT_PROMPTS,
    AUDIO_CONTINUATION_PROMPTS,
)

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.lang_utils import LANGUAGE_PAIRS as language_pairs
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.lang_utils import LOCALE_SCRIPT_IN_TARGET, LOCALE_SCRIPT_IN_PROMPT

# MAX_PROMPT = "Based on the attached audio, generate a comprehensive text transcription of the spoken content."
# This might not be the longest prompt after tokenization.

def process_asr_annotation(asr_label, n_audios=1):
    assert n_audios > 0
    prompt = random.choice(ASR_PROMPTS)
    
    if asr_label is None:
        asr_label = [asr_label]
        assert n_audios == 1
    if isinstance(asr_label, list) and any(l is None for l in asr_label):
        assert all(l is None for l in asr_label)
        interleaved_data = ["Here is an audio"] + [None] * n_audios + [".\n", f"Instruction: {prompt}\nOutput: "]
        return interleaved_data

    if not isinstance(asr_label, list):
        asr_label = [asr_label]

    if n_audios == 1:
        interleaved_data = ["Here is an audio", None, ".\n", f"Instruction: {prompt}\nOutput: ", asr_label[0], "\n<|endoftext|>"]
    else:
        interleaved_data = ["Here are several audios"] + [None] * n_audios + [".\n", f"Instruction: {prompt}\nOutput: "] + asr_label + ["\n<|endoftext|>"]
    return interleaved_data

def combine_audio_and_text_ordered(audios, texts):
    # audio1 text1, audio2 text2,....,
    n_audios = len(audios)
    interleaved_data = []

    for i in range(len(audios)):
        if len(texts[i]) == 1:
            interleaved_data += process_asr_annotation(texts[i][0])
        else:
            # the data label has already been processed into prompt during data preprocessing.
            interleaved_data += texts[i]

    return audios, interleaved_data

def combine_audio_and_text_random(audios, texts):
    # audio1, audio2, text2, text1
    # random groups, random order

    n_audios = len(audios)
    num_groups = random.randint(1, n_audios)
    indexes = np.random.choice(np.arange(1, n_audios), num_groups-1, replace=False)
    indexes.sort()
    indexes = indexes.tolist()
    indexes.append(n_audios)

    interleaved_data = []
    prev_idx = 0
    for idx in indexes:

        group_text = texts[prev_idx:idx]
        random.shuffle(group_text)

        asr_labels = [text[0] for text in group_text]
        interleaved_data += process_asr_annotation(asr_labels, len(asr_labels))
        
        prev_idx = idx
 
    return audios, interleaved_data

def is_audio_token(text):
    if text is None or text.startswith('<|speech_') or text.startswith('<|audio_'):
        return True
    return False

def get_audio_id(text):
    # parse audio id from the token, token might be '<|speech_{}|>', '<|audio_{}|>'
    audio_ids = re.findall(r'<\|(?:speech|audio)_(\d+)\|>', text)
    assert len(audio_ids) == 1, f'Invalid Audio token: {text}'
    return audio_ids[0]

def pad_features(inputs, pad_value=0, return_pt=True):
    """pad list of 2d array for audio features to batch"""
    nframe = [i.shape[0] for i in inputs]
    max_t = max(nframe)

    pad_value = [pad_value for i in inputs]

    padded_inputs = [
        np.pad(
            input,
            ((0, max_t - nframe[i]), (0, 0)),
            "constant",
            constant_values=pad_value[i],
        )
        for i, input in enumerate(inputs)
    ]

    tensors = np.stack(padded_inputs)
    
    if return_pt:
        tensors = torch.from_numpy(tensors)

    return tensors, nframe
    

def convert_to_prompt(roles, text, prompt_formatter, fixed_prompt=True, user_prompt=None, language=None, context=None, audio_transcripts=None):
    """
    user_prompt: 3 cases:
    1. In stage-1 training, if use pre-defined hard-coded prompt:
        - user_prompt = None
        - sampled from pre-defined ASR-prompts
    2. For stage-2 SFT, audio context + text prompt
        - user_prompt = str, for QA tasks, the user_prompt is the question
        - user_prompt = None, for other tasks as there is no user_prompt, the prompt would be generate from the pre-defined list
    3. For stage-2 SFT, dialogue data
        user_prompt = None
        audio only query and  audio context + audio prompt
        maybe need a system prompt 
        single-turn + multi-turn (implemented by interleave)
    """
    user_role_str, assistant_role_str = roles[0], roles[1]

    if prompt_formatter.startswith('asr'):
        assert user_prompt is None, f"Check the data config as {prompt_formatter} task prompt is generated from pre-defined list"
        prompt = ASR_PROMPTS[0] if fixed_prompt else random.choice(ASR_PROMPTS)
        if context is not None and context.strip() != "":
            bias_prompt = BIAS_PROMPTS[0] if fixed_prompt else random.choice(BIAS_PROMPTS)
            prompt += f" {bias_prompt} {context}."
    elif prompt_formatter in ['audio-response', 'audio-response-with-transcription']:
        assert user_prompt is None, f"{prompt_formatter} data requires no additional text prompt"
        prompt = None
    elif prompt_formatter in ['audio-continuation', 'audio-continuation-with-transcription']:
        assert user_prompt is None, f"{prompt_formatter} data requires no additional text prompt"
        prompt = AUDIO_CONTINUATION_PROMPTS[0] if fixed_prompt else random.choice(AUDIO_CONTINUATION_PROMPTS)
    elif prompt_formatter == 'lid':
        prompt = LID_PROMPTS[0] if fixed_prompt else random.choice(LID_PROMPTS)
        text = random.choice(LID_ANSWERS).format(lang=language_pairs[language])
    elif prompt_formatter in ['sqa', 'sqa-lid']:
        assert user_prompt is not None, "please provide the question for sqa task"
        prompt = user_prompt
    elif prompt_formatter in ['ast', 'ast-lid', "ast-cot", "ast-lid-cot"]:
        assert language is not None, f"Please provide language information for the translation tasks!"
        tgt_language = language[1] if isinstance(language, list) else language
        if prompt_formatter in ['ast', 'ast-lid']:
            PROMPT_LIST_TO_USE = ST_PROMPTS
        elif prompt_formatter == 'ast-cot':
            PROMPT_LIST_TO_USE = ST_COT_PROMPTS
        elif prompt_formatter == 'ast-lid-cot':
            PROMPT_LIST_TO_USE = ST_LID_COT_PROMPTS
        prompt = PROMPT_LIST_TO_USE[0].format(lang=language_pairs[tgt_language]) if fixed_prompt \
            else random.choice(PROMPT_LIST_TO_USE).format(lang=language_pairs[tgt_language])
    elif prompt_formatter == 'dialogue':
        assert user_prompt is None, f"dialogue data requires no text prompt"
        raise NotImplementedError
    elif prompt_formatter == 'diarization':
        assert user_prompt is None, f"diarization data requires no text prompt"
        prompt = ASR_DIARIZATION_PROMPT[0] if fixed_prompt else random.choice(ASR_DIARIZATION_PROMPT)
    else:
        raise NotImplementedError
    
    if prompt_formatter in {
        'audio-response-with-transcription',
        'audio-continuation-with-transcription',
    }:
        if (
            audio_transcripts is None
            or len(audio_transcripts) != 1
            or not isinstance(audio_transcripts[0], str)
            or not audio_transcripts[0].strip()
        ):
            raise ValueError(
                f"{prompt_formatter} requires exactly one non-empty audio transcript"
            )
        transcription = audio_transcripts[0].strip()
        text = f"<think></think> <ASR> The user said: {transcription} </ASR> {text}"

    if prompt_formatter in ['audio-continuation', 'audio-continuation-with-transcription']:
        messages = [[user_role_str, f"{prompt}\n\n<audio>"], [assistant_role_str, f"{text}"]]
    elif prompt is not None:
        messages = [[user_role_str, f"<audio>\n{prompt}"], [assistant_role_str, f"{text}"]]
    else:
        messages = [[user_role_str, f"<audio>\n"], [assistant_role_str, f"{text}"]]
    return messages

TASK_FORMAT_MAPPING = {
    "asr-lid": "<ASR><lang={lang}><TXT>{content}</TXT></ASR>",
    "asr-lid-codeswitch": "<ASR><lang={lang1}><TXT>{content1}</TXT>\n<lang={lang2}><TXT>{content2}</TXT></ASR>",
    "asr-lid-pl": "<ASR_BIASING><lang={lang}><TXT>{content}</TXT></ASR_BIASING>",
    "asr-lid-verbatim-pl": "<ASR_BIASING><lang={lang}><TXT>{content}</TXT></ASR_BIASING>",
    "asr-lid-lexical-pl": "<ASR_BIASING><lang={lang}><TXT>{content}</TXT></ASR_BIASING>",
    "asr-lid-lexical": "<ASR_LEXICAL><lang={lang}><TXT>{content}</TXT></ASR_LEXICAL>",
    "asr-lid-verbatim": "<ASR_VERBATIM><lang={lang}><TXT>{content}</TXT></ASR_VERBATIM>",
    "ast-lid": "<AST><lang={lang}><TXT>{content}</TXT></AST>",
    "ast-lid-cot": "<AST-COT><Transcription><lang={src_lang}><TXT>{content1}</TXT></Transcription><Translation><lang={tgt_lang}><TXT>{content2}</TXT></Translation></AST-COT>",
    "asr-lid-ghost": "<ASR><lang={lang}><TXT>{content}</TXT></ASR>",
    "sqa-lid": "<SQA><lang={lang}><TXT>{content}</TXT></SQA>",
    "diarization-lid": "<DIARIZATION><lang={lang}><TXT>{content}</TXT></DIARIZATION>"
}

def add_audio_text_in_conversation(
    texts, user_prompts, languages, contexts, conv_template, prompt_formatter: str = None, fixed_prompt: bool = True, instruction_first: bool = False, bias_after_audio: bool = False,
):
    # audio1 text1, audio2 text2,....,

    for i in range(len(texts)):
        roles = conv_template.get_roles()
        system_role = conv_template.get_system_role()

        if len(texts[i]) == 1 and (prompt_formatter is not None):
            messages = convert_to_prompt(roles, texts[i][0], prompt_formatter, fixed_prompt, user_prompts[i], languages[i], contexts[i])
        else:
            # the data label has already been processed into formats of conversation
            raw_messages = texts[i]
            role_mapping = {"user": roles[0], "assistant": roles[1], "system": system_role}
            messages = []
            for m in raw_messages:
                messages.append([role_mapping[m['role']],  m['content']])

        if prompt_formatter in ['asr-lid-only', 'asr-lid-mix']:
            print(f"{prompt_formatter} is deprecated, enforce asr-lid for language ID in the prompt")
            prompt_formatter = "asr-lid"
            
        if prompt_formatter in ["asr-lid-lexical", "asr-lexical", "asr-lid-lexical-pl"]:
            user_role_str, assistant_role_str = roles[0], roles[1]
            for m in messages:
                if m[0] == user_role_str:
                    prompt = m[1].replace("<audio>", "").replace("\n", "")
                    prompt = f"{prompt} Output must be in lexical format."
                    m[1] = f"<audio>\n{prompt}"
                
                if prompt_formatter == "asr-lexical" and m[0] == assistant_role_str:
                    m[1] = f"<LEXICAL>{m[1]}"

        if prompt_formatter in ["asr-lid-verbatim", "asr-verbatim", "asr-lid-verbatim-pl"]:
            user_role_str, assistant_role_str = roles[0], roles[1]
            for m in messages:
                if m[0] == user_role_str:
                    prompt = m[1].replace("<audio>", "").replace("\n", "")
                    prompt = f"{prompt} Transcribe verbatim, including all filler words and disfluencies."
                    m[1] = f"<audio>\n{prompt}"

                if prompt_formatter == "asr-verbatim" and m[0] == assistant_role_str:
                    m[1] = f"<VERBATIM>{m[1]}"

        # This is for stage-2 training, we enforce the language output for each task
        if "lid" in prompt_formatter:
            user_role_str, assistant_role_str = roles[0], roles[1]
            
            for m in messages:
                if m[0] == user_role_str:
                    prompt = m[1].replace("<audio>", "").replace("\n", "")
                    # lowercase the first character of the prompt to make it more natural after adding the language information in front of the prompt.
                    new_prompt = re.sub(r'^\s*([^\s]+)', lambda m: m.group(1).lower(), prompt, count=1)
                    # manully replace the prompt for use <sep> format>
                    # "Transcribe the audio to text, and then translate the audio to English. Use <sep> as a separator between the original transcript and the translation."
                    if "cot" in prompt_formatter:
                        new_prompt = new_prompt.replace("Use <sep> as a separator between the original transcript and the translation.","").strip()

                    # Add script hint for locales with ambiguous writing systems (e.g. zh-CN vs zh-TW)
                    script_hint = ""
                    if  prompt_formatter.startswith("asr"):
                        if isinstance(languages[i], dict):
                            for key in ["audio_0", "audio_1"]:
                                if languages[i].get(key) in LOCALE_SCRIPT_IN_PROMPT:
                                    script = LOCALE_SCRIPT_IN_PROMPT[languages[i][key]]
                                    script_hint_text = SCRIPT_PROMPTS[0] if fixed_prompt else random.choice(SCRIPT_PROMPTS)
                                    script_hint = f" {script_hint_text.format(script=script)}"
                                    break
                        else:
                            locale_key = languages[i][0] if isinstance(languages[i], list) else languages[i]
                            if locale_key in LOCALE_SCRIPT_IN_PROMPT:
                                script = LOCALE_SCRIPT_IN_PROMPT[locale_key]
                                script_hint_text = SCRIPT_PROMPTS[0] if fixed_prompt else random.choice(SCRIPT_PROMPTS)
                                script_hint = f" {script_hint_text.format(script=script)}"

                    m[1] = f"<audio>\nDetect the language and {new_prompt}{script_hint}"
            
                if m[0] == assistant_role_str:
                    if prompt_formatter == "asr-lid-codeswitch" and isinstance(languages[i], dict):
                        lang1 = language_pairs.get(languages[i]["audio_0"], 'None')
                        lang2 = language_pairs.get(languages[i]["audio_1"], 'None')
                        content_parts = m[1].split("\n\n")
                        assert len(content_parts) == 2, f"Code-switching example should have two parts of text separated by \\n\\n, but got {m[1]}"
                        assistant_text = f"Audio Language: {lang1} and {lang2}\n"
                        assistant_text += TASK_FORMAT_MAPPING["asr-lid-codeswitch"].format(lang1=lang1, content1=content_parts[0], lang2=lang2, content2=content_parts[1])
                    elif prompt_formatter == "ast-lid-cot":
                        source_lang = language_pairs.get(languages[i][0], 'None')
                        target_lang = language_pairs.get(languages[i][1], 'None')
                        content_parts = [part.strip() for part in m[1].split("<sep>")]
                        assert len(content_parts) == 2, f"ST with CoT example should have two parts of text separated by <sep>, but got {m[1]}"
                        assistant_text = f"Audio Language: {source_lang}\n"
                        assistant_text += TASK_FORMAT_MAPPING["ast-lid-cot"].format(src_lang=source_lang, content1=content_parts[0], tgt_lang=target_lang, content2=content_parts[1])
                    elif prompt_formatter == "asr-lid-ghost":
                        source_lang = "None"
                        target_lang = source_lang
                        assistant_text = f"Audio Language: {source_lang}\n"
                        assistant_text += TASK_FORMAT_MAPPING.get(prompt_formatter, "{content}").format(lang=target_lang, content=m[1])
                    else:
                        if isinstance(languages[i], list):
                            source_lang = language_pairs.get(languages[i][0], 'None')
                            target_lang = language_pairs.get(languages[i][1], 'None')
                        else:
                            source_lang = language_pairs.get(languages[i], 'None')
                            if languages[i] in LOCALE_SCRIPT_IN_TARGET:
                                target_lang = LOCALE_SCRIPT_IN_TARGET[languages[i]]
                            else:
                                target_lang = language_pairs.get(languages[i], 'None')

                        assistant_text = f"Audio Language: {source_lang}\n"
                        #in code-switching case, if only one language is detected, so no second language output, we will use the same language for both parts of the output and let the model generate the correct answer based on the content, instead of providing the language information for each part of the output.
                        if prompt_formatter == "asr-lid-codeswitch":
                            assistant_text += TASK_FORMAT_MAPPING.get("asr-lid", "{content}").format(lang=target_lang, content=m[1])
                        else:
                            assistant_text += TASK_FORMAT_MAPPING.get(prompt_formatter, "{content}").format(lang=target_lang, content=m[1])
                    m[1] = assistant_text

        for m in messages:
            # for phi-o release, the \n is removed from the end of the audio tag
            # <|end>| as sep indicating the chat tempalte does not use newlines
            if conv_template.sep2 in ["<|end|>", "</s>"]:
                m[1] = m[1].replace("<audio>\n", "<audio>")

            if instruction_first and m[0] == roles[0] and re.search("<audio>", m[1]) is not None:
                instruction = m[1].replace("<audio>", "").strip()
                if instruction != "":
                    if bias_after_audio:
                        # Keep biasing prompt after <audio> for better biasing performance
                        # but keep verbatim/lexical format prompts before <audio>
                        FORMAT_SUFFIXES = [
                            " Transcribe verbatim, including all filler words and disfluencies.",
                            " Output must be in lexical format.",
                        ]
                        bias_split_idx = None
                        for bp in BIAS_PROMPTS:
                            if bp in instruction:
                                bias_split_idx = instruction.index(bp)
                                break
                        if bias_split_idx is not None:
                            main_instruction = instruction[:bias_split_idx].strip()
                            bias_part = instruction[bias_split_idx:]
                            # Extract verbatim/lexical format suffixes from bias_part and keep before <audio>
                            for suffix in FORMAT_SUFFIXES:
                                if suffix in bias_part:
                                    bias_part = bias_part.replace(suffix, "")
                                    main_instruction = f"{main_instruction}{suffix}"
                            m[1] = f"{main_instruction}<audio>\n{bias_part}" if main_instruction else f"<audio>\n{bias_part}"
                        else:
                            m[1] = f"{instruction}<audio>"
                    else:
                        m[1] = f"{instruction}<audio>"
            
            if m[0] == system_role:
                conv_template.append_system_message(m[0], m[1])
            else:
                conv_template.append_message(m[0], m[1])

    return conv_template
