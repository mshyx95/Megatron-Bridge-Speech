# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re

from .audio_utils import LOCALE_SCRIPT_IN_PROMPT, apply_chat_template, language_pairs


metric_choices = {"hfwer", "bleu"}


def clean_asr_mode_tags(text):
    return re.sub(r"</?(?:asr_)?(?:lexical|verbatim)>", "", text, flags=re.IGNORECASE)


def _move_audio_after_instruction(instruction: str, bias_after_audio: bool = False) -> str:
    if re.search("<audio>", instruction) is None:
        return instruction
    prompt = instruction.replace("<audio>", "").strip()
    if not prompt:
        return instruction
    if bias_after_audio:
        # Keep biasing prompt after <audio> for better biasing performance
        bias_marker = " Pay extra attention to the following phrases/words:"
        if bias_marker in prompt:
            main_prompt, bias_part = prompt.split(bias_marker, 1)
            return f"{main_prompt.strip()}<audio>\n{bias_marker.strip()}{bias_part}"
    return f"{prompt}<audio>"


def apply_instruction_first_to_source(source: str, template: str, bias_after_audio: bool = False) -> str:
    if template == "phi_o_long":
        systems = re.findall(r"<\|system\|>(.*?)<\|user\|>", source, re.DOTALL)
        instructions = re.findall(r"<\|user\|>(.*?)<\|end\|>", source, re.DOTALL)
        answers = re.findall(r"<\|assistant\|>(.*?)<\|end\|>", source, re.DOTALL)
        assert len(instructions) == (len(answers) + 1)

        new_source = ""
        if len(systems) > 0:
            new_source += f"<|system|>{systems[0]}"

        for i, instruction in enumerate(instructions):
            new_source += f"<|user|>{_move_audio_after_instruction(instruction, bias_after_audio)}<|end|>"
            if i < len(answers):
                new_source += f"<|assistant|>{answers[i]}<|end|>"

        return new_source + "<|assistant|>"

    if template in [
        "qwen3_audio_instruct",
        "qwen3_audio_nothink_instruct",
        "qwen3_5_audio_instruct",
        "qwen3_5_audio_nothink_instruct",
        "qwen3_5_audio_transcribe_answer_instruct",
        "qwen3_5_mtp_audio_instruct",
    ]:
        return re.sub(
            r"(<\|im_start\|>user\n)(.*?)(<\|im_end\|>)",
            lambda match: f"{match.group(1)}{_move_audio_after_instruction(match.group(2), bias_after_audio)}{match.group(3)}",
            source,
            flags=re.DOTALL,
        )

    raise NotImplementedError(f"instruction first is not supported for template {template}")


def asr_prompt_map(
    inputs, sub_task="wer_en", lid_mode="0", template="phi_3", instruction_first="false", bias_after_audio="false"
):
    if sub_task not in metric_choices and len(sub_task.split("_")) == 2:
        sub_task, language = sub_task.split("_")
        language = language.lower()
        if "locale" in inputs:
            if inputs["locale"] not in language_pairs:
                language = inputs["locale"].split("-")[0].lower()
            else:
                language = inputs["locale"].lower()

            locale = inputs["locale"].lower()
        else:
            locale = language
        assert language in language_pairs, f"language {language} not found in language pairs mapping"

    elif len(sub_task.split("_")) == 3:
        sub_task, src_lang, tgt_lang = sub_task.split("_")
        src_lang = src_lang.lower()
        tgt_lang = tgt_lang.lower()
        language = src_lang
        locale = None
        assert sub_task in ["bleu", "bleucot"], (
            "currently we only support bleu and bleu cot for multilingual evaluation"
        )

    if sub_task in ["wer", "hfwer", "bleu", "bleucot"] and lid_mode != "0":
        if template in ["phi_o", "phi_3"]:
            prompt = inputs["source"].split("<|end|>")[0].split("<audio>\n")[1]
            new_prompt = f"<|user|>\n<audio>\nDetect the language and {prompt[0].lower() + prompt[1:]} Language: {language_pairs[language]}. Transcription: <|end|>\n<|assistant|>\n"
        elif template == "phi_3_200k":
            prompt = inputs["source"].split("<|endofprompt|>")[0].split("<audio>\n")[1]
            new_prompt = f"user:<audio>\nDetect the language and {prompt[0].lower() + prompt[1:]} Language: {language_pairs[language]}. Transcription: <|endofprompt|>assistant:"
        elif template == "phi_o_long":
            prompt = inputs["source"].split("<|end|>")[0].split("<audio>")[1]
            prompt = f"Detect the language and {prompt[0].lower() + prompt[1:]}"
            new_prompt = None
        elif template in [
            "qwen3_audio_instruct",
            "qwen3_audio_nothink_instruct",
            "qwen3_5_audio_instruct",
            "qwen3_5_audio_nothink_instruct",
            "qwen3_5_audio_transcribe_answer_instruct",
            "qwen3_5_mtp_audio_instruct",
        ]:
            prompt = inputs["source"].split("<audio>\n")[1].split("<|im_end|>")[0]
            prompt = f"Detect the language and {prompt[0].lower() + prompt[1:]}"
            new_prompt = None
        elif template == "mistral3_audio_instruct":
            prompt = inputs["source"].split("<audio>")[1].split("[/INST]")[0]
            prompt = f"Detect the language and {prompt[0].lower() + prompt[1:]}"
            new_prompt = None
        elif template == "gemma3_audio_instruct":
            prompt = inputs["source"].split("<audio>\n")[1].split("<end_of_turn>")[0]
            prompt = f"Detect the language and {prompt[0].lower() + prompt[1:]}"
            new_prompt = None

        if new_prompt is None:
            # tmp solution for capitalization issue in the prompt
            # to make sure the phrase list is not conerted to lowercase, we only capitalize the first letter of the prompt and keep the rest unchanged
            prompt = ". ".join([s[0].upper() + s[1:] if s else s for s in prompt.split(". ")])
            if locale in LOCALE_SCRIPT_IN_PROMPT:
                script = LOCALE_SCRIPT_IN_PROMPT[locale]
                prompt = f"{prompt} Use {script} for the transcription."
            new_prompt = apply_chat_template(prompt, template)

        # lid mode=1 is to predict the language id by model itself while lid_mode=2 is to provide lid as condition
        if lid_mode == "2":
            # task_tag = "<ASR>" if sub_task == "wer" else "<AST>"
            language_id = language_pairs[language]
            new_prompt = f"{new_prompt}Audio Language: {language_id}\n"  # {task_tag}<lang={language_id}>"

        inputs["source"] = new_prompt

    if sub_task == "lid":
        raise ValueError("LID-only evaluation is outside the seven-suite contract")

    if str(instruction_first).lower() == "true":
        inputs["source"] = apply_instruction_first_to_source(
            inputs["source"], template, bias_after_audio=str(bias_after_audio).lower() == "true"
        )

    return inputs
