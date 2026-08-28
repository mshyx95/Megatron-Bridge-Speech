# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dataclasses
import random
from enum import auto, Enum
from typing import List, Tuple, Union
import logging
import re
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_AUDIO_TOKEN = '<audio>'
DEFAULT_AUDIO_TOKEN_SET = {'<audio>', '<audio_1>'}
DEFAULT_AUIDO_TOKEN_PATTERN = r'<\|audio_\d+\|>'


def _audio_tok_in_message(message: str) -> Union[str, None]:
    matches = re.findall(rf'({DEFAULT_AUIDO_TOKEN_PATTERN})', message)
    if len(matches):
        return DEFAULT_AUIDO_TOKEN_PATTERN
    for tok in DEFAULT_AUDIO_TOKEN_SET:
        if tok in message:
            return tok
    return None

def _audio_tok_match_message(message: str) -> Union[str, None]:
    matches = re.match(rf'({DEFAULT_AUIDO_TOKEN_PATTERN})', message)
    return matches!=None

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    system_role: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    system_sep: str = None
    version: str = "Unknown"
    skip_next: bool = False
    max_num_audios: int = 64
    audio_token_length: int = 145
    role_suffix: str = ": "
    role_prefix: str = ""
    tokenizer: AutoTokenizer = None

    def get_prompt(self, is_train=False, audio_input_id=0, only_add_first_audio_occurrence=True, audio_start_token='', audio_end_token='', num_audio_tokens=[]):
        """
        only_add_first_audio_occurrence: only add the first audio occurrence, and ignore the rest in multi-turn message for the same audio
        """
        audio_idx = 0 # index to find audio embedding length from num_audio_tokens
        if is_train:
            input_ids = []
            labels = []
            audio_positions = []

        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            raise NotImplementedError

        assert self.sep_style == SeparatorStyle.TWO, "Currently only support LLAVA v1.5"
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            if len(self.system):
                system_sep = self.system_sep if self.system_sep is not None else seps[0]
                ret = self.system + system_sep
                has_system_prompt = True
            else:
                ret = ''
                has_system_prompt = False

            if is_train:
                sys_input_ids = self.tokenizer.encode(ret, add_special_tokens=False)
                input_ids.extend(sys_input_ids)
                labels.extend([IGNORE_INDEX] * len(sys_input_ids))
                # assert len(input_ids) == len(labels), f'sys input_ids {len(input_ids)} and labels {len(labels)} length not match'

            first_audio_token_processed = False
            for i, (role, message) in enumerate(messages):
                # for Mistral 3, we need to remove the <s> prefix in role when there is system prompt, because the <s> is used as the prefix for system role, and the user role should not have it
                if i > 0 and role == self.roles[0]:
                    # add an extra sep before the user message except for the first user message
                    ret += "\n"
                    input_ids.append(self.tokenizer.encode("\n", add_special_tokens=False)[0])
                    labels.append(IGNORE_INDEX)

                if has_system_prompt:
                    used_role = role.replace(self.role_prefix, '', 1) if role.startswith(self.role_prefix) else role
                else:
                    used_role = role

                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if type(message) is list:
                        # math competition is a list
                        message = random.choice(message)
                    ret += used_role + self.role_suffix + message + seps[i % 2]

                    if is_train:
                        # prefix
                        prefix = used_role + self.role_suffix
                        prefix_input_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                        input_ids.extend(prefix_input_ids)
                        labels.extend([IGNORE_INDEX] * len(prefix_input_ids)) # not learned for both roles
                        # assert len(input_ids) == len(labels), f'prefix input_ids {len(input_ids)} and labels {len(labels)} length not match'
                        # message

                        __audio_token = _audio_tok_in_message(message)
                        if __audio_token and __audio_token in message and first_audio_token_processed and only_add_first_audio_occurrence:
                            # ignore the rest of the audio token
                            message = message.replace(__audio_token, '')

                        if __audio_token and _audio_tok_in_message(message):
                            # split message by audio
                            split_message = re.split(rf'({__audio_token})', message)
                            split_message = [_split_message for _split_message in split_message if _split_message]
                            for _idx, _message in enumerate(split_message):
                                if _message == __audio_token or _audio_tok_match_message(_message):
                                    if len(audio_start_token):
                                        input_ids.append(self.tokenizer.convert_tokens_to_ids(audio_start_token))
                                        labels.append(IGNORE_INDEX)
                                    
                                    audio_token_length = num_audio_tokens[audio_idx]

                                    audio_positions.append(len(input_ids))
                                    input_ids.extend([audio_input_id] * audio_token_length)
                                    labels.extend([IGNORE_INDEX] * audio_token_length)
                                    if len(audio_end_token):
                                        input_ids.append(self.tokenizer.convert_tokens_to_ids(audio_end_token))
                                        labels.append(IGNORE_INDEX)
                                    audio_input_id -= 1
                                    audio_idx += 1
                                    # add <end> after audio if the audio ends in user message or assistant message
                                    if _idx == len(split_message) - 1 and len(seps[i % 2]) > 0:
                                        user_end_id = self.tokenizer.encode(seps[i % 2], add_special_tokens=False)
                                        input_ids.extend(user_end_id)
                                        if role in [self.roles[0], self.system_role]: # from user, not learned
                                            labels.extend([IGNORE_INDEX] * len(user_end_id))
                                        else: # from model, learned
                                            labels.extend(user_end_id)
                                else:
                                    if _idx == len(split_message) - 1:
                                        _message = _message + seps[i % 2]
                                    _message_input_ids = self.tokenizer.encode(_message, add_special_tokens=False)
                                    input_ids.extend(_message_input_ids)
                                    if role in [self.roles[0], self.system_role]: # from user, not learned
                                        labels.extend([IGNORE_INDEX] * len(_message_input_ids))
                                    else: # from model, learned
                                        labels.extend(_message_input_ids)

                            first_audio_token_processed = True
                        else:
                            # append text
                            message_input_ids = self.tokenizer.encode(message + seps[i % 2], add_special_tokens=False)
                            input_ids.extend(message_input_ids)

                            # Mask out the user prompts
                            if role in [self.roles[0], self.system_role]: # from user or system, not learned
                                labels.extend([IGNORE_INDEX] * len(message_input_ids))
                            else: # from model, learned
                                labels.extend(message_input_ids)
                else:
                    if not is_train:
                        ret += role + ":"
                    else:
                        # in training, message should not be empty, need to debug
                        # there are actualy empty message:
                        # [['USER', '<audio>\nWhat vitamin is this vegetable associated with?\nAnswer the question using a single word or phrase.'], ['ASSISTANT', '']]
                        logger.info(self.messages)
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        if is_train:
            assert len(input_ids) == len(labels), f'all input_ids {len(input_ids)} and labels {len(labels)} length not match'
            # audio_positions.extend([0] * (self.max_num_audios - len(audio_positions))) # extend to full length
            return ret, input_ids, labels, audio_positions
        else:
            return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def append_system_message(self, role, message):
        assert role == self.system_role
        self.system = role + self.role_suffix + message

    def get_audios(self, return_pil=False):
        audios = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import audio
                    msg, audio, audio_process_mode = msg
                    if audio_process_mode == "Pad":
                        def expand2square(pil_audio, background_color=(122, 116, 104)):
                            width, height = pil_audio.size
                            if width == height:
                                return pil_audio
                            elif width > height:
                                result = audio.new(pil_audio.mode, (width, width), background_color)
                                result.paste(pil_audio, (0, (width - height) // 2))
                                return result
                            else:
                                result = audio.new(pil_audio.mode, (height, height), background_color)
                                result.paste(pil_audio, ((height - width) // 2, 0))
                                return result
                        audio = expand2square(audio)
                    elif audio_process_mode in ["Default", "Crop"]:
                        pass
                    elif audio_process_mode == "Resize":
                        audio = audio.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid audio_process_mode: {audio_process_mode}")
                    max_hw, min_hw = max(audio.size), min(audio.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = audio.size
                    if longest_edge != max(audio.size):
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        audio = audio.resize((W, H))
                    if return_pil:
                        audios.append(audio)
                    else:
                        buffered = BytesIO()
                        audio.save(buffered, format="PNG")
                        audio_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        audios.append(audio_b64_str)
        return audios

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, audio, audio_process_mode = msg
                    max_hw, min_hw = max(audio.size), min(audio.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = audio.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    audio = audio.resize((W, H))
                    buffered = BytesIO()
                    audio.save(buffered, format="JPEG")
                    audio_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    audio_str = f'<audio src="data:audio/png;base64,{audio_b64_str}" alt="user upload audio" />'
                    msg = audio_str + msg.replace('<audio>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def get_roles(self):
        return self.roles

    def get_system_role(self):
        return self.system_role

    def copy(self):
        return Conversation(
            system=self.system,
            system_role=self.system_role,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            system_sep=self.system_sep,
            version=self.version,
            role_suffix=self.role_suffix,
            role_prefix=self.role_prefix,
        )

    def dict(self):
        if len(self.get_audios()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
                "role_suffix": self.role_suffix
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "role_suffix": self.role_suffix,
        }


conv_phi_3 = Conversation(
    system="",
    system_role="<|system|>\n",
    roles=("<|user|>\n", "<|assistant|>\n"),
    role_suffix="",
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="<|end|>\n",
    sep2="<|end|>\n",
)

conv_phi_3_200k_vocab = Conversation(
    system="",
    system_role="system:",
    roles=("user:", "assistant:"),
    role_suffix="",
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="<|endofprompt|>",
    sep2="<|endoftext|>",
)

conv_phi_o_wo_newlines = Conversation(
    system="",
    system_role="<|system|>",
    roles=("<|user|>", "<|assistant|>"),
    role_suffix="",
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="<|end|>",
    sep2="<|end|>",
    system_sep="<|end|>\n",
)

conv_qwen3_instruct = Conversation(
    system="",
    system_role="<|im_start|>system\n",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    role_suffix="",
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="<|im_end|>\n",
    sep2="<|im_end|>",
)

conv_qwen3_nothink_instruct = Conversation(
    system="",
    system_role="<|im_start|>system\n",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    role_suffix="",
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="<|im_end|>\n",
    sep2="<|im_end|>",
)

conv_mistral3_instruct = Conversation(
    system="",
    system_role="<s>[SYSTEM_PROMPT]",
    roles=("<s>[INST]", "[/INST]"),
    role_suffix="",
    role_prefix="<s>",
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="",
    sep2="</s>",
    system_sep="[/SYSTEM_PROMPT]\n",
)

conv_gemma3_instruct = Conversation(
    system="",
    system_role="",
    roles=("<bos><start_of_turn>user\n", "<start_of_turn>model\n"),
    role_suffix="",
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="<end_of_turn>\n",
    sep2="<end_of_turn>",
)


conv_templates = {
    "phi_3": conv_phi_3,
    "phi_o": conv_phi_3,
    "phi_3_200k": conv_phi_3_200k_vocab,
    "phi_o_long": conv_phi_o_wo_newlines,
    "qwen3_audio_instruct": conv_qwen3_instruct,
    "mistral3_audio_instruct": conv_mistral3_instruct,
    "gemma3_audio_instruct": conv_gemma3_instruct,
    "qwen3_audio_nothink_instruct": conv_qwen3_nothink_instruct,
}


if __name__ == "__main__":
    conv_phi_3.append_message("<|user|>\n", "<audio>\nPlease provide a short description for this region: [0.18, 0.5, 0.27, 0.6].")
    conv_phi_3.append_message("<|assistant|>\n", "A large bag on a woman's arm.")
    conv_phi_3.append_message("<|user|>\n", "Please provide a short description for this region: [0.65, 0.59, 0.78, 0.74].")
    conv_phi_3.append_message("<|assistant|>\n", "Black pants on a woman.")
    print(conv_phi_3.get_prompt())
