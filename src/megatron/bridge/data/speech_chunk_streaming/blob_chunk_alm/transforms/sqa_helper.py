"""SQA helper functions"""

from itertools import zip_longest
import re
import numpy as np


def split_lines(text):
    """Split text into lines"""
    if not isinstance(text, str):
        return text, ""
    # Extend regex to support multiple punctuation marks across languages
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    lines = [s.strip() for s in sentences if s.strip()]
    if len(lines) > 0:
        return lines[0], " ".join(lines[1:])
    return "", ""


# The order matters, order of the roles is to convert turns to messages
ROLES = ["system", "user", "assistant"]


def messages2turns(messages):
    """Convert messages to turns"""
    turns = [{}]
    for msg in messages:
        role = msg.pop("role", None)
        if role not in ROLES:
            print(f"Skip invalid role: {role}, content: {msg.get('content', None)} ")
            continue
        turns[-1].update({role: msg})
        if role == "assistant":
            turns.append({})
    turns.pop()  # remove last empty or uncompleted turn
    return turns


def turns2messages(turns):
    """Convert turns to messages"""
    messages = []
    for turn in turns:
        for role in ROLES:
            if role not in turn:
                continue
            messages.append({"role": role, **turn[role]})
    return messages


def add_cot_pair(turn, prompt, response):
    """Add COT pair to the turn"""
    cot_prompts = turn["user"].get("cot_prompts", [])
    cot_prompts.append(prompt)
    turn["user"]["cot_prompts"] = cot_prompts
    cot_responses = turn["assistant"].get("cot_responses", [])
    cot_responses.append(response)
    turn["assistant"]["cot_responses"] = cot_responses
    return turn


def apply_cot(turns):
    """Apply COT"""
    for turn in turns:
        prompt = " and ".join(turn["user"].get("cot_prompts", []))
        if prompt:
            turn["user"]["content"] += f" Please {prompt} first,"
        responses = turn["assistant"].get("cot_responses", []) + [turn["assistant"]["content"]]
        turn["assistant"]["content"] = " ".join(responses)
    return turns


def apply_audio_query(turns, audios, answer_cv_tokens=None, **kwargs):
    """Apply audio query"""
    first_only = kwargs.get("first_only", False)
    prob = kwargs.get("speech_query_prob", 0.5)
    if not audios:
        return turns
    for i, (turn, audio) in enumerate(zip_longest(turns, audios)):
        if audio is None:
            if answer_cv_tokens is not None:
                response = turn["assistant"]
                response["answer_cv_token"] = answer_cv_tokens[i]
            continue
        if np.random.rand() >= prob:
            continue
        if answer_cv_tokens is not None:
            response = turn["assistant"]
            response["answer_cv_token"] = answer_cv_tokens[i]
        msg = turn["user"]
        msg["audio"] = audio
        if first_only:
            query, rest = split_lines(msg["content"])
        else:
            query, rest = msg["content"], ""
        msg["transcript"] = query
        msg["content"] = "<audio>" + rest
    return turns


def apply_think(turns, **kwargs):
    """Apply SQA think"""
    think_prob = kwargs.get("think_prob", 0.0)
    full_think_prob = kwargs.get("full_think_prob", 0.5)
    max_think_len = kwargs.get("max_think_len", 1e6)
    if think_prob <= 0.0:
        return turns

    for turn in turns:
        if "assistant" not in turn:
            continue
        ai_msg = turn["assistant"]
        think_str = ai_msg.get("think", None)
        if think_str is None:
            continue
        n_think = len(think_str.split())
        if n_think > max_think_len:
            print(f"Filtered too long think: {n_think} > {max_think_len}")
            continue
        if np.random.rand() >= think_prob:
            continue
        field = "think_content" if np.random.rand() <= full_think_prob else "content"
        if field not in ai_msg:
            print("Content field not found: {content_field}")
            print(ai_msg)
            continue
        ai_msg["content"] = ai_msg[field]
        add_cot_pair(turn, "think", think_str)
    return turns


def cut_turns(turns, **kwargs):
    """cut turns"""
    max_turns = kwargs.get("max_turns", 100)
    n_turn = np.random.randint(1, max_turns + 1)
    si = np.random.randint(max(0, len(turns) - n_turn) + 1)
    turns = turns[si : si + n_turn]
    return turns


def apply_transcribe(turns, prob=0.0):
    """Apply SQA transcribe"""
    for turn in turns:
        if "assistant" not in turn:
            continue
        if np.random.rand() >= prob:
            continue
        if turn["user"]["content"].startswith("<audio>"):  # audio query
            add_cot_pair(turn, "transcribe", f"<transcribe> {turn['user']['transcript']} </transcribe>")
    return turns


def apply_transcribe_cot(query, answer, transcript, **kwargs):
    """Apply SR COT"""
    prob = kwargs.get("transcribe_prob", 0.0)
    if np.random.rand() >= prob:
        return query, answer
    query = f"{query} Please transcribe first,"
    answer = f"<transcribe> {transcript} </transcribe> {answer}"
    return query, answer


def generate_sqa_sample(sample, **kwargs):
    """Generate SQA sample
    kwargs:
    transcribe_prob, 0.0,
    max_turns, 100,
    speech_query_prob, 0.5,
    first_only, False,
    think_prob, 0.0,
    full_think_prob, 0.5,
    max_think_len, 1e+6,
    """
    assert "sft" in sample, f"sft is not in the {sample.keys()}"
    turns = messages2turns(sample["sft"][0]["messages"])
    if "answer_cv_tokens" in sample:
        turns = apply_audio_query(turns, sample["audios"], answer_cv_tokens=sample["answer_cv_tokens"], **kwargs)
    else:
        turns = apply_audio_query(turns, sample["audios"], **kwargs)
    turns = apply_transcribe(turns, kwargs.get("transcribe_prob", 0.0))
    turns = apply_think(turns, **kwargs)
    turns = apply_cot(turns)
    turns = cut_turns(turns, **kwargs)
    messages = turns2messages(turns)
    sample["audios"] = [msg["audio"] for msg in messages if "audio" in msg and msg["content"].startswith("<audio>")]
    sample["sft"][0]["messages"] = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    if "answer_cv_tokens" in sample:
        sample["answer_cv_tokens"] = [msg["answer_cv_token"] for msg in messages if "answer_cv_token" in msg]
    return sample
