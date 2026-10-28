# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
import math
import json
import string

def remove_utterance_by_length(
    input_,
    max_feat_len,
    max_label_len=5000000,
    min_feat_len=0,
    min_label_len=0,
    feature_key="feature",
    label_key="label",
):
    """Remove ultra-short and long utterance according to both feature sequeence
    length and label sequence length. It assumes the first dimension of
    input_['feature'] and input_['transcription'] are sentence length.
    """

    return (
        (max_feat_len <= 0 or len(input_[feature_key]) <= max_feat_len)
        and (
            label_key not in input_ or max_label_len <= 0 or len(input_[label_key]) <= max_label_len
        )
        and (min_feat_len <= 0 or len(input_[feature_key]) >= min_feat_len)
        and (
            label_key not in input_ or min_label_len <= 0 or len(input_[label_key]) >= min_label_len
        )
    )


def remove_utterance_by_wer(input_, **kwargs):
    """Remove utterances if wer is out of range. Usually, the wer is the normalized wer
    between transcription and recognition result.
    """
    args = {
        "max_wer": 1.0,
        "min_wer": 0.0,
        "wer_info_keys": ["normalized_WER_speechinsight", "normalized_WER"],
    }
    for k in args.copy():  # override defaults if specified
        if k in kwargs:
            args[k] = kwargs[k]

    for wer_info_key in args["wer_info_keys"]:  # "normalized_WER_speechinsight", "normalized_WER"
        if wer_info_key in input_["info"]:
            wer = input_["info"][wer_info_key]
            return args["min_wer"] <= wer <= args["max_wer"]

    # for ZH GPT data: WER metrics live under alternative_transcription_meta["display_GPT4o"]
    alt_meta = input_["info"].get("alternative_transcription_meta")
    if isinstance(alt_meta, dict) and "display_GPT4o" in alt_meta:
        gpt_meta = alt_meta["display_GPT4o"]
        if isinstance(gpt_meta, dict) and "wer_metrics" in gpt_meta:
            wer_metrics = gpt_meta["wer_metrics"]
            # wer_metrics may be a list of dicts or a single dict depending on the data source.
            if isinstance(wer_metrics, list):
                wer_metrics = wer_metrics[0] if wer_metrics else {}

            retain = True
            # Check if wer exists
            if "wer" in wer_metrics:
                retain &= (wer_metrics["wer"] <= args["max_wer"])

            if "new_chars" in wer_metrics:
                retain &= (len(wer_metrics["new_chars"]) <= 0)

            # Drop the utterance if too many chars were removed relative to the lexical length.
            if "removed_chars" in wer_metrics:
                alt_trans = input_["info"].get("alternative_transcription", {})
                lexical_text = alt_trans.get("lexical", "") if isinstance(alt_trans, dict) else ""
                if isinstance(lexical_text, list):
                    lexical_text = lexical_text[0] if lexical_text else ""
                if len(lexical_text) > 0:
                    retain &= (len(wer_metrics["removed_chars"]) <= len(lexical_text) / 10)
            return retain

    return True  # Use this utterance by default if there is no WER information

def remove_punctuation(text: str) -> str:
    """Remove punctuation from text (including Chinese punctuation)"""
    if not text:
        return ""
    punctuation = string.punctuation + "。，！？；：""''（）【】《》、·…—"
    return "".join(char for char in text if char not in punctuation)

def check_zh_en_wer(input_, transcription_keys):
    """Check if the utterance is Chinese with English content, and return skip_wer flag and result.
    """
    if "info" in input_ and "alternative_transcription" in input_["info"]:
        info_chunk = input_["info"]
        for transcription_key in transcription_keys:
            if transcription_key in info_chunk["alternative_transcription"]:
                trans_text = info_chunk["alternative_transcription"][transcription_key]
                # Handle list type: take first element
                if isinstance(trans_text, list) and len(trans_text) > 0:
                    trans_text = trans_text[0]
                # Check if contains English letters (a-z, A-Z)
                if trans_text and isinstance(trans_text, str) and re.search(r'[a-zA-Z]', trans_text):
                    # Check length difference between trans (without punctuation) and lexical
                    lexical_text = info_chunk["alternative_transcription"].get("lexical", "")
                    if lexical_text:
                        trans_no_punc = remove_punctuation(trans_text)
                        lexical_len = len(lexical_text)
                        trans_len = len(trans_no_punc)
                        if lexical_len > 0:
                            length_diff_ratio = abs(trans_len - lexical_len) / lexical_len
                            if length_diff_ratio > 0.25:                                    
                                return True, False                        
                    return True, True
                break  # Only check the first available transcription
    return False, True

def remove_utterance_by_ctc_length_filter(
    input_,
    feature_key="feature",
    label_key="label",
    subsampling_factor=4,
):
    """Remove utterances in which feature sequence length is less than label sequence length for
    CTC loss computation. It assumes the first dimension of
    input_['feature'] and input_['label'] are sentence length.
    """
    if label_key not in input_:
        return True
    return (
        math.ceil(len(input_[feature_key]) / subsampling_factor)
        > (len(input_[label_key]) + 1) * 2 - 1
    )


def illegal_char_filter(input_):
    """Remove any utterance containing non-english word"""
    transcription = input_["transcription"]

    try:
        transcription.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False

    for letter in transcription:
        if ord(letter) < 0x20 or ord(letter) > 0x7F:
            return False
    return True


def remove_utterance_by_comparing_number(input_):
    """Remove sentences that miss numbers comparing with hybrid recos
    We compared the number of number and remove if the number of the mismatch > 3
    Use case: phone number/credit card number might be deleted due to PII, this filter can remove
    the utterances where those numbers were deleted.
    """
    if "alternative_transcription" not in input_["info"]:
        return True
    if "lexical_hybrid_decode" not in input_["info"]["alternative_transcription"]:
        return True
    if "lexical_tned_human_caption_adjusted" not in input_["info"]["alternative_transcription"]:
        return True

    hybrid_reco = input_["info"]["alternative_transcription"]["lexical_hybrid_decode"].split()
    trans = input_["info"]["alternative_transcription"][
        "lexical_tned_human_caption_adjusted"
    ].split()
    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    hybrid_reco_total = 0
    trans_total = 0
    for n in numbers:
        hybrid_reco_total += hybrid_reco.count(n)
        trans_total += trans.count(n)

    if abs(hybrid_reco_total - trans_total) > 3:
        return False

    return True


def remove_utterance_by_consecutive_numbers(input_):
    """Remove sentences that has three consecutive lexicel numbers
    Use case: removing display format data which still contains lexical format number
    """
    consecutive_count = 0
    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for word in input_["transcription"].split():
        if word in numbers:
            consecutive_count += 1
            if consecutive_count >= 3:
                return False
        else:
            consecutive_count = 0

    return True


def get_complete_sentences(text, cut_type="loose"):
    """Get complete sentences from text by cutting at the punctuations.

    Args:
        text (_type_): the text to be processed.
        cut_type (str, optional): the way to cut the text. Defaults to "loose".

    Returns:
        _type_: the processed complete sentences.
    """
    # Split the paragraph into sentences
    if cut_type == "loose":
        sentences = re.split(r"(?<=[.!?,])[\s\n]+", text)
    elif cut_type == "strict":
        sentences = re.split(r"(?<=[.!?])[\s\n]+", text)
    else:
        raise ValueError(f"cut_type {cut_type} is not supported")

    # Check if the last sentence is complete
    pattern = r".*[.!?,]$" if cut_type == "loose" else r".*[.!?]$"
    if not re.match(pattern, sentences[-1]):
        sentences = sentences[:-1]

    # Return the first n complete sentences
    return " ".join(sentences)


def messages_from_response(input_, train_response=True, response_cut=None):
    """construct messages from generation model response.

    Args:
        input_ (_type_): input sample with "alignment" key, where each alignment is a dict with "instruct" and "answer" keys.
        train_response (bool, optional): whether to train on response. Defaults to True.
        response_cut (_type_, optional): how to cut response to complete. Defaults to None.

    Returns:
        _type_: list of messages, each message is a list of dicts with "role" and "content" keys.
    """

    # response_cut: loose, strict, None
    msgs = []
    if not train_response:
        return msgs
    for alignment in input_["alignment"]:
        instruction = alignment["instruct"]
        if instruction not in ["", "{}"]:
            continue
        response = alignment["answer"]
        if response_cut in ["loose", "strict"]:
            response = get_complete_sentences(response, response_cut)
        if response != "":
            msg = [
                {"role": "user", "content": "<audio>\n"},
                {"role": "assistant", "content": response},
            ]
            msgs.append({"messages": msg})
    return msgs

def remove_utterance_by_punctuation_numbers(
    input_,
    punctuation_list,
    threshold=20):
    """Remove the utterance that punctuation_number is too less.
    """

    punctuation_count = sum(input_["transcription"].count(p) for p in punctuation_list)
    words = input_["transcription"].split()
    word_count = len(words)
    if punctuation_count == 0:
        return False
    
    ratio = word_count / punctuation_count
    return ratio <= threshold


def illegal_word_filter(input_, illegal_words):
    """Remove any utterance containing illegal words"""
    transcription = input_["transcription"]
    for word in illegal_words:
        if word in transcription:
            return False
    return True


def remove_utterance_by_corpus_transcript_pattern(input_, corpus_patterns, original_transcript_patterns):
    """Remove utterances where both the corpus name and the original transcription
    match specified regex patterns. Used to filter bad annotations (e.g. <FILL/>)
    from specific corpora.

    Args:
        input_ (Dict): Sample with input_["info"]["corpusname"] and optionally
            input_["info"]["segmentation"][*]["original_transcription"].
        corpus_patterns (List[str]): Regex patterns for corpus names.
        original_transcript_patterns (List[str]): Regex patterns for original transcription.

    Returns:
        bool: True to keep the utterance, False to filter it out.
    """
    info_chunk = input_.get("info", {})
    corpus_name = info_chunk.get("corpusname", "")

    if not re.search("|".join(corpus_patterns), corpus_name):
        return True  # Corpus doesn't match, keep

    # Get original transcription from segmentation or utterance level
    segs = info_chunk.get("segmentation", [])
    if segs:
        original_transcript = " ".join(seg.get("original_transcription", "") for seg in segs)
    else:
        original_transcript = info_chunk.get("original_transcription", "")

    if not original_transcript:
        return True  # No original transcription to check

    if not re.search("|".join(original_transcript_patterns), original_transcript):
        return True  # Transcription doesn't match pattern, keep

    return False  # Both match, filter out


re_tag = re.compile(r"\(.{1,30}?\)")
re_tag2 = re.compile(r"\[.{1,30}?\]")
re_speaker = re.compile(r"^ *-? *([A-Za-z]+ ){0,2}[A-Za-z]+ *:")

def human_caption_v3_tn(text):
    pos = text.find(":")
    if pos < 32 and pos >= 0:
        text = text[pos + 1 :]
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace(">>", " ")
    text = text.replace(". --", ". ")
    text = text.replace(". -", ". ")
    text = text.replace(", --", ", ")
    text = text.replace(", -", ", ")
    text = text.replace("? --", "? ")
    text = text.replace("? -", "? ")
    text = text.replace("! --", "! ")
    text = text.replace("! -", "! ")
    text = text.replace('" --', '" ')
    text = text.replace('" -', '" ')
    text = text.replace("&gt;&gt;", " ")
    text = text.replace(">>>", " ")
    text = text.replace("&gt;&gt;&gt;", " ")
    text = text.replace("♪", " ")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace(" - ", " ")
    text = re_tag.sub(" ", text)
    text = re_tag2.sub(" ", text)
    text = re_speaker.sub(" ", text)
    text = text.strip()
    if text and text[0] == "-":
        text = text[1:]
    text = " ".join(text.strip().split())

    return text

def get_long_form_transcription(sample, max_dur, min_dur, preserve_speaker_tag=False, **kwargs):
    """Get long form transcription Human caption data.

    Returns:
        _type_: long form transcription.
    """
    assert "info" in sample and "Meta" in sample["info"]
    meta_info = json.loads(sample["info"]["Meta"])
    
    # we need wer info to filter low-quality samples
    wer_info = meta_info["spk_tag_meta"]["Normalized_WER"]
    sample["info"]["Normalized_WER"] = wer_info

    segmentations = meta_info["segmentation"]
    transcriptions = []
    
    total_dur = segmentations[-1]["time"][1] * 100

    if total_dur > max_dur or total_dur < min_dur:
        # filter out here to save time from loading extrme long samples
        return None

    pattern = r'<[^>]*>'
    # Pattern to match <SPK_CHANGE>*</SPK_CHANGE>
    spk_change_pattern = r'<SPK_CHANGE>.*?</SPK_CHANGE>'

    for i in range(len(segmentations)):
        transcription_w_spk_tag = segmentations[i]["original_transcription_with_spk_change"]
        assert preserve_speaker_tag == False, "preserve_speaker_tag is not supported for pravicy issue"
        
        # Remove both patterns from the text
        
        cleaned_tran = re.sub(spk_change_pattern, '', transcription_w_spk_tag)
        cleaned_tran = re.sub(pattern, '', cleaned_tran)
        cleaned_tran = ' '.join(cleaned_tran.split())
        transcriptions.append(cleaned_tran)
    
    sample["transcription"] = human_caption_v3_tn(" ".join(transcriptions))
    return sample
    
def remove_utterance_by_audio_token_length(input_, min_audio_tokens=25, max_audio_tokens=1500):
    if "answer_cv_token" not in input_:
        return True

    semantic_token = input_["answer_cv_token"]
    if len(semantic_token[0]) > 1:
        semantic_length = len(semantic_token[0])
    else:
        semantic_length = len(semantic_token)
    return min_audio_tokens < semantic_length < max_audio_tokens