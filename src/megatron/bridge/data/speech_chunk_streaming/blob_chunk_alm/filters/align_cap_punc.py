"""Align GPT transcript and GPT apply capitalization and punctuation to DPP transcriptions."""

import re
import logging
import numpy as np

from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger

logger = get_logger(__name__)

WER_COST_DEL = 3
WER_COST_INS = 3
WER_COST_SUB = 3

def edit_distance(r, h):
    """
    This function is to calculate the edit distance of reference sentence and the hypothesis
    sentence. Main algorithm used is dynamic programming.

    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py

    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint16).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                # d[0][j] = j
                d[0][j] = j * WER_COST_INS
            elif j == 0:
                d[i][0] = i * WER_COST_DEL
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + WER_COST_SUB
                insert = d[i][j - 1] + WER_COST_INS
                delete = d[i - 1][j] + WER_COST_DEL
                d[i][j] = min(substitute, insert, delete)
    return d


def get_alignment(r, h, d):  # pylint: disable=too-many-statements
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calculating the editing distance of h and r.
    """
    x = len(r)
    y = len(h)
    max_len = 3 * (x + y)

    align_ref_words = []
    align_hyp_words = []
    alignlist = []
    align_ref = ""
    align_hyp = ""
    alignment = ""

    while True:
        if (x <= 0 and y <= 0) or (len(alignlist) > max_len):
            break
        if x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            align_hyp = " " + h[y - 1] + align_hyp
            align_hyp_words.append(h[y - 1])
            align_ref = " " + r[x - 1] + align_ref
            align_ref_words.append(r[x - 1])
            alignment = " " * (len(r[x - 1]) + 1) + alignment
            alignlist.append("C")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + WER_COST_SUB:
            ml = max(len(h[y - 1]), len(r[x - 1]))
            align_hyp = " " + h[y - 1].ljust(ml) + align_hyp
            align_hyp_words.append(h[y - 1])
            align_ref = " " + r[x - 1].ljust(ml) + align_ref
            align_ref_words.append(r[x - 1])
            alignment = " " + "S" + " " * (ml - 1) + alignment
            alignlist.append("S")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif y >= 1 and d[x][y] == d[x][y - 1] + WER_COST_INS:
            align_hyp = " " + h[y - 1] + align_hyp
            align_hyp_words.append(h[y - 1])
            align_ref = " " + "*" * len(h[y - 1]) + align_ref
            align_ref_words.append("*" * len(h[y - 1]))
            alignment = " " + "I" + " " * (len(h[y - 1]) - 1) + alignment
            alignlist.append("I")
            x = max(x, 0)
            y = max(y - 1, 0)
        else:
            align_hyp = " " + "*" * len(r[x - 1]) + align_hyp
            align_hyp_words.append("*" * len(r[x - 1]))
            align_ref = " " + r[x - 1] + align_ref
            align_ref_words.append(r[x - 1])
            alignment = " " + "D" + " " * (len(r[x - 1]) - 1) + alignment
            alignlist.append("D")
            x = max(x - 1, 0)
            y = max(y, 0)

    align_ref = align_ref[1:]
    align_hyp = align_hyp[1:]
    alignment = alignment[1:]

    return (
        alignlist[::-1],
        {
            "align_ref": align_ref,
            "align_hyp": align_hyp,
            "align_ref_words": align_ref_words[::-1],
            "align_hyp_words": align_hyp_words[::-1],
            "alignment": alignment,
        },
    )

def wer_single(r, h):
    """
    This function is to calculate the WER of reference sentence and the hypothesis
    sentence.

    Attributes:
        r -> the reference sentence (str).
        h -> the hypothesis sentence (str).
    """
    if isinstance(r, str) and isinstance(h, str):
        r = r.strip().split()
        h = h.strip().split()
    edit_distance_matrix = edit_distance(r=r, h=h)
    alignment, alignment_out = get_alignment(r=r, h=h, d=edit_distance_matrix)

    num_cor = np.sum([s == "C" for s in alignment])
    num_del = np.sum([s == "D" for s in alignment])
    num_ins = np.sum([s == "I" for s in alignment])
    num_sub = np.sum([s == "S" for s in alignment])
    num_err = num_del + num_ins + num_sub
    num_ref = len(r)

    return {
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": num_cor,
        "num_del": num_del,
        "num_ins": num_ins,
        "num_sub": num_sub,
        "num_err": num_err,
        "num_ref": num_ref,
    }

def apply_capitalization(
    original: str,
    model: str,
    punc_str: str,
    puncs_to_remove: str,
    contains_punc: bool = False,
    log_debug: bool = False,
):
    """
    Apply capitalization from model output to original text.

    Parameters
    ----------
    original : str
        Original text (lowercase and has no punctuation).
    model : str
        Model text (may contain capitalization and punctuation).
    punc_str : str
        String of punctuation characters.
    puncs_to_remove : str
        Punctuation removal strategy: "LEADING_TRAILING" or "ALL".
    contains_punc : bool
        Whether the model text contains punctuation.
    log_debug : bool
        Whether to print debug information (optional, default is False).
    """
    if contains_punc:
        model = remove_punctuation(model, punc_str, puncs_to_remove)

    updated_original = list(original)
    if len(original) == len(model):
        # if the lengths are the same, we can apply all capitalization
        for i, c in enumerate(model):
            if c.isupper():
                updated_original[i] = original[i].upper()
            elif c.islower():
                updated_original[i] = original[i].lower()
            # for digits, whitespace, and punctuation we keep the original text.
    else:
        # if the lengths are different, we can only apply capitalization to first letter
        if log_debug:
            print(
                "Length mismatch between original and model text:"
                " only apply capitalization to first letter."
            )
        if len(model) > 0 and len(original) > 0:
            if model[0].isupper():
                updated_original[0] = original[0].upper()
            elif model[0].islower():
                updated_original[0] = original[0].lower()
            # for digits, whitespace, and punctuation we keep the original text.
        else:
            if log_debug:
                print("[CAUTION] Model or original text is empty: cannot apply capitalization.")

    return "".join(updated_original)


def apply_punctuation(
    original: str,
    model: str,
    punc_str: str,
    puncs_to_remove: str,
    contains_cap: bool = False,
    log_debug: bool = False,
):
    """
    Apply punctuation from model output to original text.

    Parameters
    ----------
    original : str
        Original text (may contain capitalization but has no punctuation).
    model : str
        Model text (may contain capitalization and punctuation).
    punc_str : str
        String of punctuation characters.
    puncs_to_remove : str
        Punctuation removal strategy: "LEADING_TRAILING" or "ALL".
    contains_cap : bool
        Whether the model text contains capitalization.
    log_debug : bool
        Whether to print debug information (optional, default is False).
    """
    if contains_cap:
        model = model.lower()

    if original.lower() == remove_punctuation(model, punc_str, puncs_to_remove):
        # The base words are the same, so apply/replace all DPP punctuation with GPT punctuation.
        if puncs_to_remove != "ALL":
            # Remove all punctuation from original text, since it may contain inner-word punctuation
            original = remove_punctuation(original, punc_str, "ALL")
        updated_original = []
        j = 0  # index in original string
        for i, character in enumerate(model):
            if character in punc_str:
                updated_original.append(character)
            else:
                updated_original.append(original[j])
                j += 1
    else:
        # The base words are different, so we can only apply leading/trailing punctuation.
        if log_debug:
            print(
                "Word mismatch between original and model text:"
                " only apply leading/trailing punctuation."
            )
        updated_original = []

        # Add leading punctuation
        i = 0
        while i < len(model) and model[i] in punc_str:
            updated_original.append(model[i])
            i += 1

        # Add original text
        updated_original.append(original)

        # Add trailing punctuation
        j = len(model) - 1
        while j >= 0 and model[j] in punc_str:
            updated_original.append(model[j])
            j -= 1
    return "".join(updated_original)


def validate_indication(
    align_dpp: list, align_gpt: list, indication: list, log_debug: bool = False
):
    """
    Check if the indication codes match the aligned DPP and GPT outputs.

    Parameters
    ----------
    align_dpp : list
        Aligned DPP output.
    align_gpt : list
        Aligned GPT output.
    indication : list
        Indication codes for alignment.
    log_debug : bool
        Whether to print debug information (optional, default is False).

    Returns
    -------
    bool
        True if the indication codes match the aligned outputs, False otherwise.
    """
    for i, code in enumerate(indication):
        if code == "C":
            if align_dpp[i] != align_gpt[i]:
                if log_debug:
                    print(
                        f"[CAUTION] Indication C mismatch at {i}: {align_dpp[i]} != {align_gpt[i]}"
                    )
                return False
        elif code == "I":
            if "*" not in align_dpp[i]:
                if log_debug:
                    print(
                        f"[CAUTION] Indication I mismatch at {i}: {align_dpp[i]} does not contain *"
                    )
                return False
        elif code == "D":
            if "*" not in align_gpt[i]:
                if log_debug:
                    print(
                        f"[CAUTION] Indication D mismatch at {i}: {align_gpt[i]} does not contain *"
                    )
                return False
        elif code == "S":
            if align_dpp[i] == align_gpt[i]:
                if log_debug:
                    print(
                        f"[CAUTION] Indication S mismatch at {i}: {align_dpp[i]} == {align_gpt[i]}"
                    )
                return False
        else:
            raise ValueError(f"Unrecognized indication code: {code}")

    return len(align_dpp) == len(indication) == len(align_gpt)


def remove_punctuation(text: str, punc_str: str, puncs_to_remove: str):
    """
    Remove punctuation from the text. Depending on puncs_to_remove parameter, may either remove
    leading/trailing or all punctuation.

    Parameters
    ----------
    text : str
        The text from which to remove punctuation.
    punc_str : str
        String of punctuation characters to remove.
    puncs_to_remove : str
        Punctuation removal strategy: "LEADING_TRAILING" or "ALL".

    Returns
    -------
    str
        The text with punctuation removed.
    """
    if puncs_to_remove == "LEADING_TRAILING":
        # Remove leading and trailing punctuation
        return text.strip(punc_str)

    if puncs_to_remove == "ALL":
        # Remove all punctuation
        return text.translate(str.maketrans("", "", punc_str))

    raise ValueError(
        f"Invalid value for puncs_to_remove: {puncs_to_remove}."
        f"Must be 'LEADING_TRAILING' or 'ALL'."
    )


def identify_changes(
    ori_hyp: list, align_hyp: list, indication: list, punc_str: str, puncs_to_remove: str, wav=None
):
    """
    Identify what changed between original and aligned GPT output.

    Parameters
    ----------
    ori_hyp : list
        Original GPT output.
    align_hyp : list
        Aligned GPT output.
    indication : list
        Indication of changes between original and aligned GPT output.
    punc_str : str
        String of punctuation characters.
    puncs_to_remove : str
        Punctuation removal strategy: "LEADING_TRAILING" or "ALL".
    wav : str
        The wav file name (optional, for debugging purposes).

    Returns
    -------
    list
        A list of changes between original and aligned GPT output. At each index, the
        value can be "C" (capitalization), "P" (punctuation), "CP" (capitalization and
        punctuation), "D" (deletion), or None (no change).
    """

    changes = []
    for i, word in enumerate(ori_hyp):
        lowercase_ori_word = word.lower()
        ori_word_no_punc = remove_punctuation(word, punc_str, puncs_to_remove)
        normalized_ori_word = ori_word_no_punc.lower()

        lowercase_align_word = align_hyp[i].lower()
        align_word_no_punc = remove_punctuation(align_hyp[i], punc_str, puncs_to_remove)
        normalized_align_word = align_word_no_punc.lower()
        if indication[i] == "D":
            # case: word was deleted by GPT
            changes.append("D")
        elif word == align_hyp[i]:
            # case: word = aligned word (no change)
            changes.append(None)
        elif lowercase_ori_word == lowercase_align_word:
            # case: word.lower = aligned word.lower (cap diff)
            changes.append("C")
        elif ori_word_no_punc == align_word_no_punc:
            # case: word.remove_punc = aligned word.remove_punc (punc diff)
            changes.append("P")
        elif normalized_ori_word == normalized_align_word:
            # case: word.lower.remove_punc = aligned word.lower.remove_punc (cap/punc diff)
            changes.append("CP")
        else:
            raise ValueError(
                f"Unrecognized change between original and aligned GPT output for {wav}: "
                f"{word} vs. {align_hyp[i]}"
            )
    return changes


def apply_identified_changes(
    align_ref: list,
    ori_ref: list,
    ori_hyp: list,
    changes: list,
    punc_str: str,
    puncs_to_remove: str,
    wav: str = None,
    log_debug: bool = False,
):
    """
    Apply identified changes to aligned DPP output.

    Parameters
    ----------
    align_ref : list
        Aligned DPP output.
    ori_ref : list
        Original DPP output.
    ori_hyp : list
        Original GPT output.
    changes : list
        List of changes between original and aligned GPT output. The value at each index can
        only be "C" (capitalization), "P" (punctuation), "CP" (capitalization and punctuation),
        "D" (deletion), or None (no change).
    punc_str : str
        String of punctuation characters.
    puncs_to_remove : str
        Punctuation removal strategy: "LEADING_TRAILING" or "ALL".
    wav : str
        The wav file name (optional, for debugging purposes).
    log_debug : bool
        Whether to print debug information (optional, default is False).

    Returns
    -------
    list
        The updated aligned DPP output after applying the changes from the change list.
    """
    # apply changes to aligned DPP output word-by-word
    updated_dpp = []
    for i, change_code in enumerate(changes):
        if change_code == "C":
            # apply GPT capitalization
            updated_dpp.append(
                apply_capitalization(
                    align_ref[i],
                    ori_hyp[i],
                    punc_str,
                    puncs_to_remove,
                    contains_punc=True,
                    log_debug=log_debug,
                )
            )
        elif change_code == "P":
            # apply GPT punctuation
            updated_dpp.append(
                apply_punctuation(
                    align_ref[i],
                    ori_hyp[i],
                    punc_str,
                    puncs_to_remove,
                    contains_cap=True,
                    log_debug=log_debug,
                )
            )
        elif change_code == "CP":
            # apply both capitalization and punctuation. Capitalization must be applied first.
            updated_dpp.append(
                apply_punctuation(
                    apply_capitalization(
                        align_ref[i],
                        ori_hyp[i],
                        punc_str,
                        puncs_to_remove,
                        contains_punc=True,
                        log_debug=log_debug,
                    ),
                    ori_hyp[i],
                    punc_str,
                    puncs_to_remove,
                    contains_cap=True,
                    log_debug=log_debug,
                )
            )
        elif change_code == "D":
            # deletion compared to DPP, insert aligned GPT output with *
            # for easier alignment with original GPT output in step 3
            updated_dpp.append(ori_ref[i])
        else:
            if change_code is not None:
                if log_debug:
                    print(f"Unrecognized change code for {wav}: {change_code}")
                return None, None, None
            updated_dpp.append(align_ref[i])
    return updated_dpp


def align_utterance(  # pylint: disable=too-many-return-statements
    dpp_trans: str,
    gpt_trans: str,
    punc_str: str,
    puncs_to_remove: str,
    enable_alignment_err_fallback: bool,
    alignment_err_threshold: float,
    wav: str = None,
    log_debug: bool = False,
):
    """
    Apply capitalization and punctuation from GPT to DPP for a single utterance
    after doing alignment.

    Parameters
    ----------
    dpp_trans : str
        The DPP transcription.
    gpt_trans : str
        The GPT transcription.
    punc_str : str
        String of punctuation characters.
    puncs_to_remove : str
        Punctuation removal strategy: "LEADING_TRAILING" or "ALL".
    enable_alignment_err_fallback : bool
        Whether to fallback to DPP segment transcription when alignment error is high.
    alignment_err_threshold : float
        Threshold for alignment error rate fallback.
    wav : str
        The wav file name (optional, for debugging purposes).
    log_debug : bool
        Whether to print debug information (optional, default is False).

    Returns
    -------
    (align_info, err_rate, err_count_dict) : tuple
        align_info : dict
            A dictionary containing the aligned DPP and GPT transcriptions,
            as well as additional debugging information.
        err_rate : float
            The error rate of the alignment.
        err_count_dict : dict
            A dictionary containing the number of substitutions, insertions,
            and deletions made by GPT.
    """
    # 1. align DPP output with GPT output (no punctuation or capitalization)
    alignment = wer_single(  # function which uses DP for edit distance
        [
            remove_punctuation(
                i, punc_str, puncs_to_remove
            )  # remove punctutation from the DPP output
            for i in dpp_trans.lower().split()
        ],
        [
            remove_punctuation(
                i, punc_str, puncs_to_remove
            )  # remove punctutation from the GPT output
            for i in gpt_trans.lower().split()
        ],
    )

    ori_ref = re.split(r"\s+", dpp_trans.strip())  # split on whitespace
    ori_hyp = re.split(r"\s+", gpt_trans.strip())  # split on whitespace
    align_ref = alignment["alignment_out"]["align_ref_words"]
    align_hyp = alignment["alignment_out"]["align_hyp_words"]
    if "".join(align_ref) != alignment["alignment_out"]["align_ref"].replace(" ", "") or "".join(
        align_hyp
    ) != alignment["alignment_out"]["align_hyp"].replace(" ", ""):
        return None, None, None
    indication = alignment["alignment"].copy()
    if len(align_ref) != len(align_hyp):
        if log_debug:
            print("WER alignment length mismatch")
        return None, None, None
    validate_indication(align_ref, align_hyp, indication, log_debug=log_debug)

    # Calculating error
    err = [i for i in indication if i != "C"]  # total num of edits (sub + ins + del)
    tot_tokens = [i for i in indication if i != "I"]  # tokens from ref
    if len(tot_tokens) == 0:
        err_rate = 100
    else:
        err_rate = round(len(err) / len(tot_tokens), 5) * 100

    if enable_alignment_err_fallback and err_rate > alignment_err_threshold:
        # alignment error is too high, fallback to DPP result directly
        if log_debug:
            print(
                f"{wav} alignment error rate {err_rate} exceeds threshold"
                f"{alignment_err_threshold}: falling back to DPP result"
            )
        return (
            {
                "ori_ref": ori_ref,
                "ori_hyp": ori_hyp,
                "indication": indication,
                "align_dpp": align_ref,
                "align_gpt": align_hyp,
                "changes": None,
                "updated_dpp": dpp_trans.split(" "),  # split by single space, not whitespace
            },
            err_rate,
            {
                "sub": alignment["num_sub"],
                "ins": alignment["num_ins"],
                "del": alignment["num_del"],
            },
        )

    # 2. align GPT original output with GPT aligned output (no punctuation or capitalization)
    i = 0  # for aligned GPT output and indication
    while i < len(align_hyp):
        if indication[i] == "D":
            # deletion compared to DPP, insert aligned GPT output for easier
            # alignment with original GPT output in step 3
            ori_hyp.insert(i, align_hyp[i])
            i += 1
        elif indication[i] == "I":
            # original GPT output word was inserted by GPT.
            # remove it from the original GPT output, aligned DPP output,
            # aligned GPT output, and indication for alignment with DPP later
            indication.pop(i)
            align_ref.pop(i)
            align_hyp.pop(i)
            ori_hyp.pop(i)
        else:
            # Substitution or correct. Substitution is normally punctuation related,
            # so we need to check if the original GPT output word is the same as the
            # aligned GPT output word after removing punctutation.
            normalized_ori_hyp = (
                remove_punctuation(ori_hyp[i], punc_str, puncs_to_remove).lower()
                if i < len(ori_hyp)
                else None
            )
            if normalized_ori_hyp != align_hyp[i]:
                return None, None, None
            i += 1

    if len(ori_hyp) != len(align_hyp):
        if log_debug:
            print(f"Alignment length mismatch for {wav}: {len(ori_hyp)} != {len(align_hyp)}")
        return None, None, None

    # 3. identify changes are between originlal GPT output and aligned GPT output for each word
    changes = identify_changes(ori_hyp, align_hyp, indication, punc_str, puncs_to_remove, wav=wav)

    # 4. apply identified changes from step 3 to aligned DPP output
    if len(align_hyp) != len(align_ref) != len(ori_ref) != len(changes):
        if log_debug:
            print(
                f"Alignment length mismatch for {wav}: "
                f"{len(align_hyp)} != {len(align_ref)} != {len(ori_ref)} != {len(changes)}"
            )
        return None, None, None

    updated_dpp = apply_identified_changes(
        align_ref,
        ori_ref,
        ori_hyp,
        changes,
        punc_str,
        puncs_to_remove,
        wav=wav,
        log_debug=log_debug,
    )

    if len(updated_dpp) != len(align_ref) != len(ori_ref):
        if log_debug:
            print(
                f"Updated DPP length mismatch for {wav}: "
                f"{len(updated_dpp)} != {len(align_ref)} != {len(ori_ref)}"
            )
        return None, None, None

    return (
        {
            "ori_ref": ori_ref,
            "ori_hyp": ori_hyp,
            "indication": indication,
            "align_dpp": align_ref,
            "align_gpt": align_hyp,
            "changes": changes,
            "updated_dpp": updated_dpp,
        },
        err_rate,
        {
            "sub": alignment["num_sub"],
            "ins": alignment["num_ins"],
            "del": alignment["num_del"],
        },
    )
