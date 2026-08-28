# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""On-the-fly abbreviation correction for lexical HT transcripts."""

import json
import logging
import re
from difflib import SequenceMatcher
from itertools import zip_longest

LOG = logging.getLogger(__name__)


class TransAbbrFix:
    """Fix state/direction/address abbreviations in hypothesis transcripts.

    The block aligns a hypothesis transcript, usually ``sample["transcription"]``,
    with a reference transcript, usually
    ``sample["info"]["alternative_transcription"]["lexical_hybrid_decode"]``.
    If the hypothesis uses a full state/direction/address word and the aligned
    reference uses the abbreviation form, the output uses the reference form.
    """

    def __init__(
        self,
        hypothesis_key: str = "transcription",
        reference_key: str = "info/alternative_transcription/lexical_hybrid_decode",
        output_key: str = "transcription",
        hyp_ref_dictionary_json_file: str | None = None,
        log_debug: bool = False,
        aggressive_filter: bool = False,
    ):
        self.hypothesis_key = hypothesis_key.split("/")
        self.reference_key = reference_key.split("/")
        self.output_key = output_key.split("/")
        self.log_debug = log_debug
        self.aggressive_filter = aggressive_filter

        if hyp_ref_dictionary_json_file is None:
            self.abridge_dict = self._default_abridge_dict()
        else:
            try:
                with open(hyp_ref_dictionary_json_file, encoding="utf-8") as json_file:
                    self.abridge_dict = json.load(json_file)
            except FileNotFoundError:
                LOG.warning(
                    "TransAbbrFix dictionary %s not found; using built-in fallback dictionary.",
                    hyp_ref_dictionary_json_file,
                )
                self.abridge_dict = self._default_abridge_dict()

    @staticmethod
    def _default_abridge_dict():
        return {
            "alabama": ["al", "a l"],
            "alaska": ["ak", "a k"],
            "arizona": ["az", "a z"],
            "arkansas": ["ar", "a r"],
            "california": ["ca", "c a"],
            "colorado": ["co", "c o"],
            "connecticut": ["ct", "c t"],
            "delaware": ["de", "d e"],
            "florida": ["fl", "f l"],
            "georgia": ["ga", "g a"],
            "hawaii": ["hi", "h i"],
            "idaho": ["id", "i d"],
            "illinois": ["il", "i l"],
            "indiana": ["in", "i n"],
            "iowa": ["ia", "i a"],
            "kansas": ["ks", "k s"],
            "kentucky": ["ky", "k y"],
            "louisiana": ["la", "l a"],
            "maine": ["me", "m e"],
            "maryland": ["md", "m d"],
            "massachusetts": ["ma", "m a"],
            "michigan": ["mi", "m i"],
            "minnesota": ["mn", "m n"],
            "mississippi": ["ms", "m s"],
            "missouri": ["mo", "m o"],
            "montana": ["mt", "m t"],
            "nebraska": ["ne", "n e"],
            "nevada": ["nv", "n v"],
            "new hampshire": ["nh", "n h"],
            "new jersey": ["nj", "n j"],
            "new mexico": ["nm", "n m"],
            "new york": ["ny", "n y"],
            "north carolina": ["nc", "n c"],
            "north dakota": ["nd", "n d"],
            "ohio": ["oh", "o h"],
            "oklahoma": ["ok", "o k"],
            "oregon": ["or", "o r"],
            "pennsylvania": ["pa", "p a"],
            "rhode island": ["ri", "r i"],
            "south carolina": ["sc", "s c"],
            "south dakota": ["sd", "s d"],
            "tennessee": ["tn", "t n"],
            "texas": ["tx", "t x"],
            "utah": ["ut", "u t"],
            "vermont": ["vt", "v t"],
            "virginia": ["va", "v a"],
            "washington": ["wa", "w a"],
            "west virginia": ["wv", "w v"],
            "wisconsin": ["wi", "w i"],
            "wyoming": ["wy", "w y"],
            "northeast": ["north east", "ne", "n e"],
            "north east": ["northeast", "ne", "n e"],
            "northwest": ["north west", "nw", "n w"],
            "north west": ["northwest", "nw", "n w"],
            "southeast": ["south east", "se", "s e"],
            "south east": ["southeast", "se", "s e"],
            "southwest": ["south west", "sw", "s w"],
            "south west": ["southwest", "sw", "s w"],
            "south": ["s"],
            "east": ["e"],
            "west": ["w"],
            "north": ["n"],
            "square": ["sq", "s q"],
            "place": ["pl", "p l"],
            "street": ["st", "s t"],
            "road": ["rd", "r d"],
            "court": ["ct", "c t"],
            "drive": ["dr", "d r"],
            "lane": ["ln", "l n"],
            "avenue": ["ave", "av e", "a ve", "a v e"],
            "highway": ["hwy", "hw y", "h wy", "h w y"],
            "circle": ["cir", "ci r", "c ir", "c i r"],
            "boulevard": ["blvd", "blv d", "bl vd", "b lvd", "bl v d", "b lv d", "b l vd", "b l v d"],
            "parkway": ["pkwy", "pkw y", "pk wy", "p kwy", "pk w y", "p wk y", "p w ky", "p k w y"],
        }

    @staticmethod
    def _get_nested(obj, keys):
        for key in keys:
            if not isinstance(obj, dict) or key not in obj:
                return None
            obj = obj[key]
        return obj

    @staticmethod
    def _set_nested(obj, keys, value):
        for key in keys[:-1]:
            if key not in obj or not isinstance(obj[key], dict):
                obj[key] = {}
            obj = obj[key]
        obj[keys[-1]] = value

    def __call__(self, example):
        text_hypothesis = self._get_nested(example, self.hypothesis_key)
        text_reference = self._get_nested(example, self.reference_key)
        if text_hypothesis is None or text_reference is None:
            return example

        if "transcription_preTN" in self.reference_key and isinstance(text_reference, str):
            text_reference = text_reference.lower()

        text_fixed = text_hypothesis
        if text_hypothesis:
            text_fixed = self.align_text_seg(text_hypothesis, text_reference, fuzzy_match_word_count=4)
            if self.log_debug and text_fixed != text_hypothesis:
                LOG.info("%s: %s", self.reference_key[-1], text_reference)
                LOG.info("text_hypothesis: %s", text_hypothesis)
                LOG.info("output        : %s", text_fixed)

        self._set_nested(example, self.output_key, text_fixed)
        return example

    @staticmethod
    def process_opcodes(input_opcodes, tx_words, reco_words):
        opcodes = []
        for opcode in input_opcodes:
            op = opcode[0]
            left = tx_words[opcode[1] : opcode[2]]
            right = reco_words[opcode[3] : opcode[4]]

            if op == "equal":
                for word in left:
                    opcodes.append({"op": "equal", "left": word, "right": word})
            elif op == "insert":
                for word in right:
                    opcodes.append({"op": "insert", "left": "", "right": word})
            elif op == "delete":
                for word in left:
                    opcodes.append({"op": "delete", "left": word, "right": ""})
            elif op == "replace":
                for left_word, right_word in zip_longest(left, right):
                    if left_word and right_word:
                        opcodes.append({"op": "replace", "left": left_word, "right": right_word})
                    elif left_word:
                        opcodes.append({"op": "delete", "left": left_word, "right": ""})
                    elif right_word:
                        opcodes.append({"op": "insert", "left": "", "right": right_word})
        return opcodes

    def align_text_seg(self, text_hypothesis, text_reference, fuzzy_match_word_count):
        """
        This is main logic of the TransAbbrFix block. It first aligns the mixedcase transcript and
        richland reco. If the a word in mixedcase trans is a full state/direction name
        (california, nothwest) and the matching word in richland reco is spelled out letters, i.e.
        abbreviation of the full name (CA, NW), then we replace the mixedcase full state/direction
        word with richland's spelled out letters.
        """
        ref_words = re.split(" ", text_hypothesis) if isinstance(text_hypothesis, str) else []
        seg_words = re.split(" |-", text_reference) if isinstance(text_reference, str) else []

        matcher = SequenceMatcher(None, ref_words, seg_words)
        opcodes = self.process_opcodes(matcher.get_opcodes(), ref_words, seg_words)

        ref_segs = []
        ref_segs_str = ""
        index = 0
        set_right_word_matched = set()
        while index < len(opcodes):
            if index + 1 < len(opcodes):
                left_word_two = opcodes[index]["left"] + " " + opcodes[index + 1]["left"]
            else:
                left_word_two = ""
            left_word = opcodes[index]["left"]
            found_match = False
            # (1) First check exact match
            if opcodes[index]["op"] == "equal":
                ref_segs.append(opcodes[index]["left"])  # left_word
                set_right_word_matched.add(index)
                index += 1
                found_match = True
            # (2) Then try to match 2-word case, e.g. "new york" to "n y", "west virginia" to "w v"
            if not found_match and str.lower(left_word_two) in self.abridge_dict:
                right_word_two = opcodes[index]["right"] + " " + opcodes[index + 1]["right"]
                # (2-1) Check corresponding two right words match the abbreviation of the left word
                if str.lower(right_word_two) in self.abridge_dict[str.lower(left_word_two)]:
                    ref_segs.append(right_word_two)
                    set_right_word_matched.add(index)
                    set_right_word_matched.add(index + 1)
                    index += 2
                    found_match = True
                # (2-2) Check single right word from neighboring positions
                if not found_match:
                    for i in range(
                        index - int(fuzzy_match_word_count) // 2,
                        index + int(fuzzy_match_word_count) // 2 + 2,
                    ):
                        if i < 0 or i >= len(opcodes) or i in set_right_word_matched:
                            continue
                        right_word_neighbor = opcodes[i]["right"]
                        if (
                            str.lower(right_word_neighbor)
                            in self.abridge_dict[str.lower(left_word_two)]
                        ):
                            ref_segs.append(right_word_neighbor)
                            set_right_word_matched.add(i)
                            index += 2
                            found_match = True
                            break
                # (2-3) Consider two right words from neighboring positions
                if not found_match:
                    for i in range(
                        index - int(fuzzy_match_word_count) // 2,
                        index + int(fuzzy_match_word_count) // 2 + 1,
                    ):
                        if (
                            i < 0
                            or i + 1 >= len(opcodes)
                            or i in set_right_word_matched
                            or i + 1 in set_right_word_matched
                        ):
                            continue
                        right_word_neighbor_two = (
                            opcodes[i]["right"] + " " + opcodes[i + 1]["right"]
                        )
                        if (
                            str.lower(right_word_neighbor_two)
                            in self.abridge_dict[str.lower(left_word_two)]
                        ):
                            ref_segs.append(right_word_neighbor_two)
                            set_right_word_matched.add(i)
                            set_right_word_matched.add(i + 1)
                            index += 2
                            found_match = True
                            break

                if not found_match and self.aggressive_filter:
                    return ""

            if not found_match and str.lower(left_word) in self.abridge_dict:
                right_word = opcodes[index]["right"]
                # (3-1) Check corresponding right word matches the abbreviation of the left word
                if str.lower(right_word) in self.abridge_dict[str.lower(left_word)]:
                    ref_segs.append(right_word)
                    set_right_word_matched.add(index)
                    index += 1
                    found_match = True

                max_num_right_words_to_search = 2
                for right_phrase in self.abridge_dict[str.lower(left_word)]:
                    max_num_right_words_to_search = max(
                        len(right_phrase.split(" ")), max_num_right_words_to_search
                    )

                # (3-2) Check single right word from neighboring positions
                # (3-3) Consider two right words from neighboring positions
                for right_word_count in range(1, max_num_right_words_to_search + 1):
                    if found_match:
                        break
                    for start in range(
                        index - int(fuzzy_match_word_count) // 2,
                        index + int(fuzzy_match_word_count) // 2 + 1,
                    ):
                        ref_phrase_viable = True
                        word_range_ref = list(range(start, start + right_word_count))
                        for i in word_range_ref:
                            if i in set_right_word_matched:
                                ref_phrase_viable = False
                                break
                        if word_range_ref[0] < 0 or word_range_ref[-1] >= len(opcodes):
                            ref_phrase_viable = False
                        if ref_phrase_viable:
                            ref_phrase = " ".join(
                                [
                                    opcodes[i]["right"]
                                    for i in word_range_ref
                                    if opcodes[i]["right"] != ""
                                ]
                            )
                            if str.lower(ref_phrase) in self.abridge_dict[str.lower(left_word)]:
                                ref_segs.append(ref_phrase)
                                for i in word_range_ref:
                                    set_right_word_matched.add(i)
                                index += 1
                                found_match = True
                                break

                if not found_match and self.aggressive_filter:
                    return ""

            if not found_match:
                index += 1
                if left_word != "":
                    # directly copy left_word (mixedcase) to output
                    ref_segs.append(left_word)
        ref_segs_str = " ".join(ref_segs)

        return ref_segs_str
