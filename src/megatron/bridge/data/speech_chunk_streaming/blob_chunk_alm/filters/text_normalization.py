# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""TN process that takes a rule file which is focusing on word level tn.
It requires a specified format in the rule file with specified file name.
And the rule file is locale-specific.
"""

import os
import unicodedata
from unidecode import unidecode
import regex as re
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger

logger = get_logger(__name__)


class TNRuleWord:
    """TN process that takes a rule file which is focusing on word level tn

    Args:
        rule_file (string, optional): TN rule file. If not provided, will use
            this one: cascades/data/resource/en-us/e2e.tn.rule.regex.word.txt
        case_normalize (string, optional): "upper" or "lower" or None. Default is None,
            which means doing nothing.
        unicode_normalize (bool, optional): If True will do unidecode(),
            default is True
    """

    def __init__(self, rule_file=None, case_normalize=None, unicode_normalize=True):
        """Init"""
        if case_normalize is not None:
            self.case_normalize = case_normalize.lower()
        else:
            self.case_normalize = case_normalize

        self.unicode_normalize = unicode_normalize

        self.textnorm_rules = []
        # Dict for O(1) exact whole-word substitution rules (pattern "^word$" -> replacement)
        self.exact_word_sub = {}

        assert rule_file is not None, "please provide rule file if using text normalization"
            

        logger.info("Load TN rule file %s", rule_file)
        self._read_tn_rule_data(rule_file)

        # cached already TNed tokens to save computation cost
        self.tn_pair_memory = {}

    def _process_a_word(self, input_word):
        """TN a single word"""

        if input_word in self.tn_pair_memory:
            return self.tn_pair_memory[input_word]

        tned_token = input_word

        for tn_rule in self.textnorm_rules:
            if tn_rule[0] == "sub":
                tned_token = re.sub(tn_rule[1], tn_rule[2], tned_token)
            elif tn_rule[0] == "find":
                cand = re.findall(tn_rule[1], tned_token)
                if cand:
                    if isinstance(cand[0], (list, tuple)):
                        tned_token = "".join(cand[0])
                    else:
                        tned_token = " ".join(cand)
            elif tn_rule[0] == "find-sub":
                if re.findall("[0-9]", tned_token):
                    tned_token = re.sub("-", " ", tned_token)

        # Exact whole-word substitution after regex normalization
        if tned_token in self.exact_word_sub:
            tned_token = self.exact_word_sub[tned_token]

        if self.unicode_normalize:
            norm_token = unicodedata.normalize("NFD", tned_token)
            norm_token = "".join(c for c in norm_token if not unicodedata.combining(c))
            if norm_token != tned_token:
                tned_token = "###REMOVE_WHOLE_LINE###"

        if self.case_normalize == "lower":
            tned_token = tned_token.lower()
        elif self.case_normalize == "upper":
            tned_token = tned_token.upper()

        tned_token = tned_token.strip()
        self.tn_pair_memory[input_word] = tned_token

        return tned_token

    def process_a_line(self, input_line):
        """process a line"""
        words = input_line.strip().split(" ")
        if not words:
            return ""

        tn_trans = []
        for word in words:
            tn_word = self._process_a_word(word)
            if "###REMOVE_WHOLE_LINE###" in tn_word.upper():
                return ""

            if tn_word:
                tn_trans.append(tn_word)

        tn_trans_str = " ".join(tn_trans)

        if self.unicode_normalize:
            tn_trans_str = unidecode(tn_trans_str)

        return tn_trans_str

    @staticmethod
    def _is_exact_word_sub(rule_tuple):
        """Check if a sub rule is an exact whole-word match (^literal$) with no regex metacharacters."""
        if rule_tuple[0] != "sub":
            return False
        pattern = rule_tuple[1]
        if not (pattern.startswith("^") and pattern.endswith("$")):
            return False
        inner = pattern[1:-1]
        # Reject if inner part contains any regex metacharacters
        if re.search(r"[\\.+*?\[\](){}|^$]", inner):
            return False
        return True

    def _read_tn_rule_data(self, tn_rule_file):
        """read the tn rule data file"""
        if tn_rule_file.startswith("az://"):
            import blobfile as bf
            file = bf.BlobFile(tn_rule_file, "r")
        else:
            file = open(tn_rule_file, encoding="utf8")
        
        exact_count = 0
        for line in file:
            if line.strip() == "" or line.startswith("#"):
                logger.info("skip : %s", line)
                continue

            sline = line.rstrip().split("\t")
            assert len(sline) == 3 or len(sline) == 4 or len(sline) == 5, (
                "Error format of rule: " + line
            )
            rule = tuple(sline)
            if self._is_exact_word_sub(rule):
                word = rule[1][1:-1]  # strip ^ and $
                self.exact_word_sub[word] = rule[2]
                exact_count += 1
            else:
                self.textnorm_rules.append(rule)

        if exact_count > 0:
            logger.info(
                "Loaded %d exact-word rules into dict, %d regex rules remain",
                exact_count,
                len(self.textnorm_rules),
            )


class CharMapper:
    """Character-level mapper that loads a mapping file and performs:
    (A) Character-level text replacement (before tokenization)
    (B) Token-level OOV fallback (during token→ID conversion)

    The mapping file format is one mapping per line: ``source_char\\ttarget_char``
    Lines starting with ``#`` and empty lines are skipped.

    Args:
        map_file (str): Path to the ``.map`` file (local path or ``az://`` blob path).
    """

    def __init__(self, map_file: str):
        self.char_map = {}
        self._load_map_file(map_file)
        # Build a str.translate table for O(n) replacement
        self._trans_table = str.maketrans(self.char_map)
        logger.info("CharMapper loaded %d mappings from %s", len(self.char_map), map_file)

    def _load_map_file(self, map_file: str):
        if map_file.startswith("az://"):
            import blobfile as bf
            fh = bf.BlobFile(map_file, "r")
        else:
            fh = open(map_file, encoding="utf-8")

        with fh:
            for line in fh:
                line = line.rstrip("\n\r")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    logger.warning("CharMapper: skipping malformed line: %r", line)
                    continue
                src, tgt = parts
                if len(src) == 1:
                    self.char_map[src] = tgt

    def replace_chars(self, text: str) -> str:
        """Apply character-level replacement to text (pre-tokenization)."""
        return text.translate(self._trans_table)


class CaseNorm:

    def __init__(self, case_normalize="lower"):
        """Init"""
        self.case_normalize = case_normalize.lower()
        
    def process_a_line(self, input_line):
        """process a line"""
        
        if self.case_normalize == "lower":
            return input_line.lower()
        elif self.case_normalize == "upper":
            return input_line.upper()
        elif self.case_normalize == "capitalize":
            return input_line.lower().capitalize()
        else:
            raise NotImplementedError