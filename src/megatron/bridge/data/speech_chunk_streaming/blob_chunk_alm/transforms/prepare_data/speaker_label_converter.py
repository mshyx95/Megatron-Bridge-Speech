import logging
import os
import json
import re
import ast

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ToolDir = os.path.dirname(os.path.abspath(__file__))


class SpeakerLabelConverter:
    """Detect and convert speaker labels in human captions to <SPK-XXX> format"""
    def __init__(self, llm_decision_dir):
        (self.parenthesis_pattern2decision,
         self.bracket_pattern2decision,
         self.semicolon_pattern2decision) = self.load_llm_decisions(llm_decision_dir)

        self.BRACKET_RX = re.compile(
            r'''                        # VERBOSE mode
                (?P<block>              # whole bracketed bit
                \[ \s* (?P<text_sq> [^\]]+?) \s* \]     # [ … ]
                | \( \s* (?P<text_pa> [^)]+?)  \s* \)     # ( … )
                )
            ''',
            re.VERBOSE
        )

        with open(ToolDir + "/vocab/sounds.txt", "r", encoding="utf-8") as file:
            SOUND_WORDS = [line.strip() for line in file if line.strip()]
        self.SOUND_WORDS = set([wd.lower() for wd in SOUND_WORDS])  # Normalize to lowercase

        with open(ToolDir + "/vocab/speakers.txt", "r", encoding="utf-8") as file:
            SPEAKER_WORDS = [line.strip() for line in file if line.strip()]
        self.SPEAKER_WORDS = set([wd.lower() for wd in SPEAKER_WORDS])  # Normalize to lowercase

        with open(ToolDir + "/vocab/titles.txt", "r", encoding="utf-8") as file:
            TITLE_WORDS = [line.strip() for line in file if line.strip()]
        self.TITLE_WORDS = set([wd.lower() for wd in TITLE_WORDS])  # Normalize to lowercase

        pattern_to_debug_file = ToolDir + "/vocab/patterns_to_debug.txt"
        if os.path.exists(pattern_to_debug_file):
            with open(pattern_to_debug_file, "r", encoding="utf-8") as file:
                self.patterns_to_debug = set([line.strip() for line in file if line.strip()])

    def format_decision_key(self, pattern):
        """Format the pattern/decision key for comparison"""
        # replace all special characters with spaces
        pattern = re.sub(r"[^a-zA-Z0-9]+", " ", pattern)
        pattern = pattern.replace(":", "").replace("-", " ").replace("  ", " ")
        pattern = pattern.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        return pattern.strip().lower()

    def parse_llm_decision_for_semicolon(self, pattern, decision):
        """Check whether the LLM decision indicates that the pattern is a speaker label."""
        if decision["decision"].lower() == "yes":
            return True
        if decision["decision"].lower() == "no":
            return False

        decision_str = self.format_decision_key(decision["decision"])
        pattern_str = self.format_decision_key(pattern)
        if pattern_str == decision_str:
            return True

        if not pattern_str or not decision_str:     # maybe empty after formatting
            return False

        pattern_words = pattern_str.split()
        decision_words = decision_str.split()
        # if pattern_words[0] == decision_str or pattern_words[-1] == decision_str:
        # if pattern_str.startswith(decision_str) or pattern_str.endswith(decision_str):
        # if there is intersection between pattern_words and decision_words
        common_words = set(pattern_words).intersection(set(decision_words))
        if common_words and len(pattern_words) <= 5:  # avoid treating a whole sentence as match
            return True

        return False

    def load_llm_decisions(self, llm_decision_dir):
        decision_files = [f"{llm_decision_dir}/output_for_global_batch_part{i}.tsv"
                          for i in range(1, 5)]

        pattern2decision = {}
        unequal_lines = []
        failed_lines = []
        uncertain_lines = []
        for decision_file in decision_files:
            with open(decision_file, "r", encoding="utf-8") as file:
                lines = [line.strip().split("\t") for line in file.readlines() if line.strip()]

            for line in lines:
                try:
                    patterns = ast.literal_eval(line[1])
                except:
                    patterns = re.findall(r"'''(.*?)'''", line[1])

                if line[2].startswith("```json"):
                    line[2] = line[2][7:]
                if line[2].endswith("```"):
                    line[2] = line[2][:-3]
                try:
                    decisions = ast.literal_eval(line[2])
                except:
                    try:
                        decisions = json.loads(line[2])
                    except:
                        failed_lines.append(line)
                        continue

                if len(patterns) != len(decisions):
                    # print(f"Length mismatch in line: {line}, skipped")
                    unequal_lines.append(line)
                    continue
                for pattern, decision in zip(patterns, decisions):
                    if decision["decision"].lower() != "no" and not self.parse_llm_decision_for_semicolon(pattern, decision):
                        uncertain_lines.append([pattern, decision["decision"]])
                    if pattern not in pattern2decision:
                        pattern2decision[pattern] = decision
                    else:
                        print(f"Duplicate pattern found: {pattern}")

        # output_file = f"{llm_decision_dir}/llm_decision_load_report.txt"
        # with open(output_file, "w", encoding="utf-8") as file:
        #     for line in uncertain_lines:
        #         file.write(f"{line[0]}\t{line[1]}\n")

        # separate into three categories: [], (), others
        parenthesis_pattern2decision = {}
        bracket_pattern2decision = {}
        semicolon_pattern2decision = {}
        for pattern, decision in pattern2decision.items():
            if pattern.startswith("(") or pattern.endswith(")"):
                parenthesis_pattern2decision[pattern] = decision
            elif pattern.startswith("[") or pattern.endswith("]"):
                bracket_pattern2decision[pattern] = decision
            else:
                semicolon_pattern2decision[pattern] = decision

        return parenthesis_pattern2decision, bracket_pattern2decision, semicolon_pattern2decision

    def _is_sound_word(self, word):
        word = word.lower()
        return any(wd in word for wd in self.SOUND_WORDS)

    def _is_speaker_word(self, word):
        word = word.lower()
        return any(wd in word for wd in self.SPEAKER_WORDS)

    def convert_speaker_labels_bracket_parenthesis(self, text):
        def replace_spk_id(detected, match_type):
            pattern2decision = (self.bracket_pattern2decision
                                if match_type == "bracket" else
                                self.parenthesis_pattern2decision)
            spk_tag = None
            spk_id = re.sub("\[|\]", "", detected) if match_type == "bracket" else re.sub("\(|\)", "", detected)
            if self._is_speaker_word(spk_id):
                spk_tag = f"<SPK-{spk_id}>"
            elif self._is_sound_word(spk_id):
                spk_tag = ""
            elif detected in pattern2decision:
                decision = pattern2decision[detected]
                if decision["decision"].lower == "yes" or decision["decision"] == spk_id:
                    spk_tag = f"<SPK-{spk_id}>"
            if spk_tag is None:
                spk_tag = ""
            return spk_tag

        new_text = text
        for m in self.BRACKET_RX.finditer(text):
            spk_tag = None
            if m.group("text_sq"):
                detected = m.group()
                spk_tag = replace_spk_id(detected, "bracket")

            elif m.group("text_pa"):
                detected = m.group()
                spk_tag = replace_spk_id(detected, "parenthesis")

            # replace the detected bracketed text with the speaker tag
            if spk_tag is not None:
                new_text = new_text.replace(detected, spk_tag)

        return new_text

    def is_initial(self, word):
        """Check if the word is an initial (e.g., "D.")"""
        if re.match(r'^[A-Z]\.$', word):
            return True
        return False

    def is_title(self, word):
        """Check if the word is a common title (e.g., "Dr.", "Ms.", "SEN."). 
        Ignore the case of the word."""
        word = word.lower()
        return word in self.TITLE_WORDS

    def find_colons_excluding_times(self, text):
        """
        Return the positions of colons that are NOT part of a time format.
        Time formats include HH:MM and HH:MM:SS (e.g., 00:12, 01:12:34).
        """
        # First, find all time-like patterns so we can skip their colons
        time_pattern = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b')

        # Record all indices of colons that are part of valid time formats
        time_colons = set()
        for match in time_pattern.finditer(text):
            span_text = match.group()
            start = match.start()
            for i, ch in enumerate(span_text):
                if ch == ':':
                    time_colons.add(start + i)

        # Find all colons in the text
        all_colons = [m.start() for m in re.finditer(r':', text)]

        # Return only those not part of a time format
        return [i for i in all_colons if i not in time_colons]

    def detect_candidate_before_semicolon(self, seg, idx):
        """Detect candidate speaker label before the colon at index idx in seg.
        Example input segment and expected output:
            "Well, Dr. Smith:" -> "Dr. Smith:"
            "(Ben Haley):" -> "Ben Haley:"
            "GARY B. NASH:" -> "GARY B. NASH:"
            "SEN. MITCH MCCONNELL (R-KY):" -> "SEN. MITCH MCCONNELL (R-KY):"
            "Ms. Catherine Fife:" -> "Ms. Catherine Fife:"
            "(00:00):" -> None
            "twenty election was stolen from Donald Trump Speaker 1:" -> "Speaker 1:"
            "VOLODYMYR ZELENSKY, Ukrainian President:" -> "VOLODYMYR ZELENSKY, Ukrainian President:"
        """
        # debugging lines:
        # seg = "Hello there. GARY B. NASH: Hello there."
        # seg = "Hello there. Dr. Brown: Good to see you."
        # idx = seg.find(":")
        if seg.startswith("Dr. Fauci"):
            pass
        candidate = None
        candidate_extended = None
        punctuations = ".,;:!?()[]"
        punct_idx = max([seg.rfind(c, 0, idx) for c in punctuations] + [-1])
        # take the part before colon and after punctuation
        label = seg[punct_idx + 1:idx + 1].strip()
        if label and label != ":":
            candidate = label

            # sometimes, candidate does not include name initials or titles if they are before the
            # punctuation, e.g., "Dr. Smith:" -> punctuation is ".", candidate is "Smith:"
            # we need to search backwards for the first space before the punctuation
            candidate_extended = candidate
            space_idx = seg.rfind(" ", 0, punct_idx)
            if seg[punct_idx] == ".":
                if space_idx != -1:
                    word_before = seg[space_idx + 1:punct_idx + 1].strip()
                else:
                    word_before = seg[:punct_idx + 1].strip()

                if self.is_title(word_before):
                    candidate_extended = seg[space_idx + 1:idx + 1].strip()
                elif self.is_initial(word_before):
                    # search for another space before this initial
                    space_idx2 = seg.rfind(" ", 0, space_idx)
                    if space_idx2 != -1:
                        word_before2 = seg[space_idx2 + 1:space_idx].strip()
                        if word_before2[-1] not in ".,;:!?()[]’":
                            # if the word before the initial ends with a punctuation, we do not extend
                            candidate_extended = seg[space_idx2 + 1:idx + 1].strip()
                        else:
                            candidate_extended = seg[space_idx + 1:idx + 1].strip()
                    else:
                        candidate_extended = seg[:idx + 1].strip()

        return candidate, candidate_extended

    def gen_text_to_speaker_mapping(self, text, colon_match, text_to_speaker_mapping):
        candidate, candidate_extended = self.detect_candidate_before_semicolon(text, colon_match)

        if candidate is None or re.match(r"<SPK-.*?>", candidate):
            return

        if candidate in self.semicolon_pattern2decision:
            decision = self.semicolon_pattern2decision[candidate]
            # decision_word = decision["decision"].replace(":", "").replace("-", " ").replace("  ", " ").strip()
            # candidate_word = candidate.replace(":", "").replace("-", " ").replace("  ", " ").strip()
            # if decision["decision"].lower() == "yes" or decision_word == candidate_word:
            if self.parse_llm_decision_for_semicolon(candidate, decision):
                text_to_speaker_mapping[candidate_extended] = f"<SPK-{candidate_extended}>"

        candidate_extended_words = candidate_extended.replace(":", "").split()
        if self.is_title(candidate_extended_words[0]) or candidate_extended.startswith(">>") \
           or self._is_speaker_word(candidate_extended_words[0]):
            if len(candidate_extended_words) <= 5:  # avoid too long names. The chance is low for a long name to be a real speaker label
                text_to_speaker_mapping[candidate_extended] = f"<SPK-{candidate_extended}>"

        if len(candidate_extended_words) <= 2:
            text2 = re.sub(r'\s+', ' ', text.strip())
            candidate_extended2 = re.sub(r'\s+', ' ', candidate_extended.strip())
            if text2.startswith(candidate_extended2) and candidate_extended_words[0][0].isupper():
                # if the candidate is at the beginning of the text, we can consider it as a speaker label
                text_to_speaker_mapping[candidate_extended] = f"<SPK-{candidate_extended}>"

    def convert_speaker_labels_semicolon_v2(self, text):
        # text = "Mr. Smith: Hello there. Dr. Brown: Good to see you. This is a timestamp. Speaker 1: Let's begin."
        col3 = text

        # Step 1: Trim after the last ":"
        last_colon_index = col3.rfind(':')
        if last_colon_index != -1:
            col3 = col3[:last_colon_index + 1]

        # Step 2: Process for first colon
        first_colon_index = col3.find(':')
        if first_colon_index != -1:
            before_colon = col3[:first_colon_index]
            if "\\n" in before_colon:
                last_nl = before_colon.rfind("\\n")
                col3 = col3[last_nl + 2:]  # remove up to \n (2 chars)

        text_to_speaker_mapping = {}
        # Split on \n to handle multi-line content correctly
        segments = col3.strip().split('\\n')  # handle literal \n from input

        new_text = text
        for seg in segments:
            # seg = "Start: (09:30), note: this is fine. Next: (01:12:34), event: later: maybe"
            colon_matches = self.find_colons_excluding_times(seg)

            for colon_match in colon_matches:
                self.gen_text_to_speaker_mapping(seg, colon_match, text_to_speaker_mapping)

        new_text = text
        for candidate, spk_tag in text_to_speaker_mapping.items():
            new_text = new_text.replace(candidate, spk_tag)

        return new_text

    def convert_speaker_labels_semicolon(self, text):
        # text = "Mr. Smith: Hello there. Dr. Brown: Good to see you. (00:00): This is a timestamp. Speaker 1: Let's begin."
        col3 = text

        # Step 1: Trim after the last ":"
        last_colon_index = col3.rfind(':')
        if last_colon_index != -1:
            col3 = col3[:last_colon_index + 1]

        # Step 2: Process for first colon
        first_colon_index = col3.find(':')
        if first_colon_index != -1:
            before_colon = col3[:first_colon_index]
            if "\\n" in before_colon:
                last_nl = before_colon.rfind("\\n")
                col3 = col3[last_nl + 2:]  # remove up to \n (2 chars)

        candidates = set()
        # Split on \n to handle multi-line content correctly
        segments = col3.strip().split('\\n')  # handle literal \n from input

        new_text = text
        for seg in segments:
            colon_matches = [m.start() for m in re.finditer(r':', seg)]

            for colon_match in colon_matches:
                candidate, _ = self.detect_candidate_before_semicolon(seg, colon_match)
                if candidate is not None:
                    candidates.add(candidate)

        new_text = text
        for candidate in candidates:
            # if candidate is already in <SPK-#> format, skip it. This could happend for examples like: (Ben Haley):
            if re.match(r"<SPK-.*?>", candidate):
                continue
            if candidate in self.semicolon_pattern2decision:
                decision = self.semicolon_pattern2decision[candidate]
                if decision["decision"].lower() == "yes" or decision["decision"].replace(":", "") == candidate.replace(":", ""):
                    spk_tag = f"<SPK-{candidate}>"
                    new_text = new_text.replace(candidate, spk_tag)
            else:
                pass

        return new_text

    def __call__(self, lines):
        """Convert speaker labels in the lines using LLM decisions and heuristics."""
        converted_lines = []

        for line in lines:
            # Split the line into time1, time2, and text
            parts = line.strip().split('\t')
            if len(parts) != 3:
                converted_lines.append(line)
                continue

            start, end, text = parts
            text = re.sub(r'\s+', ' ', text).strip()
            if any(pat in text for pat in self.patterns_to_debug):
                # find and remove the pattern from self.patterns_to_debug to avoid duplicate logs
                for pat in self.patterns_to_debug:
                    if pat in text:
                        self.patterns_to_debug.remove(pat)
                        break
                LOG.info(f"Debug pattern found in line: {line}")
            new_text = self.convert_speaker_labels_bracket_parenthesis(text)
            new_text = self.convert_speaker_labels_semicolon_v2(new_text)
            new_text = new_text.strip()
            new_text = new_text.replace("\t", " ")
            new_text = re.sub(r'\s+', ' ', new_text)
            converted_lines.append(f"{start}\t{end}\t{new_text}")

        return converted_lines
