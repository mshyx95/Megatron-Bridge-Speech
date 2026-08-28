# %%
import re
import random
import unicodedata
from collections import defaultdict, deque

def text_norm(word, tn_prob=1.0):
    """Normalize the text by removing special characters and converting to lowercase."""
    if random.random() <= tn_prob:
        word = " ".join(re.findall(r"[\w'\-]+", word))
        word = word.lower()
    return word


def is_thai_char(char):
    """Check if a character is a Thai script character."""
    cp = ord(char)
    return 0x0E00 <= cp <= 0x0E7F


def is_non_space_script_char(char):
    """Check if a character belongs to a script that does not use spaces between words.
    Currently covers CJK and Thai."""
    return is_cjk_char(char) or is_thai_char(char)


def is_cjk_char(char):
    """Check if a character is a CJK (Chinese, Japanese, or Korean) character."""
    cp = ord(char)
    return (
        # CJK Unified Ideographs (Chinese, Japanese Kanji, Korean Hanja)
        (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or
        (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F) or
        (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF) or
        (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F) or
        # Japanese Hiragana & Katakana
        (0x3040 <= cp <= 0x309F) or  # Hiragana
        (0x30A0 <= cp <= 0x30FF) or  # Katakana
        (0x31F0 <= cp <= 0x31FF) or  # Katakana Phonetic Extensions
        (0xFF65 <= cp <= 0xFF9F) or  # Halfwidth Katakana
        # Korean Hangul
        (0xAC00 <= cp <= 0xD7AF) or  # Hangul Syllables
        (0x1100 <= cp <= 0x11FF) or  # Hangul Jamo
        (0x3130 <= cp <= 0x318F) or  # Hangul Compatibility Jamo
        (0xA960 <= cp <= 0xA97F) or  # Hangul Jamo Extended-A
        (0xD7B0 <= cp <= 0xD7FF)     # Hangul Jamo Extended-B
    )


def is_korean_char(char):
    """Check if a character is a Korean Hangul character."""
    cp = ord(char)
    return (
        (0xAC00 <= cp <= 0xD7AF) or  # Hangul Syllables
        (0x1100 <= cp <= 0x11FF) or  # Hangul Jamo
        (0x3130 <= cp <= 0x318F) or  # Hangul Compatibility Jamo
        (0xA960 <= cp <= 0xA97F) or  # Hangul Jamo Extended-A
        (0xD7B0 <= cp <= 0xD7FF)     # Hangul Jamo Extended-B
    )


def has_korean(text):
    """Check if the text contains any Korean Hangul characters."""
    return any(is_korean_char(c) for c in text)


def has_thai(text):
    """Check if the text contains any Thai characters."""
    return any(is_thai_char(c) for c in text)


def is_cjk_punctuation(char):
    """Check if a character is a CJK punctuation mark (Chinese, Japanese, or Korean)."""
    cp = ord(char)
    return (
        char in '，。！？；：、︰' 
    )


def is_english_punctuation(char):
    """Check if a character is an English punctuation mark."""
    return char in '.,!?;:' #\'"()[]{}—–-…'
    #return char in '.,!?;:\'"()[]{}—–-…'


def is_punctuation(char):
    """Check if a character is any punctuation (CJK or English)."""
    return is_cjk_punctuation(char) or is_english_punctuation(char)


def is_non_space_script(char):
    """Check if a character belongs to a non-space-delimited script (CJK/Thai character or CJK punctuation)."""
    return is_non_space_script_char(char) or is_cjk_punctuation(char)


def has_english(text):
    """Check if the text contains any English letters."""
    return bool(re.search(r'[a-zA-Z]', text))


def split_text(text, max_len=1):
    """Split the text into pieces of max_len words.
    Characters from non-space-delimited scripts (Chinese, Japanese, Korean, Thai)
    are split into individual tokens."""
    contains_english = has_english(text)
    contains_korean = has_korean(text)
    # if contains_english:
    #     print(f"Text contains English: {text}")
    tokens = []
    current_word = []
    for char in text:
        if char.isspace():
            if current_word:
                tokens.extend("".join(current_word).split())
                current_word = []
            if contains_korean:
                tokens.append(char)
            continue
        if is_non_space_script(char):
            if current_word:
                tokens.extend("".join(current_word).split())
                current_word = []
            tokens.append(char)
        elif is_english_punctuation(char):
            if current_word:
                tokens.extend("".join(current_word).split())
                current_word = []
            tokens.append(char)
        else:
            current_word.append(char)
    if current_word:
        tokens.extend("".join(current_word).split())
    
    if max_len <= 1:
        return tokens
    pieces = []
    while tokens:
        # If current token is punctuation (Chinese or English), keep it as individual piece
        if len(tokens[0]) == 1 and is_punctuation(tokens[0]):
            pieces.append(tokens[0])
            tokens = tokens[1:]
            continue
        # Find how many non-punctuation tokens we can take
        max_take = 0
        for t in tokens:
            if len(t) == 1 and is_punctuation(t):
                break
            max_take += 1
        n = random.randint(1, min(max_len, max_take)) if max_take > 0 else 1
        chunk = tokens[:n]
        # Smart join: no space around CJK chars, space between non-CJK tokens
        if len(chunk) == 1:
            if chunk[0].strip() == "":
                tokens = tokens[n:]
                continue
            pieces.append(chunk[0])
        else:
            parts = [chunk[0]]
            for i in range(1, len(chunk)):
                prev_is_non_space = all(is_non_space_script_char(c) for c in chunk[i - 1])
                curr_is_non_space = all(is_non_space_script_char(c) for c in chunk[i])
                if prev_is_non_space or curr_is_non_space:
                    parts.append(chunk[i])
                else:
                    parts.append(" " + chunk[i])
            phrase = "".join(parts).strip()
            if phrase == "":
                tokens = tokens[n:]
                continue
            pieces.append(phrase)
        tokens = tokens[n:]
    return pieces


def tag_piece(piece, tag="*"):
    """Tag the piece with a specified tag."""
    return f"{tag}{piece}{tag}"


def rand_sample(lst, max_num, new=False, hit_ratio=1.0):
    """Randomly sample random num <max_num elements from the list."""
    if new:
        n = min(max_num, len(lst))
        n = min(n, int(hit_ratio * len(lst)))
        n = random.randint(0, n)
    else:
        n = random.randint(0, max_num)
        n = min(n, len(lst))
    # print(f"Sampling {n} pieces from {len(lst)}[{max_num}]")
    return random.sample(lst, n)

def read_words(file_path, N=None, tn_prob=1.0):
    """Read the top N lines from a file."""
    words = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if N is not None and i >= N:
                break
            word = line.split()[0]
            words.append(text_norm(word, tn_prob))
    return words

class PieceSampler:
    """Sample segments from the text using instance parameters."""

    def __init__(self, buffer_size=100000, bias_prob=1.0, hit_prob=0.5,  max_piece_len=1, new_sampling=False, common_word_file=None, common_word_num=None,
                 max_num=10, tag="*", tag_all=True, log_interval=None, max_hit_ratio=1.0, per_language_buffer=True):
        """Initialize the PieceSampler with configuration parameters.

        Args:
            buffer_size: Maximum size of the buffer for negative pieces
            max_len: Maximum words of text piece
            bias_prob: biasing prompt probability
            max_num: Maximum number of pieces to sample
            hit_prob: Probability of sampling from positive words
            per_language_buffer: If True, maintain a separate replay buffer
                per language so negative samples are language-specific.
                If False (default), a single shared buffer is used.
        """
        self._buffer_size = buffer_size
        self.per_language_buffer = per_language_buffer
        self._buffers = defaultdict(lambda: deque(maxlen=buffer_size))
        self.max_piece_len = max_piece_len
        self.bias_prob = bias_prob
        self.max_num = max_num
        self.hit_prob = hit_prob
        self.tag = tag
        self.tag_all = tag_all
        self.new_sampling = new_sampling
        self.log_interval = int(log_interval) if log_interval else None
        self.common_words = set(read_words(common_word_file, common_word_num) if common_word_file else [])
        self.idx = 0
        self.max_hit_ratio = max_hit_ratio

    def filter_commons(self, examples):
        """Filter out common words from the examples."""
        if not self.common_words:
            return examples
        new_examples = []
        for phrase in examples:
            if all(text_norm(wd) in self.common_words for wd in phrase.split()):
                continue
            new_examples.append(phrase)
        return new_examples

    def tag_pieces(self, pieces, specified=None):
        """Tag the pieces with a specified tag."""
        if specified is None:
            return [tag_piece(p, self.tag) for p in pieces]
        return [tag_piece(p, self.tag) if p in specified else p for p in pieces]


    def _sample(self, pieces, language=None):
        """Sample segments from the positive pieces."""
        #remove pure punctuation pieces from the candidates 
        pieces_nopunc = [p for p in pieces if not all(unicodedata.category(c).startswith('P') or c.isspace() for c in p)]
        lang_key = language if (self.per_language_buffer and language is not None) else "__default__"
        buffer = self._buffers[lang_key]
        buffer.extend(pieces_nopunc)
        if random.random() > self.bias_prob:
            return []
        num_pieces = random.randint(1, self.max_num)
        examples = []
        if random.random() <= self.hit_prob:
            examples = rand_sample(pieces_nopunc, num_pieces, new=self.new_sampling, hit_ratio=self.max_hit_ratio)
        examples += rand_sample(buffer, num_pieces - len(examples), new=self.new_sampling)
        if self.common_words:
            examples = [p for p in examples if text_norm(p) not in self.common_words]
        return self.filter_commons(examples)

    def sample(self, text, language=None):
        """Sample segments from the text using instance parameters.

        Args:
            text: The text to sample biasing pieces from.
            language: Optional language identifier. When provided, a
                language-specific replay buffer is used so that negative
                samples are drawn from the same language.
        """
        pieces = split_text(text, self.max_piece_len)
        examples = self._sample(pieces, language=language)
        examples = list(set(examples))
        random.shuffle(examples)
        # Tag the pieces with shared tags
        shared = set(examples) & set(pieces)
        pieces = self.tag_pieces(pieces, shared)
        if self.tag_all: # tag the input sample pieces as well
            examples = self.tag_pieces(examples)
        self.idx += 1
        prompt, trans = ", ".join(examples), " ".join(pieces)
        if self.log_interval is not None and self.idx % self.log_interval == 0:
            print(f"[{self.idx}] biasing  list: {prompt}")
            print(f"[{self.idx}] transcription: {trans}")
        
        return prompt, trans
#%%
if __name__ == "__main__":
    sample_utterances = [
        "Hello, how are you?",
        "This is a test sentence.",
        "I love programming in Python!",
        "The quick brown fox jumps over the lazy dog.",
        "Data science is an interdisciplinary field.",
        "Artificial intelligence is transforming many industries.",
        "Learning to code is a valuable skill in today's job market.",
        "The meeting is scheduled for tomorrow at 9 AM.",
        "She sold seashells by the seashore.",
        "Machine learning models require large amounts of data.",
        "Please remember to submit your assignment by Friday.",
        "The restaurant on Main Street serves delicious pasta.",
        "Climate change is affecting ecosystems worldwide.",
        "Thank you for your help with this project.",
        "The concert was canceled due to bad weather.",
        "Neural networks are inspired by the human brain.",
        "Remember to back up your files regularly.",
        "The new movie received excellent reviews from critics.",
        "Quantum computing has the potential to revolutionize technology.",
        "The library closes at 8 PM on weekdays.",
        "Self-driving cars use various sensors to navigate.",
        "Regular exercise is important for maintaining good health.",
        "The art exhibition features works from local artists.",
        "Natural language processing helps computers understand human language.",
        "Don't forget to water the plants while I'm away."
    ]

    # Create an instance of PieceSampler
    sampler = PieceSampler(buffer_size=100, max_piece_len=3, bias_prob=1, max_num=100, hit_prob=0.9,log_interval=2)
    # Sample pieces from the utterances
    for text in sample_utterances:
        examples, trans = sampler.sample(text, language="en-US")
        print(f"Original Trans: {text}")
        print(f"Updated  Trans: {trans}")
        print(f"Sampled examples: {examples}")
        print("Buffers:", {lang: len(buf) for lang, buf in sampler._buffers.items()})
        print("-" * 40)
    # # %%

# %%
