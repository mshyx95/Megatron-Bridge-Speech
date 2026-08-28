import difflib
import sys
import re
import copy
import torch

ALIGNER_ROOT = "/mnt/tsstd01wus2/users/xioxiao/tools/word_timing_mahmoud/wordaligner"
sys.path.append(ALIGNER_ROOT)
from word_aligner import WordTimingAlignerChenoneLex2Lts
from ciphone_generator_onnx import CiPhoneGeneratorOnnx
from dataloader import WavAudioDataset


class ForcedAlignment:
    """Global diarization for Phi model."""
    def __init__(self):
        self.generator = CiPhoneGeneratorOnnx(ALIGNER_ROOT + "/wt.quantized.onnx", use_cuda=False)
        self.aligner = WordTimingAlignerChenoneLex2Lts()
        self.fbank_extractor = WavAudioDataset("")._extract_fbank

    def handle_unmatched_words(self, word_list, aligned_word_list):
        """Handle unmatched words between original word list and aligned word list.
        The difference usually comes from the text normalization performed in alignement process.
        Inputs:
            word_list: list of words from original transcription
            aligned_word_list: list of [word, start_time, end_time] from alignment
        Outputs:
            word_list_with_timing: list of [start_time, end_time, word]
        """
        # map aligned words (lexical format) to original words, which may be in display format.
        word_list_ali = [re.sub(r"[^\w\s]", "", w[0].lower()) for w in aligned_word_list]
        word_list_ori = [re.sub(r"[^\w\s]", "", w.lower()) for w in word_list]
        matcher = difflib.SequenceMatcher(None, word_list_ali, word_list_ori)

        word_list_with_timing = copy.deepcopy(word_list)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":   # matched words, copy timing info
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    start_time = round(aligned_word_list[i][1], 2)
                    end_time = round(aligned_word_list[i][2], 2)
                    word_list_with_timing[j] = [start_time, end_time, word_list_with_timing[j]]
            else:    # unmatched words, need to estimate timing info
                if j2 == j1:
                    continue  # No display words to map, skip
                elif i1 == i2:  # no alignment words
                    if i1 == 0:  # inserted original word at the beginning
                        start_time = 0.0
                        total_duration = aligned_word_list[i1][1]
                    elif i2 == len(aligned_word_list):  # inserted original word at the end
                        start_time = aligned_word_list[i1][2]
                        total_duration = (j2 - j1) * 0.3  # assume 0.3s per word
                    else:  # inserted original word in the middle
                        start_time = aligned_word_list[i1 - 1][2]
                        total_duration = aligned_word_list[i1][1] - start_time
                else:
                    start_time = aligned_word_list[i1][1]
                    total_duration = aligned_word_list[i2 - 1][2] - aligned_word_list[i1][1]
                avg_duration = total_duration / (j2 - j1)

                for j in range(j1, j2):
                    word_start_time = round(start_time + avg_duration * (j - j1), 2)
                    word_end_time = round(word_start_time + avg_duration, 2)
                    word_list_with_timing[j] = [word_start_time, word_end_time, word_list_with_timing[j]]

        return word_list_with_timing

    def __call__(self, wav, word_list):
        """Perform forced alignment to get word timing.
        Inputs:
            wav: 1D numpy array, audio waveform
            word_list: list of words
        Outputs:
            word_list_with_timing: list of [start_time, end_time, word]
        """
        feature = self.fbank_extractor(wav, 16000)  # T x D
        feature = torch.tensor(feature).unsqueeze(0)  # 1 x T x D
        if feature.shape[1] < 32:
            # pad to at least 32 frames required by ciphone generator
            pad_size = 32 - feature.shape[1]
            feature = torch.nn.functional.pad(feature, (0, 0, 0, pad_size), "constant", 0.0)
        ciphone_probs = self.generator.generate(feature, [feature.shape[1]])

        sentence_str = " ".join(word_list)
        sentence_str = sentence_str.replace(".", "")    # LTS module in alignment cannot handle period symbol
        ci_length = torch.tensor([ciphone_probs.shape[1]], dtype=torch.int32, device=ciphone_probs.device)
        ali = self.aligner.get_alignment([[sentence_str]], ciphone_probs, ci_length, force_cuda=False)
        if not ali[0] and len(word_list) > 0:
            # alignment failed, evenly distribute timing
            total_duration = len(wav) / 16000.0
            avg_duration = total_duration / len(word_list)
            word_list_with_timing = []
            for i, word in enumerate(word_list):
                start_time = round(i * avg_duration, 2)
                end_time = round(start_time + avg_duration, 2)
                word_list_with_timing.append([start_time, end_time, word])
            return word_list_with_timing

        word_list_with_timing = self.handle_unmatched_words(word_list, ali[0])  # generate word timing

        return word_list_with_timing
