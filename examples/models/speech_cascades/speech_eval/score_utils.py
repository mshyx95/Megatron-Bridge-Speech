# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from eval_metrics.normalizer import BasicTextNormalizer, EnglishTextNormalizer

from .normalizer import MultilingualNormalizer, normalize_compound_pairs


class BaseScorer:
    def __init__(self, use_normalizer=True):
        self.ref = []
        self.pred = []

        self.use_normalizer = use_normalizer
        if use_normalizer:
            self.normalizer = BasicTextNormalizer()


class HFWerScorer(BaseScorer):
    def __init__(self, sub_task, use_normalizer=True, use_kaldialign=True):
        super().__init__(use_normalizer)

        if sub_task == "hfwer_en" or sub_task == "hfwer_EN":
            self.normalizer = EnglishTextNormalizer()
        elif sub_task in ["hfwer_zh", "hfwer_ZH", "hfwer_zh-CN"]:
            raise ValueError("Chinese ASR is outside the selected OpenASR subsets")
        else:
            self.normalizer = MultilingualNormalizer(remove_diacritics=False)

        self.sub_task = sub_task
        self.language = sub_task.split("_")[-1].lower()

        try:
            import editdistance as ed
        except ImportError:
            raise ImportError("Please install editdistance to use WER scorer")

        self.ed = ed

        # Use kaldialign for English to match official Open ASR Leaderboard metric
        # Can be disabled for long-form tasks where batch scoring is too slow
        self._use_kaldialign = False
        if self.language == "en" and use_kaldialign:
            try:
                from kaldialign import batch_error_rate
            except ImportError:
                raise ImportError(
                    "Please install kaldialign (`pip install kaldialign`) for English WER scoring. "
                    "This is required to match the official Open ASR Leaderboard metric."
                )
            self._batch_error_rate = batch_error_rate
            self._use_kaldialign = True

        self.reset()

    def reset(self):
        self.distance = 0
        self.ref_length = 0
        self._all_refs = []
        self._all_preds = []

    def add_string(self, ref, pred):
        if self.language == "en":
            pred = self.normalizer(pred) if self.use_normalizer else pred
        else:
            pred = self.normalizer(pred, lang=self.language) if self.use_normalizer else pred

        if isinstance(self.normalizer, MultilingualNormalizer):
            refs, preds = normalize_compound_pairs([ref], [pred])
            ref, pred = refs[0], preds[0]

        if self._use_kaldialign:
            # Accumulate for batch scoring with kaldialign
            self._all_refs.append(tuple(ref.split()))
            self._all_preds.append(tuple(pred.split()))
            self.ref_length += len(ref.split())
        else:
            ref_items, pred_items = ref.split(), pred.split()
            self.distance += self.ed.eval(ref_items, pred_items)
            self.ref_length += len(ref_items)

    def result_string(self):
        return f"WER: {self.score():.2f}, #REF: {self.ref_length}"

    def score(self):
        if self._use_kaldialign and self._all_refs:
            r = self._batch_error_rate(self._all_refs, self._all_preds, merge_compounds=True)
            return 100.0 * r["err_rate"]
        return 100.0 * self.distance / self.ref_length if self.ref_length > 0 else 0


class SacrebleuScorer(BaseScorer):
    def __init__(self, sub_task, use_normalizer):
        super(SacrebleuScorer, self).__init__(use_normalizer)
        import sacrebleu

        self.sacrebleu = sacrebleu
        lang = sub_task.split("_")[-1][:2].lower()
        if lang == "zh":
            self.tokenize = "zh"
        elif lang == "ja":
            self.tokenize = "ja-mecab"
        else:
            self.tokenize = "13a"

    def reset(self):
        self.ref = []
        self.pred = []

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.pred.append(pred)

    def _score(self, order=4):
        if order != 4:
            raise NotImplementedError
        self.sacrebleu.corpus_bleu(self.pred, [self.ref], tokenize=self.tokenize)
        bleu = self.sacrebleu.BLEU(tokenize=self.tokenize)
        chrf = self.sacrebleu.CHRF()
        bleu_score = bleu.corpus_score(self.pred, [self.ref])
        chrf_score = round(chrf.corpus_score(self.pred, [self.ref]).score, 2)
        return bleu_score, chrf_score

    def score(self, order=4):
        return self._score(order)

    def result_string(self, order=4):
        bleu_score, chrf_score = self._score(order)
        results = f"BLEU: {bleu_score}, chrF: {chrf_score}"
        return results
