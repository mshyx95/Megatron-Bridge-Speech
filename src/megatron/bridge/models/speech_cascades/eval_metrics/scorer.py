# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Word-error-rate scorer vendored to match the production ``HFWerScorer``.

Reproduces ``phyagi/eval/utils/score_utils.py::HFWerScorer`` from the speech
training branch so that Open-ASR-Leaderboard numbers are computed with the exact
same text normalization and alignment metric as production:

* English (``hfwer_en``) uses the custom :class:`EnglishTextNormalizer` (acronym,
  name, and compound normalization on top of the Whisper normalizer) and, when
  available, ``kaldialign.batch_error_rate(merge_compounds=True)`` — the official
  Open ASR Leaderboard metric. It falls back to word edit distance only if
  ``kaldialign`` is not installed.
* Only the prediction is normalized in :meth:`add_string`; the reference is
  expected to already be normalized by the caller (mirroring production, which
  normalizes the reference once at data-load time).
"""

from .normalizer import EnglishTextNormalizer


class HFWerScorer:
    """Accumulate reference/hypothesis pairs and report corpus WER (percent)."""

    def __init__(self, sub_task: str = "hfwer_en", use_normalizer: bool = True, use_kaldialign: bool = True):
        self.sub_task = sub_task
        self.language = sub_task.split("_")[-1].lower()
        self.use_normalizer = use_normalizer

        if sub_task in ("hfwer_en", "hfwer_EN"):
            self.normalizer = EnglishTextNormalizer()
        else:
            # Multilingual/Chinese normalizers pull in num2words/datasets; import lazily.
            from .normalizer import BasicMultilingualTextNormalizer

            self.normalizer = BasicMultilingualTextNormalizer(remove_diacritics=False)

        try:
            import editdistance as ed
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Please install editdistance to use WER scorer") from exc
        self.ed = ed

        # kaldialign matches the official Open ASR Leaderboard metric for English.
        self._use_kaldialign = False
        if self.language == "en" and use_kaldialign:
            try:
                from kaldialign import batch_error_rate

                self._batch_error_rate = batch_error_rate
                self._use_kaldialign = True
            except ImportError:
                self._use_kaldialign = False

        self.reset()

    def reset(self) -> None:
        self.distance = 0
        self.ref_length = 0
        self._all_refs = []
        self._all_preds = []

    def add_string(self, ref: str, pred: str) -> None:
        """Add a (reference, hypothesis) pair. Only the hypothesis is normalized."""
        if self.use_normalizer:
            pred = self.normalizer(pred)

        if self._use_kaldialign:
            self._all_refs.append(tuple(ref.split()))
            self._all_preds.append(tuple(pred.split()))
            self.ref_length += len(ref.split())
        else:
            ref_items, pred_items = ref.split(), pred.split()
            self.distance += self.ed.eval(ref_items, pred_items)
            self.ref_length += len(ref_items)

    def score(self) -> float:
        """Corpus WER in percent."""
        if self._use_kaldialign and self._all_refs:
            r = self._batch_error_rate(self._all_refs, self._all_preds, merge_compounds=True)
            return 100.0 * r["err_rate"]
        return 100.0 * self.distance / self.ref_length if self.ref_length > 0 else 0.0

    def normalize_reference(self, ref: str) -> str:
        """Normalize a reference string with the same normalizer (parity helper)."""
        return self.normalizer(ref) if self.use_normalizer else ref
