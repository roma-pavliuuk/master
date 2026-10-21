# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Character / word error rates for OCR evaluation.

    CER = (S + D + I) / N

where S, D and I are the numbers of substituted, deleted and inserted
characters of the minimal (Levenshtein) alignment between the reference and
the hypothesis, and N is the number of characters of the reference. WER is the
same quantity computed on word sequences (words are separated by whitespace).

Corpus level rates are ``sum(S + D + I) / sum(N)`` over all lines, not a mean
of per-line rates. Rates can exceed 1.0 when the hypothesis has many
insertions.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from rapidfuzz.distance import Levenshtein


@dataclass
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_len: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.ref_len == 0:
            # No reference tokens: any insertion is an error, otherwise perfect.
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.ref_len

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.ref_len + other.ref_len,
        )


def edit_counts(reference: Sequence, hypothesis: Sequence) -> EditCounts:
    """Count S, D, I of the minimal alignment turning reference into hypothesis.

    Works on strings (characters) and on lists of tokens (e.g. words).
    """
    counts = EditCounts(ref_len=len(reference))
    for tag, _, _ in Levenshtein.editops(reference, hypothesis).as_list():
        if tag == "replace":
            counts.substitutions += 1
        elif tag == "delete":
            counts.deletions += 1
        elif tag == "insert":
            counts.insertions += 1
    return counts


def normalize_for_eval(
    text: str,
    unicode_nfc: bool = True,
    collapse_spaces: bool = True,
    ignore_case: bool = False,
    apostrophe_map: Optional[Dict[str, str]] = None,
) -> str:
    """Normalization applied identically to references and hypotheses."""
    if unicode_nfc:
        text = unicodedata.normalize("NFC", text)
    if apostrophe_map:
        for src, dst in apostrophe_map.items():
            text = text.replace(src, dst)
    if collapse_spaces:
        text = " ".join(text.split())
    if ignore_case:
        text = text.lower()
    return text


def char_edit_counts(reference: str, hypothesis: str) -> EditCounts:
    return edit_counts(reference, hypothesis)


def word_edit_counts(reference: str, hypothesis: str) -> EditCounts:
    return edit_counts(reference.split(), hypothesis.split())


def cer(reference: str, hypothesis: str) -> float:
    return char_edit_counts(reference, hypothesis).rate


def wer(reference: str, hypothesis: str) -> float:
    return word_edit_counts(reference, hypothesis).rate


@dataclass
class ErrorRateAccumulator:
    """Accumulates corpus level CER, WER and line accuracy."""

    chars: EditCounts = field(default_factory=EditCounts)
    words: EditCounts = field(default_factory=EditCounts)
    num_lines: int = 0
    num_correct_lines: int = 0

    def update(self, reference: str, hypothesis: str) -> None:
        self.chars = self.chars + char_edit_counts(reference, hypothesis)
        self.words = self.words + word_edit_counts(reference, hypothesis)
        self.num_lines += 1
        self.num_correct_lines += int(reference == hypothesis)

    def summary(self) -> Dict[str, float]:
        return {
            "cer": self.chars.rate if self.num_lines else 0.0,
            "wer": self.words.rate if self.num_lines else 0.0,
            "line_acc": (
                self.num_correct_lines / self.num_lines if self.num_lines else 0.0
            ),
            "num_lines": self.num_lines,
            "num_ref_chars": self.chars.ref_len,
            "num_ref_words": self.words.ref_len,
            "char_substitutions": self.chars.substitutions,
            "char_deletions": self.chars.deletions,
            "char_insertions": self.chars.insertions,
            "word_substitutions": self.words.substitutions,
            "word_deletions": self.words.deletions,
            "word_insertions": self.words.insertions,
        }
