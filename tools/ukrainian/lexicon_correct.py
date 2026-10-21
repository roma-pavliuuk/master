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

"""Lexicon post-correction of recognized Ukrainian text.

A recognized word that is not in the lexicon is replaced by a lexicon word at
edit distance 1 (deletion, insertion, substitution, adjacent transposition)
when the choice is unambiguous: a single candidate, or the most frequent
candidate according to an optional frequency list. Short words
(< ``min_len`` letters), words with digits / Latin letters and words without
a candidate are left as they are; letter case is preserved.

Lexicon: one word form per line, e.g. ``/usr/share/dict/ukrainian`` (Debian /
Ubuntu package ``wukrainian``, ~1.5 M forms from the dict_uk project).
Frequency list: ``word<space>count`` lines, e.g. FrequencyWords ``uk_50k.txt``
(https://github.com/hermitdave/FrequencyWords, OpenSubtitles 2018).

    python tools/ukrainian/lexicon_correct.py --lexicon /usr/share/dict/ukrainian \\
        --freq train_data/lexicon/uk_50k.txt --pred_file pred_test.txt
    echo "Украіна" | python tools/ukrainian/lexicon_correct.py --lexicon ...

``--pred_file`` takes the ``img<TAB>hyp<TAB>ref`` files written by
``eval_rec.py --output_dir`` and prints CER / WER before and after correction.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from ppocr.utils.ukrainian import normalize_uk_text  # noqa: E402

LOWER_LETTERS = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя’"
WORD_RE = re.compile(r"(?<!\w)[А-ЩЬЮЯҐЄІЇа-щьюяґєії’'ʼ]+(?!\w)")
_HYPHEN_TAIL_RE = re.compile(r"\s*[-¬‐]\s*$")
_APOSTROPHES = str.maketrans({"'": "’", "ʼ": "’"})


def _key(word):
    return word.lower().translate(_APOSTROPHES)


def _edits1(word):
    out = set()
    for i in range(len(word) + 1):
        a, b = word[:i], word[i:]
        if b:
            out.add(a + b[1:])
            out.update(a + c + b[1:] for c in LOWER_LETTERS)
        if len(b) > 1:
            out.add(a + b[1] + b[0] + b[2:])
        out.update(a + c + b for c in LOWER_LETTERS)
    out.discard(word)
    return out


def _restore_case(src, dst):
    if len(src) > 1 and src.isupper():
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


class LexiconCorrector:
    def __init__(self, lexicon_paths, freq_path=None, min_len=4):
        self.min_len = min_len
        self.lexicon = set()
        for path in lexicon_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.lexicon.add(_key(normalize_uk_text(word)))
        self.freq = {}
        if freq_path:
            with open(freq_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        word = _key(normalize_uk_text(parts[0]))
                        if word in self.lexicon:
                            self.freq[word] = self.freq.get(word, 0) + int(parts[1])

    def correct_word(self, word):
        key = _key(word)
        if len(key) < self.min_len or key in self.lexicon:
            return word
        cands = [c for c in _edits1(key) if c in self.lexicon]
        if not cands:
            return word
        if len(cands) > 1:
            cands.sort(key=lambda c: (-self.freq.get(c, 0), c))
            if self.freq.get(cands[0], 0) <= self.freq.get(cands[1], 0):
                return word  # ambiguous
        return _restore_case(word, cands[0])

    def correct(self, text):
        """Correct every word of one text line. A fragment of a word hyphenated
        at the end of the line ("загро-") is kept as is."""

        def replace(m):
            if _HYPHEN_TAIL_RE.match(text, m.end()):
                return m.group(0)
            return self.correct_word(m.group(0))

        return WORD_RE.sub(replace, text)


def add_lexicon_args(parser):
    parser.add_argument(
        "--lexicon",
        nargs="*",
        help="word list(s) for post-correction, e.g. /usr/share/dict/ukrainian",
    )
    parser.add_argument("--lexicon_freq", help="'word count' frequency list")
    parser.add_argument("--lexicon_min_len", type=int, default=4)


def corrector_from_args(args):
    if not args.lexicon:
        return None
    return LexiconCorrector(args.lexicon, args.lexicon_freq, args.lexicon_min_len)


def main(argv=None):
    from ppocr.utils.ocr_error_rate import ErrorRateAccumulator, normalize_for_eval

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_lexicon_args(parser)
    parser.add_argument("--pred_file", help="eval_rec.py predictions (img, hyp, ref)")
    parser.add_argument("--output", help="write corrected predictions here")
    args = parser.parse_args(argv)
    if not args.lexicon:
        parser.error("--lexicon is required")
    corrector = corrector_from_args(args)

    if not args.pred_file:
        for line in sys.stdin:
            print(corrector.correct(line.rstrip("\n")))
        return 0
    before, after = ErrorRateAccumulator(), ErrorRateAccumulator()
    rows = []
    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            img, hyp, ref = line.rstrip("\n").split("\t")
            fixed = corrector.correct(hyp)
            before.update(normalize_for_eval(ref), normalize_for_eval(hyp))
            after.update(normalize_for_eval(ref), normalize_for_eval(fixed))
            rows.append((img, fixed, ref))
    b, a = before.summary(), after.summary()
    print(
        "lines={} CER {:.4f} -> {:.4f} | WER {:.4f} -> {:.4f}".format(
            len(rows), b["cer"], a["cer"], b["wer"], a["wer"]
        )
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for row in rows:
                f.write("\t".join(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
