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

"""Select trustworthy lines of an automatically annotated split (RUKOPYS silver).

The silver transcriptions were produced automatically and are often wrong. A
line is kept only when an independent recognizer (our fine-tuned model) agrees
with the silver label: CER(silver, prediction) <= --max_cer. The silver text is
kept as the label (the prediction only votes), so the recognizer's own errors
are not copied into the training set unless the silver label has them too.

With ``--lexicon`` a second, looser band is accepted: lines with
CER <= --max_cer_lexicon whose silver text is plausible Ukrainian (at least
--min_lexicon_rate of its words of >= 3 letters are in the word list). On
RUKOPYS silver most disagreements in that band are errors of our recognizer,
not of the silver label (docs/ukrainian_ocr.md).

Input is the prediction file of ``eval_rec.py --output_dir`` run on the silver
label file (``img<TAB>prediction<TAB>silver``).

    python tools/ukrainian/filter_silver.py --pred_file output/eval/silver_v7/pred_silver.txt \\
        --max_cer 0.15 --lexicon /usr/share/dict/ukrainian --max_cer_lexicon 0.35 \\
        --output train_data/uk_handwritten_silver/silver_agree.txt
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from ppocr.utils.ocr_error_rate import cer, normalize_for_eval  # noqa: E402

sys.path.insert(0, __dir__)
from lexicon_correct import WORD_RE, _key  # noqa: E402


def lexicon_rate(text, lexicon, min_len=3):
    words = [w for w in WORD_RE.findall(text) if len(w) >= min_len]
    if not words:
        return None
    return sum(_key(w) in lexicon for w in words) / len(words)


def select(
    rows,
    max_cer,
    min_chars=3,
    lexicon=None,
    max_cer_lexicon=0.35,
    min_lexicon_rate=0.9,
):
    """rows: (img, prediction, silver). Yields (img, silver, cer)."""
    for img, hyp, ref in rows:
        ref_n = normalize_for_eval(ref)
        if len(ref_n.replace(" ", "")) < min_chars:
            continue
        score = cer(ref_n, normalize_for_eval(hyp))
        if score <= max_cer:
            yield img, ref, score
        elif lexicon is not None and score <= max_cer_lexicon:
            rate = lexicon_rate(ref, lexicon)
            if rate is not None and rate >= min_lexicon_rate:
                yield img, ref, score


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--max_cer", type=float, default=0.1)
    parser.add_argument("--min_chars", type=int, default=3)
    parser.add_argument("--max_text_length", type=int, default=75)
    parser.add_argument("--lexicon", nargs="*", help="word list(s)")
    parser.add_argument("--max_cer_lexicon", type=float, default=0.35)
    parser.add_argument("--min_lexicon_rate", type=float, default=0.9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    lexicon = None
    if args.lexicon:
        lexicon = set()
        for path in args.lexicon:
            with open(path, "r", encoding="utf-8") as f:
                lexicon.update(_key(w.strip()) for w in f if w.strip())

    with open(args.pred_file, "r", encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    if args.max_text_length:
        rows = [r for r in rows if len(r[2]) <= args.max_text_length]
    kept = list(
        select(
            rows,
            args.max_cer,
            args.min_chars,
            lexicon,
            args.max_cer_lexicon,
            args.min_lexicon_rate,
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for img, text, _ in kept:
            f.write("{}\t{}\n".format(img, text))
    hist = collections.Counter(min(int(s * 20), 7) for _, _, s in kept)
    print(
        "kept {} of {} lines ({:.1%}); CER buckets (5 %): {}".format(
            len(kept),
            len(rows),
            len(kept) / max(1, len(rows)),
            dict(sorted(hist.items())),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
