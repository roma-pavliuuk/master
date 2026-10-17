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

Input is the prediction file of ``eval_rec.py --output_dir`` run on the silver
label file (``img<TAB>prediction<TAB>silver``).

    python tools/ukrainian/filter_silver.py --pred_file output/eval/silver_v7/pred_silver.txt \\
        --max_cer 0.1 --min_chars 3 --output train_data/uk_handwritten_silver/silver_agree.txt
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from ppocr.utils.ocr_error_rate import cer, normalize_for_eval  # noqa: E402


def select(rows, max_cer, min_chars=3):
    """rows: (img, prediction, silver). Yields (img, silver, cer)."""
    for img, hyp, ref in rows:
        ref_n = normalize_for_eval(ref)
        if len(ref_n.replace(" ", "")) < min_chars:
            continue
        score = cer(ref_n, normalize_for_eval(hyp))
        if score <= max_cer:
            yield img, ref, score


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--max_cer", type=float, default=0.1)
    parser.add_argument("--min_chars", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    with open(args.pred_file, "r", encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    kept = list(select(rows, args.max_cer, args.min_chars))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for img, text, _ in kept:
            f.write("{}\t{}\n".format(img, text))
    hist = collections.Counter(min(int(s * 20), 4) for _, _, s in kept)
    print(
        "kept {} of {} lines ({:.1%}) with CER <= {}; CER buckets (5 %): {}".format(
            len(kept),
            len(rows),
            len(kept) / max(1, len(rows)),
            args.max_cer,
            dict(sorted(hist.items())),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
