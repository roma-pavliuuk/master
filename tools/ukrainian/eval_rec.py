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

"""Evaluate a text recognition model with CER, WER and line accuracy.

The references are the RAW transcriptions of the label file (only Unicode NFC
and apostrophe normalization is applied to both sides), unlike training-time
evaluation where labels are first filtered through the character dictionary.

Each ``--label_file`` may be given as ``name=path`` to report separate results,
e.g. ``printed=.../test.txt handwritten=.../test.txt``. Never evaluate on data
that was used for training or model selection.

Examples:
    # official model through the public PaddleOCR API
    python tools/ukrainian/eval_rec.py --model_name eslav_PP-OCRv5_mobile_rec \\
        --label_file printed=train_data/uk_printed/test.txt

    # fine-tuned model exported by training (inference.json/pdiparams/yml)
    python tools/ukrainian/eval_rec.py --model_name eslav_PP-OCRv5_mobile_rec \\
        --model_dir output/uk_printed_rec_ppocrv5/best_accuracy/inference \\
        --label_file printed=train_data/uk_printed/test.txt

    # predictions produced elsewhere: "<image>\\t<text>" lines
    python tools/ukrainian/eval_rec.py --pred_file preds.txt --label_file test.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from ppocr.utils.ocr_error_rate import (
    ErrorRateAccumulator,
    normalize_for_eval,
)  # noqa: E402
from ppocr.utils.ukrainian import (  # noqa: E402
    APOSTROPHE_ASCII,
    APOSTROPHE_MODIFIER,
    APOSTROPHE_TYPOGRAPHIC,
    UK_CRITICAL_CHARS,
)

APOSTROPHE_MODES = {
    # U+02BC is indistinguishable from U+2019 in an image
    "canonical": {APOSTROPHE_MODIFIER: APOSTROPHE_TYPOGRAPHIC},
    "strict": {},
    # all apostrophe variants are considered equal
    "merge": {
        APOSTROPHE_MODIFIER: APOSTROPHE_ASCII,
        APOSTROPHE_TYPOGRAPHIC: APOSTROPHE_ASCII,
    },
}


def read_label_file(path, data_dir=None, delimiter="\t"):
    data_dir = os.path.dirname(path) if data_dir is None else data_dir
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split(delimiter)
            if len(parts) != 2:
                raise ValueError(
                    "{}:{}: expected '<image>\\t<text>'".format(path, line_no)
                )
            img, text = parts
            samples.append((img, os.path.join(data_dir, img), text))
    return samples


class CharRecall(object):
    """Recall of selected characters: share of reference chars kept by the alignment."""

    def __init__(self, chars):
        self.chars = chars
        self.total = collections.Counter()
        self.correct = collections.Counter()

    def update(self, ref, hyp):
        for tag, i1, i2, _, _ in Levenshtein.opcodes(ref, hyp).as_list():
            for ch in ref[i1:i2]:
                if ch in self.chars:
                    self.total[ch] += 1
                    if tag == "equal":
                        self.correct[ch] += 1

    def summary(self):
        return {
            ch: {"total": self.total[ch], "recall": self.correct[ch] / self.total[ch]}
            for ch in self.chars
            if self.total[ch]
        }


def evaluate(pairs, apostrophe_mode="canonical", ignore_case=False):
    """``pairs``: iterable of (reference, hypothesis). Returns metrics dict."""
    amap = APOSTROPHE_MODES[apostrophe_mode]
    acc = ErrorRateAccumulator()
    critical = CharRecall(UK_CRITICAL_CHARS + APOSTROPHE_ASCII + APOSTROPHE_TYPOGRAPHIC)
    errors = []
    for ref, hyp in pairs:
        ref_n = normalize_for_eval(ref, apostrophe_map=amap, ignore_case=ignore_case)
        hyp_n = normalize_for_eval(hyp, apostrophe_map=amap, ignore_case=ignore_case)
        acc.update(ref_n, hyp_n)
        critical.update(ref_n, hyp_n)
        if ref_n != hyp_n:
            errors.append((Levenshtein.distance(ref_n, hyp_n), ref_n, hyp_n))
    result = acc.summary()
    result["critical_char_recall"] = critical.summary()
    result["worst_errors"] = [
        {"distance": d, "reference": r, "hypothesis": h}
        for d, r, h in sorted(errors, key=lambda x: -x[0])[:20]
    ]
    return result


def predict_with_model(samples, args):
    from paddleocr import TextRecognition

    kwargs = {"model_name": args.model_name, "device": args.device}
    if args.model_dir:
        kwargs["model_dir"] = args.model_dir
    model = TextRecognition(**kwargs)
    preds = {}
    paths = [p for _, p, _ in samples]
    start = time.perf_counter()
    for i in range(0, len(paths), args.batch_size):
        batch = paths[i : i + args.batch_size]
        for path, res in zip(batch, model.predict(batch, batch_size=args.batch_size)):
            preds[path] = res["rec_text"]
    elapsed = time.perf_counter() - start
    return preds, elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--label_file", nargs="+", required=True, help="path or name=path"
    )
    parser.add_argument(
        "--data_dir", help="image root (default: directory of each label file)"
    )
    parser.add_argument("--model_name", default="eslav_PP-OCRv5_mobile_rec")
    parser.add_argument("--model_dir")
    parser.add_argument(
        "--pred_file", help="use existing predictions instead of a model"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--apostrophe_mode", choices=sorted(APOSTROPHE_MODES), default="canonical"
    )
    parser.add_argument("--ignore_case", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_dir", help="write report.json and predictions")
    args = parser.parse_args(argv)

    external = None
    if args.pred_file:
        external = {}
        with open(args.pred_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if line:
                    img, _, text = line.partition("\t")
                    external[img] = text

    report = {
        "model_name": None if external is not None else args.model_name,
        "model_dir": args.model_dir,
        "pred_file": args.pred_file,
        "apostrophe_mode": args.apostrophe_mode,
        "ignore_case": args.ignore_case,
        "subsets": {},
    }
    for spec in args.label_file:
        name, _, path = spec.rpartition("=")
        name = name or os.path.splitext(os.path.basename(path))[0]
        samples = read_label_file(path, args.data_dir)
        if args.max_samples:
            samples = samples[: args.max_samples]
        elapsed = None
        if external is not None:
            missing = [img for img, _, _ in samples if img not in external]
            if missing:
                sys.exit(
                    "{} predictions missing, e.g. {}".format(len(missing), missing[:3])
                )
            hyps = [external[img] for img, _, _ in samples]
        else:
            preds, elapsed = predict_with_model(samples, args)
            hyps = [preds[p] for _, p, _ in samples]
        result = evaluate(
            [(ref, hyp) for (_, _, ref), hyp in zip(samples, hyps)],
            args.apostrophe_mode,
            args.ignore_case,
        )
        result["label_file"] = path
        if elapsed is not None:
            result["seconds_total"] = elapsed
            result["ms_per_line"] = 1000.0 * elapsed / max(1, len(samples))
        report["subsets"][name] = result
        print(
            "[{}] lines={} CER={:.4f} WER={:.4f} line_acc={:.4f}{}".format(
                name,
                result["num_lines"],
                result["cer"],
                result["wer"],
                result["line_acc"],
                (
                    ""
                    if elapsed is None
                    else " ({:.1f} ms/line, {})".format(
                        result["ms_per_line"], args.device
                    )
                ),
            )
        )
        if result["num_lines"] < 100:
            print(
                "  WARNING: fewer than 100 lines, metrics are not statistically reliable"
            )
        crit = result["critical_char_recall"]
        if crit:
            print(
                "  critical chars recall: "
                + ", ".join(
                    "{}:{:.2f}({})".format(c, v["recall"], v["total"])
                    for c, v in crit.items()
                )
            )
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(
                os.path.join(args.output_dir, "pred_{}.txt".format(name)),
                "w",
                encoding="utf-8",
            ) as f:
                for (img, _, ref), hyp in zip(samples, hyps):
                    f.write("{}\t{}\t{}\n".format(img, hyp, ref))

    if args.output_dir:
        with open(
            os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("report: {}".format(os.path.join(args.output_dir, "report.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
