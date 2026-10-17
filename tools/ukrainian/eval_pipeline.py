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

"""End-to-end evaluation of the OCR pipeline (detection + recognition) on pages.

Ground truth is a PaddleOCR detection label file
(``<image>\\t[{"transcription": str, "points": [[x, y], ...]}]``); regions with
transcription ``###`` are ignored. Reported metrics:

* detection precision / recall / hmean (one-to-one matching, IoU >= --iou)
* line CER / WER: matched pairs are compared, missed GT lines count as
  deletions, unmatched predictions as insertions
* page CER / WER: all lines of a page sorted in reading order (rows top to
  bottom after skew correction, left to right inside a row) and joined,
  independent of the line matching

Example:
    python tools/ukrainian/eval_pipeline.py --det_label output/uk_demo/uk_printed_document_det.txt \\
        --data_dir output/uk_demo --mode printed
    python tools/ukrainian/eval_pipeline.py --det_label train_data/rukopys_raw/test/det_test.txt \\
        --data_dir train_data/rukopys_raw/test --mode handwritten \\
        --rec_model_dir output/uk_handwritten_rec_ppocrv5/best_accuracy/inference
"""

from __future__ import annotations

import argparse
import json
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))
sys.path.insert(0, os.path.join(__dir__, "examples"))

from ppocr.utils.ocr_error_rate import (
    ErrorRateAccumulator,
    normalize_for_eval,
)  # noqa: E402
from ppocr.utils.ukrainian import (
    APOSTROPHE_MODIFIER,
    APOSTROPHE_TYPOGRAPHIC,
)  # noqa: E402

APOSTROPHES = {APOSTROPHE_MODIFIER: APOSTROPHE_TYPOGRAPHIC}


def poly_to_box(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def page_text(items):
    """Join (poly, box, text) items in reading order (see uk_ocr_common.group_rows)."""
    from uk_ocr_common import group_rows

    lines = group_rows(
        [{"poly": poly, "box": box, "text": text} for poly, box, text in items]
    )
    return " ".join(ln["text"] for ln in lines)


def match_page(gt, pred, iou_thr):
    """Greedy one-to-one matching by IoU. Returns (pairs, missed_gt, extra_pred)."""
    candidates = sorted(
        (
            (iou(g[0], p[0]), gi, pi)
            for gi, g in enumerate(gt)
            for pi, p in enumerate(pred)
        ),
        reverse=True,
    )
    used_g, used_p, pairs = set(), set(), []
    for score, gi, pi in candidates:
        if score < iou_thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi))
    missed = [gi for gi in range(len(gt)) if gi not in used_g]
    extra = [pi for pi in range(len(pred)) if pi not in used_p]
    return pairs, missed, extra


def evaluate_page(gt_regions, pred_items, iou_thr, line_acc, page_acc, det_counts):
    gt, ignored = [], []
    for region in gt_regions:
        box = poly_to_box(region["points"])
        text = region.get("transcription", "")
        item = (
            box,
            normalize_for_eval(text, apostrophe_map=APOSTROPHES),
            region["points"],
        )
        (ignored if text in ("###", "*", "") else gt).append(item)
    pred = [
        (box, normalize_for_eval(text, apostrophe_map=APOSTROPHES), poly)
        for poly, box, text in pred_items
    ]
    # drop predictions lying on ignored regions
    pred = [
        p
        for p in pred
        if not any(
            iou(p[0], ig[0]) >= iou_thr or _inside(p[0], ig[0]) for ig in ignored
        )
    ]

    pairs, missed, extra = match_page(gt, pred, iou_thr)
    det_counts["tp"] += len(pairs)
    det_counts["gt"] += len(gt)
    det_counts["pred"] += len(pred)
    for gi, pi in pairs:
        line_acc.update(gt[gi][1], pred[pi][1])
    for gi in missed:
        line_acc.update(gt[gi][1], "")
    for pi in extra:
        line_acc.update("", pred[pi][1])
    page_acc.update(
        page_text([(poly, box, t) for box, t, poly in gt]),
        page_text([(poly, box, t) for box, t, poly in pred]),
    )


def _inside(a, b, thr=0.8):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return area > 0 and ix * iy / area >= thr


def main(argv=None):
    from uk_ocr_common import build_uk_ocr, result_to_lines

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--det_label", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mode", choices=["printed", "handwritten"], default="printed")
    parser.add_argument("--rec_model_dir")
    parser.add_argument("--det_model_dir")
    parser.add_argument(
        "--det_model_name",
        help="detector architecture of --det_model_dir, e.g. PP-OCRv5_mobile_det "
        "(default PP-OCRv5_server_det)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--det_limit_side_len",
        type=int,
        help="detector input size limit (handwritten default: 960, longest side)",
    )
    parser.add_argument("--det_limit_type", choices=["max", "min"], default="max")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max_pages", type=int, default=0)
    parser.add_argument("--output", help="JSON report path")
    args = parser.parse_args(argv)

    with open(args.det_label, "r", encoding="utf-8") as f:
        pages = [line.rstrip("\n").split("\t", 1) for line in f if line.strip()]
    if args.max_pages:
        pages = pages[: args.max_pages]

    extra = {}
    if args.det_limit_side_len:
        extra = dict(
            text_det_limit_side_len=args.det_limit_side_len,
            text_det_limit_type=args.det_limit_type,
        )
    ocr = build_uk_ocr(
        args.mode,
        rec_model_dir=args.rec_model_dir,
        det_model_dir=args.det_model_dir,
        det_model_name=args.det_model_name,
        device=args.device,
        **extra,
    )
    line_acc, page_acc = ErrorRateAccumulator(), ErrorRateAccumulator()
    det_counts = {"tp": 0, "gt": 0, "pred": 0}
    for img_rel, label in pages:
        res = ocr.predict(os.path.join(args.data_dir, img_rel))[0]
        pred_items = [
            (ln["poly"], ln["box"], ln["text"]) for ln in result_to_lines(res)
        ]
        evaluate_page(
            json.loads(label), pred_items, args.iou, line_acc, page_acc, det_counts
        )

    precision = det_counts["tp"] / det_counts["pred"] if det_counts["pred"] else 0.0
    recall = det_counts["tp"] / det_counts["gt"] if det_counts["gt"] else 0.0
    hmean = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    line, page = line_acc.summary(), page_acc.summary()
    report = {
        "mode": args.mode,
        "rec_model_dir": args.rec_model_dir,
        "det_model_dir": args.det_model_dir,
        "det_model_name": args.det_model_name,
        "det_limit_side_len": args.det_limit_side_len,
        "num_pages": len(pages),
        "detection": {
            "precision": precision,
            "recall": recall,
            "hmean": hmean,
            **det_counts,
        },
        "line": line,
        "page": page,
    }
    print(
        "pages={} det P={:.3f} R={:.3f} H={:.3f} | line CER={:.4f} WER={:.4f} | page CER={:.4f} WER={:.4f}".format(
            len(pages),
            precision,
            recall,
            hmean,
            line["cer"],
            line["wer"],
            page["cer"],
            page["wer"],
        )
    )
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
