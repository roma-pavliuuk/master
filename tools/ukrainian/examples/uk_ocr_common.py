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

"""Shared helpers for the Ukrainian OCR examples (public PaddleOCR API only)."""

import logging
import math
import os

# Exported fine-tuned models keep this registered PaddleX model name, their
# dictionary is read from inference.yml (see configs/rec/PP-OCRv5/multi_language/uk_*).
UK_REC_MODEL_NAME = "eslav_PP-OCRv5_mobile_rec"
UK_DET_MODEL_NAME = "PP-OCRv5_server_det"
# Handwritten pages are 3000-4000 px phone photos / scans. Detecting text lines
# on a page downscaled to 960 px (longest side) finds far more lines than the
# pipeline default (full resolution): on 40 RUKOPYS test pages page CER
# 34.3 % -> 26.6 % with the official detector (docs/ukrainian_ocr.md).
# DB post-processing thresholds were tuned on 35 validation pages (unclip 1.8
# expands boxes over ascenders / descenders): on all 385 test pages page CER
# 22.9 % -> 22.2 % with the fine-tuned detector + v7 recognizer.
HANDWRITTEN_DET_LIMIT = {
    "text_det_limit_side_len": 960,
    "text_det_limit_type": "max",
    "text_det_thresh": 0.4,
    "text_det_box_thresh": 0.5,
    "text_det_unclip_ratio": 1.8,
}

logger = logging.getLogger("uk_ocr")


def _check_model_dir(path, what):
    if path is None:
        return
    if not os.path.isfile(os.path.join(path, "inference.yml")):
        raise FileNotFoundError(
            "{} model dir {!r} has no inference.yml; pass the directory written by "
            "tools/export_model.py or <save_model_dir>/best_accuracy/inference".format(
                what, path
            )
        )


def build_uk_ocr(
    mode="printed",
    rec_model_dir=None,
    det_model_dir=None,
    device="cpu",
    det_model_name=None,
    **kwargs,
):
    """Create a PaddleOCR pipeline for printed or handwritten Ukrainian text.

    * printed, no model dirs: ``PaddleOCR(lang="uk")`` -> PP-OCRv5_server_det +
      eslav_PP-OCRv5_mobile_rec (official weights, Russian/Belarusian/Ukrainian)
    * printed / handwritten with ``rec_model_dir``: a fine-tuned Ukrainian model
    * handwritten without ``rec_model_dir`` falls back to the official model,
      which is NOT trained for handwriting (see docs/ukrainian_ocr.md)
    * handwritten mode detects lines on the page downscaled to 960 px with
      tuned DB thresholds (``HANDWRITTEN_DET_LIMIT``); pass
      ``text_det_limit_side_len``, ``text_det_thresh``, ... to override
    """
    from paddleocr import PaddleOCR

    if mode not in ("printed", "handwritten"):
        raise ValueError(
            "mode must be 'printed' or 'handwritten', got {!r}".format(mode)
        )
    _check_model_dir(rec_model_dir, "recognition")
    _check_model_dir(det_model_dir, "detection")
    params = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
    )
    if mode == "handwritten":
        params.update(HANDWRITTEN_DET_LIMIT)
    params.update(kwargs)
    if rec_model_dir is None and det_model_dir is None and det_model_name is None:
        if mode == "handwritten":
            logger.warning(
                "no fine-tuned handwritten model given: using the official printed-text "
                "model, expect very poor handwriting accuracy"
            )
        return PaddleOCR(lang="uk", **params)
    params.update(
        text_detection_model_name=det_model_name or UK_DET_MODEL_NAME,
        text_detection_model_dir=det_model_dir,
        text_recognition_model_name=UK_REC_MODEL_NAME,
        text_recognition_model_dir=rec_model_dir,
    )
    return PaddleOCR(**params)


def estimate_skew(polys):
    """Median angle (radians) of the top edges of wide text regions."""
    angles = []
    for poly in polys:
        (x0, y0), (x1, y1) = poly[0], poly[1]
        width = math.hypot(x1 - x0, y1 - y0)
        height = math.hypot(poly[3][0] - x0, poly[3][1] - y0)
        if width > 2 * height > 0:
            angles.append(math.atan2(y1 - y0, x1 - x0))
    return sorted(angles)[len(angles) // 2] if angles else 0.0


def group_rows(lines):
    """Assign ``row`` indices: regions whose deskewed vertical centers are
    closer than half a line height belong to one row (e.g. table cells)."""
    if not lines:
        return lines
    angle = estimate_skew([ln["poly"] for ln in lines])
    sin_a, cos_a = math.sin(angle), math.cos(angle)
    for ln in lines:
        cx = sum(p[0] for p in ln["poly"]) / len(ln["poly"])
        cy = sum(p[1] for p in ln["poly"]) / len(ln["poly"])
        ln["_y"] = -sin_a * cx + cos_a * cy  # center in deskewed coordinates
        ln["_x"] = cos_a * cx + sin_a * cy
    heights = sorted(ln["box"][3] - ln["box"][1] for ln in lines)
    tol = 0.5 * heights[len(heights) // 2]
    lines.sort(key=lambda ln: ln["_y"])
    row, row_y, count = 0, lines[0]["_y"], 0
    for ln in lines:
        if abs(ln["_y"] - row_y) > tol:
            row, count = row + 1, 0
        ln["row"] = row
        count += 1
        row_y = ln["_y"] if count == 1 else row_y + (ln["_y"] - row_y) / count
    lines.sort(key=lambda ln: (ln["row"], ln["_x"]))
    for ln in lines:
        del ln["_x"], ln["_y"]
    return lines


def result_to_lines(res):
    """Convert an OCRResult into [{"text", "score", "box", "poly", "row"}] in
    reading order (rows top to bottom, regions of a row left to right)."""
    lines = []
    for text, score, box, poly in zip(
        res["rec_texts"], res["rec_scores"], res["rec_boxes"], res["rec_polys"]
    ):
        lines.append(
            {
                "text": text,
                "score": float(score),
                "box": [int(v) for v in box],  # x_min, y_min, x_max, y_max
                "poly": [[int(x), int(y)] for x, y in poly],
            }
        )
    return group_rows(lines)


def lines_to_text(lines):
    """Join recognized lines; regions of one row are separated by tabs."""
    rows = {}
    for ln in lines:
        rows.setdefault(ln["row"], []).append(ln["text"])
    return "\n".join("\t".join(texts) for _, texts in sorted(rows.items()))
