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

"""Convert a RUKOPYS split (metadata.jsonl + images/) to PaddleOCR datasets.

RUKOPYS record: {"file_name": "images/<uuid>.jpg", "image_width": W,
"image_height": H, "regions": [{"bbox": [x1, y1, x2, y2], "type":
"handwritten" | "printed" | "formula" | "table" | "annotation" | "image" |
"graph", "language": "uk" | "other", "legibility": "legible" | "illegible",
"text": "..."}]}. Handwritten and printed regions are single lines.

Outputs (only for pages whose image exists locally):
* recognition: ``<output_dir>/images/<uuid>__<k>.jpg`` line crops and
  ``<output_dir>/<name>.txt`` with ``images/<uuid>__<k>.jpg\\t<text>``
* detection (``--det_label``): ``<page path>\\t[{"transcription", "points"}]``
  relative to the split directory; unreadable / non-line text regions are
  marked ``###`` (ignored by DetLabelEncode), images and graphs are skipped.

With ``--crop_source det`` lines are cropped from the polygons returned by the
text detector (same settings and quad cropping as the PaddleOCR pipeline):
the detector pieces of every annotated line are merged into one tight,
deskewed quad. Annotated boxes are axis aligned and include parts of
neighbouring slanted lines; lines covered by detections for less than
``--min_coverage`` of their width are skipped.

Regions with editing markers (``~~struck~~``, ``~~old~~{new}``,
``[illegible]``) are skipped for recognition because the image shows both the
struck and the corrected text.

Example:
    python tools/ukrainian/convert_rukopys.py --split_dir train_data/rukopys_raw/test \\
        --output_dir train_data/uk_handwritten --name rukopys_test \\
        --det_label train_data/rukopys_raw/test/det_test.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from ppocr.utils.ukrainian import normalize_uk_text, read_char_dict  # noqa: E402

DEFAULT_DICT = os.path.join(
    __dir__, "..", "..", "ppocr", "utils", "dict", "ppocrv5_uk_dict.txt"
)
MARKER_RE = re.compile(r"~~|\[illegible\]|\{[^}]*\}", re.IGNORECASE)
LINE_TYPES = ("handwritten", "printed", "annotation")
IGNORED_TEXT_TYPES = ("formula", "table")


def load_page(path, meta_w, meta_h):
    """Load an RGB page whose size matches the annotation (EXIF aware)."""
    from PIL import Image, ImageOps

    img = Image.open(path)
    img.load()
    if img.size != (meta_w, meta_h):
        transposed = ImageOps.exif_transpose(img)
        if transposed.size == (meta_w, meta_h):
            img = transposed
        else:
            raise ValueError(
                "image size {} != annotation {}x{}".format(img.size, meta_w, meta_h)
            )
    return img.convert("RGB")


def clean_text(text):
    """Return normalized transcription or None if it can not be used for rec."""
    if text is None or "\n" in text or MARKER_RE.search(text):
        return None
    text = " ".join(normalize_uk_text(text).split())
    return text or None


def crop_box(bbox, img_w, img_h, pad_ratio):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    pad = (y2 - y1) * pad_ratio
    return (
        max(0, int(round(x1 - pad))),
        max(0, int(round(y1 - pad))),
        min(img_w, int(round(x2 + pad))),
        min(img_h, int(round(y2 + pad))),
    )


def assign_polys_to_lines(boxes, polys, min_coverage=0.85, min_inside=0.6):
    """Group detected polygons by annotated line.

    The detector often splits one handwritten line into several pieces, and
    annotated boxes of slanted lines overlap. Every polygon is assigned to the
    line box containing the largest part of it (ties: nearest vertical center);
    a line is kept if its polygons cover >= ``min_coverage`` of its width.
    Returns {box_idx: [polygon, ...]}.
    """
    groups = {}
    for poly in polys:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        r = [min(xs), min(ys), max(xs), max(ys)]
        area = max(1e-6, (r[2] - r[0]) * (r[3] - r[1]))
        cy = (r[1] + r[3]) / 2
        best, best_key = None, None
        for bi, b in enumerate(boxes):
            ix = max(0.0, min(r[2], b[2]) - max(r[0], b[0]))
            iy = max(0.0, min(r[3], b[3]) - max(r[1], b[1]))
            inside = ix * iy / area
            if inside < min_inside:
                continue
            key = (round(inside, 2), -abs(cy - (b[1] + b[3]) / 2))
            if best_key is None or key > best_key:
                best, best_key = bi, key
        if best is not None:
            groups.setdefault(best, []).append(poly)
    kept = {}
    for bi, group in groups.items():
        b = boxes[bi]
        intervals = sorted((min(p[0] for p in g), max(p[0] for p in g)) for g in group)
        covered, cur_s, cur_e = 0.0, None, None
        for s_, e_ in intervals:
            s_, e_ = max(s_, b[0]), min(e_, b[2])
            if e_ <= s_:
                continue
            if cur_e is None or s_ > cur_e:
                if cur_e is not None:
                    covered += cur_e - cur_s
                cur_s, cur_e = s_, e_
            else:
                cur_e = max(cur_e, e_)
        if cur_e is not None:
            covered += cur_e - cur_s
        if covered >= min_coverage * (b[2] - b[0]):
            kept[bi] = group
    return kept


def merge_polys(polys):
    """Minimum-area quad around all points of the given polygons."""
    import cv2
    import numpy as np

    pts = np.array([p for poly in polys for p in poly], dtype=np.float32)
    return cv2.boxPoints(cv2.minAreaRect(pts)).tolist()


class DetectorCropper(object):
    """Crops text lines exactly like the PaddleOCR pipeline: same detector
    settings as the OCR pipeline config and PaddleX ``CropByPolys`` (quad)."""

    def __init__(
        self,
        model_name,
        model_dir=None,
        device="cpu",
        limit_side_len=64,
        limit_type="min",
    ):
        from paddleocr import TextDetection
        from paddlex.inference.pipelines.components.common.crop_image_regions import (
            CropByPolys,
        )

        self.det = TextDetection(
            model_name=model_name,
            model_dir=model_dir,
            device=device,
            limit_side_len=limit_side_len,
            limit_type=limit_type,
            thresh=0.3,
            box_thresh=0.6,
            unclip_ratio=1.5,
        )
        self.crop = CropByPolys(det_box_type="quad")

    def detect(self, img_bgr):
        res = self.det.predict(img_bgr)[0]
        return [[[float(x), float(y)] for x, y in poly] for poly in res["dt_polys"]]

    def crop_poly(self, img_bgr, poly):
        return self.crop(img_bgr, [poly])[0]


DONE_MARKER = "#DONE\t"


def convert(args):
    stats = collections.Counter()
    cropper = None
    if args.crop_source == "det":
        cropper = DetectorCropper(
            args.det_model_name,
            args.det_model_dir,
            args.device,
            args.det_limit_side_len,
            args.det_limit_type,
        )
    charset = set(read_char_dict(args.dict)) | {" "} if args.dict else None
    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rec_lines, det_lines = [], []

    with open(
        os.path.join(args.split_dir, "metadata.jsonl"), "r", encoding="utf-8"
    ) as f:
        records = [json.loads(line) for line in f if line.strip()]

    # --resume: every finished page is appended to <det_label>.partial and /
    # or <name>.txt.partial (its line crops followed by a "#DONE<TAB>page"
    # marker) immediately, so an interrupted run continues where it stopped
    partial, rec_partial, done_pages = None, None, None
    rec_path = os.path.join(args.output_dir, args.name + ".txt")
    if args.resume:
        if not args.no_rec:
            done_pages = set()
            rec_partial_path = rec_path + ".partial"
            if os.path.isfile(rec_partial_path):
                page_lines = []
                with open(rec_partial_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.endswith("\n"):
                            break  # truncated by the interruption
                        line = line.rstrip("\n")
                        if line.startswith(DONE_MARKER):
                            done_pages.add(line[len(DONE_MARKER) :])
                            rec_lines.extend(page_lines)
                            page_lines = []
                        else:
                            page_lines.append(line)
                with open(rec_partial_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("".join(line + "\n" for line in rec_lines))
                    f.write("".join(DONE_MARKER + p + "\n" for p in sorted(done_pages)))
            rec_partial = open(rec_partial_path, "a", encoding="utf-8", newline="\n")
        if args.det_label:
            partial_path = args.det_label + ".partial"
            if os.path.isfile(partial_path):
                with open(partial_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not (line.endswith("\n") and "\t" in line):
                            continue
                        page_name = line.split("\t", 1)[0]
                        if done_pages is None or page_name in done_pages:
                            det_lines.append(line.rstrip("\n"))
                if done_pages is None:
                    done_pages = {ln.split("\t", 1)[0] for ln in det_lines}
                # rewrite without a possibly truncated last line
                with open(partial_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("".join(line + "\n" for line in det_lines))
            partial = open(partial_path, "a", encoding="utf-8", newline="\n")
        stats["pages_resumed"] = len(done_pages or ())
    done_pages = done_pages or set()

    for rec in records:
        if rec["file_name"] in done_pages:
            continue
        page_path = os.path.join(args.split_dir, rec["file_name"])
        if not os.path.isfile(page_path):
            stats["pages_not_downloaded"] += 1
            continue
        try:
            page = load_page(page_path, rec["image_width"], rec["image_height"])
        except Exception as e:  # noqa: BLE001
            stats["pages_unreadable"] += 1
            print("skip {}: {}".format(page_path, e))
            continue
        stats["pages"] += 1
        first_rec_line = len(rec_lines)
        uuid = os.path.splitext(os.path.basename(rec["file_name"]))[0]
        det_boxes, candidates, det_box_of = [], [], {}
        for k, region in enumerate(rec.get("regions", [])):
            rtype = region.get("type")
            bbox = region.get("bbox")
            if not bbox or len(bbox) != 4:
                stats["bad_bbox"] += 1
                continue
            legible = region.get("legibility") == "legible"
            text = clean_text(region.get("text"))

            if rtype in LINE_TYPES + IGNORED_TEXT_TYPES:
                x1, y1, x2, y2 = bbox
                det_text = text if (rtype in LINE_TYPES and legible and text) else "###"
                det_box_of[k] = len(det_boxes)
                det_boxes.append(
                    {
                        "transcription": det_text,
                        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    }
                )

            if rtype not in args.types:
                continue
            stats["regions_" + rtype] += 1
            if not legible:
                stats["skip_illegible"] += 1
                continue
            if args.language and region.get("language") != args.language:
                stats["skip_language"] += 1
                continue
            if text is None:
                stats["skip_markers_or_empty"] += 1
                continue
            if charset is not None and any(c not in charset for c in text):
                stats["skip_unknown_chars"] += 1
                continue
            if args.max_text_length and len(text) > args.max_text_length:
                stats["skip_too_long"] += 1
                continue
            candidates.append((k, bbox, text))
        if args.no_rec:
            candidates = []
        if cropper is None:
            for k, bbox, text in candidates:
                box = crop_box(bbox, page.width, page.height, args.pad_ratio)
                if box is None or box[3] - box[1] < 8 or box[2] - box[0] < 8:
                    stats["skip_tiny"] += 1
                    continue
                name = "{}__{:04d}.jpg".format(uuid, k)
                page.crop(box).save(os.path.join(img_dir, name), quality=95)
                rec_lines.append("images/{}\t{}".format(name, text))
                stats["rec_samples"] += 1
        elif candidates or args.refine_det_label:
            import cv2
            import numpy as np

            page_bgr = np.ascontiguousarray(np.asarray(page)[:, :, ::-1])
            polys = cropper.detect(page_bgr)
            # assign detected pieces to every text line of the page, so that a
            # piece is never attached to a handwritten line it does not belong to
            regions = rec["regions"]
            line_idx = [
                i
                for i, r in enumerate(regions)
                if r.get("type") in LINE_TYPES and r.get("bbox") and len(r["bbox"]) == 4
            ]
            groups = assign_polys_to_lines(
                [regions[i]["bbox"] for i in line_idx], polys, args.min_coverage
            )
            poly_of = {line_idx[bi]: merge_polys(g) for bi, g in groups.items()}
            if args.refine_det_label:
                # tight rotated quad of the whole line instead of the axis
                # aligned annotated box, which overlaps neighbouring lines
                for k, quad in poly_of.items():
                    if k in det_box_of:
                        det_boxes[det_box_of[k]]["points"] = [
                            [int(round(x)), int(round(y))] for x, y in quad
                        ]
                        stats["det_label_refined"] += 1
            for k, bbox, text in candidates:
                poly = poly_of.get(k)
                if poly is None and args.gt_fallback:
                    box = crop_box(bbox, page.width, page.height, args.pad_ratio)
                    # an annotated box of a single line needs roughly >= 0.2
                    # width/height units per character, else it spans lines
                    if (
                        box is not None
                        and box[3] - box[1] >= 8
                        and (box[2] - box[0]) / (box[3] - box[1]) >= 0.2 * len(text)
                    ):
                        name = "{}__{:04d}.jpg".format(uuid, k)
                        page.crop(box).save(os.path.join(img_dir, name), quality=95)
                        rec_lines.append("images/{}\t{}".format(name, text))
                        stats["rec_samples_gt_fallback"] += 1
                        continue
                    stats["skip_gt_fallback_multiline"] += 1
                    continue
                if poly is None:
                    stats["skip_no_det_match"] += 1
                    continue
                crop = cropper.crop_poly(page_bgr, poly)
                if crop.shape[0] < 8 or crop.shape[1] < 8:
                    stats["skip_tiny"] += 1
                    continue
                name = "{}__{:04d}d.jpg".format(uuid, k)
                cv2.imwrite(
                    os.path.join(img_dir, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                rec_lines.append("images/{}\t{}".format(name, text))
                stats["rec_samples"] += 1
        det_lines.append(
            "{}\t{}".format(rec["file_name"], json.dumps(det_boxes, ensure_ascii=False))
        )
        if partial is not None:
            partial.write(det_lines[-1] + "\n")
            partial.flush()
        if rec_partial is not None:
            rec_partial.write(
                "".join(line + "\n" for line in rec_lines[first_rec_line:])
                + DONE_MARKER
                + rec["file_name"]
                + "\n"
            )
            rec_partial.flush()

    with open(rec_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rec_lines) + ("\n" if rec_lines else ""))
    if args.det_label:
        with open(args.det_label, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(det_lines) + ("\n" if det_lines else ""))
    for f in (partial, rec_partial):
        if f is not None:
            f.close()
            os.remove(f.name)
    return stats, rec_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", default="rukopys")
    parser.add_argument(
        "--types", nargs="+", default=["handwritten"], choices=list(LINE_TYPES)
    )
    parser.add_argument("--language", default="uk", help="'' keeps all languages")
    parser.add_argument(
        "--dict", default=DEFAULT_DICT, help="skip lines with other chars; '' disables"
    )
    parser.add_argument("--max_text_length", type=int, default=0, help="0 disables")
    parser.add_argument("--pad_ratio", type=float, default=0.1)
    parser.add_argument("--det_label", help="also write a detection label file")
    parser.add_argument(
        "--crop_source",
        choices=["gt", "det"],
        default="gt",
        help="gt: annotated boxes; det: detector polygons matched to the annotation "
        "(same crops as at inference)",
    )
    parser.add_argument(
        "--gt_fallback",
        action="store_true",
        help="det mode: crop the annotated box of lines the detector did not "
        "cover, unless the box obviously spans several lines",
    )
    parser.add_argument(
        "--refine_det_label",
        action="store_true",
        help="det mode: in --det_label replace annotated boxes of lines covered "
        "by the detector with the tight rotated quad of the merged pieces",
    )
    parser.add_argument(
        "--no_rec", action="store_true", help="only write --det_label, no line crops"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run from <name>.txt.partial / <det_label>.partial",
    )
    parser.add_argument("--det_model_name", default="PP-OCRv5_server_det")
    parser.add_argument(
        "--det_limit_side_len",
        type=int,
        default=64,
        help="detector input size (pipeline default: 64 with --det_limit_type min; "
        "handwritten examples use 960 max)",
    )
    parser.add_argument("--det_limit_type", choices=["min", "max"], default="min")
    parser.add_argument("--det_model_dir")
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.85,
        help="det mode: share of the line width the detected pieces must cover",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.refine_det_label and args.crop_source != "det":
        parser.error("--refine_det_label requires --crop_source det")
    if args.resume and args.no_rec and not args.det_label:
        parser.error("--resume with --no_rec requires --det_label")

    stats, rec_path = convert(args)
    for k, v in sorted(stats.items()):
        print("{:>26}: {}".format(k, v))
    print("recognition labels: {}".format(rec_path))
    if args.det_label:
        print(
            "detection labels: {} (paths relative to {})".format(
                args.det_label, args.split_dir
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
