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

"""Validate PaddleOCR recognition / detection label files (Ukrainian OCR).

Recognition label format (SimpleDataSet):   <image_path>\\t<transcription>
Detection label format:                     <image_path>\\t<json list of
                                            {"transcription": str, "points": [[x, y], ...]}>

Detected problems: missing images, corrupted / unreadable images, malformed
lines, empty transcriptions, characters absent from the dictionary, invalid
UTF-8 and mojibake, non-NFC Unicode, duplicated records, conflicting labels,
labels longer than max_text_length (silently dropped by the PaddleOCR label
encoders), suspicious image sizes and train/val/test leakage.

Examples:
    python tools/ukrainian/check_dataset.py \\
        --label_file train_data/uk_printed/train.txt \\
        --data_dir train_data/uk_printed \\
        --dict ppocr/utils/dict/ppocrv5_uk_dict.txt --max_text_length 25

    # cleaned copy: Unicode-normalized labels, broken lines removed
    python tools/ukrainian/check_dataset.py --label_file train.txt \\
        --data_dir train_data/x --fix_output train.clean.txt
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import unicodedata

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from ppocr.utils.ukrainian import (  # noqa: E402
    APOSTROPHE_MODIFIER,
    NON_UK_CYRILLIC,
    mixed_script_words,
    normalize_uk_text,
    read_char_dict,
    validate_char_dict,
)

DEFAULT_DICT = os.path.join(
    __dir__, "..", "..", "ppocr", "utils", "dict", "ppocrv5_uk_dict.txt"
)

# UTF-8 Cyrillic (lead bytes 0xD0/0xD1) or typographic quotes (0xE2 0x80)
# decoded as latin-1 / cp1252, e.g. "Ð¿Ñ–Ð´" instead of "під".
_MOJIBAKE_RE = re.compile(
    "[ÐÑ][\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d\u017e"
    "\u0192\u02c6\u02dc\u2013-\u2122]|â€"
)
# UTF-8 Ukrainian letters decoded as cp1251, e.g. "С–" (і), "С—" (ї), "Р„" (Є)
_CP1251_MOJIBAKE = ("Р†", "Р‡", "С–", "С—", "Р„", "С”", "Тђ", "Т‘", "вЂ")

ERROR = "error"
WARNING = "warning"

# issue code -> (severity, description)
ISSUES = {
    "invalid_utf8": (ERROR, "line is not valid UTF-8"),
    "bad_format": (ERROR, "line does not contain exactly one tab delimiter"),
    "empty_label": (ERROR, "empty transcription"),
    "missing_image": (ERROR, "image file does not exist"),
    "corrupted_image": (ERROR, "image cannot be decoded"),
    "bad_det_json": (ERROR, "detection annotation is not a valid JSON list"),
    "bad_det_points": (ERROR, "polygon has < 3 points or invalid coordinates"),
    "det_points_outside": (WARNING, "polygon lies outside of the image"),
    "unknown_chars": (WARNING, "characters absent from the dictionary (dropped)"),
    "too_long": (ERROR, "label longer than max_text_length (sample is dropped)"),
    "mojibake": (ERROR, "label looks like text decoded with a wrong encoding"),
    "non_nfc": (WARNING, "label is not Unicode NFC (e.g. decomposed й/ї)"),
    "modifier_apostrophe": (
        WARNING,
        "U+02BC apostrophe (canonicalized to U+2019 by --fix_output)",
    ),
    "non_uk_cyrillic": (WARNING, "Russian/Belarusian letters (ы, э, ё, ъ, ў)"),
    "mixed_script": (WARNING, "word mixes Cyrillic and Latin letters"),
    "carriage_return": (WARNING, "label ends with \\r (Windows line ending)"),
    "edge_whitespace": (WARNING, "label has leading/trailing whitespace"),
    "duplicate_line": (WARNING, "identical record appears more than once"),
    "conflicting_label": (ERROR, "same image has different transcriptions"),
    "duplicate_image_content": (WARNING, "identical image bytes under other path"),
    "tiny_image": (ERROR, "image is smaller than min size"),
    "extreme_aspect": (WARNING, "aspect ratio unusual for a text line crop"),
    "text_too_long_for_width": (
        WARNING,
        "too many characters for the image width (CTC needs >= 1 step per char)",
    ),
    "leak": (ERROR, "image also present in another split (train/val/test leakage)"),
}


class Report(object):
    def __init__(self, max_examples=10):
        self.counts = collections.Counter()
        self.examples = collections.defaultdict(list)
        self.max_examples = max_examples
        self.bad_lines = set()
        self.unknown_chars = collections.Counter()
        self.num_records = 0

    def add(self, code, line_no, detail=""):
        self.counts[code] += 1
        if ISSUES[code][0] == ERROR:
            self.bad_lines.add(line_no)
        if len(self.examples[code]) < self.max_examples:
            self.examples[code].append({"line": line_no, "detail": detail})

    @property
    def num_errors(self):
        return sum(c for k, c in self.counts.items() if ISSUES[k][0] == ERROR)

    def to_dict(self):
        return {
            "num_records": self.num_records,
            "num_bad_records": len(self.bad_lines),
            "issues": {
                k: {
                    "severity": ISSUES[k][0],
                    "description": ISSUES[k][1],
                    "count": c,
                    "examples": self.examples[k],
                }
                for k, c in sorted(self.counts.items())
            },
            "unknown_chars": {
                "{} (U+{:04X})".format(ch, ord(ch)): n
                for ch, n in self.unknown_chars.most_common()
            },
        }


def _read_image_size(path):
    """Return (h, w) or None when the image cannot be decoded."""
    import cv2
    import numpy as np

    with open(path, "rb") as f:
        buf = f.read()
    if not buf:
        return None, buf
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, buf
    return img.shape[:2], buf


def check_text(text, line_no, report, charset, args):
    if text.endswith("\r"):
        report.add("carriage_return", line_no)
        text = text.rstrip("\r")
    if text.strip() == "":
        report.add("empty_label", line_no)
        return
    if text != text.strip():
        report.add("edge_whitespace", line_no, repr(text))
    if _MOJIBAKE_RE.search(text) or any(m in text for m in _CP1251_MOJIBAKE):
        report.add("mojibake", line_no, text)
    if unicodedata.normalize("NFC", text) != text:
        report.add("non_nfc", line_no, text)
    if APOSTROPHE_MODIFIER in text:
        report.add("modifier_apostrophe", line_no, text)
    if any(c in NON_UK_CYRILLIC for c in text):
        report.add("non_uk_cyrillic", line_no, text)
    mixed = mixed_script_words(text)
    if mixed:
        report.add("mixed_script", line_no, " ".join(mixed))
    if charset is not None:
        normalized = normalize_uk_text(text)
        unknown = sorted({c for c in normalized if c not in charset})
        if unknown:
            report.add(
                "unknown_chars", line_no, "{!r} in {}".format("".join(unknown), text)
            )
            for c in normalized:
                if c not in charset:
                    report.unknown_chars[c] += 1
    if args.max_text_length and len(normalize_uk_text(text)) > args.max_text_length:
        report.add(
            "too_long",
            line_no,
            "{} > {}: {}".format(len(text), args.max_text_length, text),
        )


def check_det_label(label, line_no, report, charset, args, img_hw):
    try:
        boxes = json.loads(label)
        assert isinstance(boxes, list)
    except Exception as e:  # noqa: BLE001
        report.add("bad_det_json", line_no, str(e))
        return
    for box in boxes:
        try:
            pts = box["points"]
            text = box.get("transcription", "")
            ok = len(pts) >= 3 and all(
                len(p) == 2 and all(isinstance(v, (int, float)) for v in p) for p in pts
            )
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            report.add("bad_det_points", line_no, str(box)[:200])
            continue
        if img_hw is not None:
            h, w = img_hw
            if any(x < -1 or y < -1 or x > w + 1 or y > h + 1 for x, y in pts):
                report.add("det_points_outside", line_no, str(pts))
        if text not in ("###", "*") and text:
            check_text(text, line_no, report, charset, _NoLength(args))


class _NoLength(object):
    # page level transcriptions are not limited by max_text_length
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.max_text_length = 0


def check_label_file(label_file, data_dir, args, charset):
    report = Report(args.max_examples)
    seen_lines = {}
    label_by_image = {}
    hash_to_image = {}
    images = set()
    clean_lines = []

    with open(label_file, "rb") as f:
        raw_lines = f.read().split(b"\n")
    if raw_lines and raw_lines[-1] == b"":
        raw_lines = raw_lines[:-1]
    if raw_lines and raw_lines[0].startswith(b"\xef\xbb\xbf"):
        raw_lines[0] = raw_lines[0][3:]

    for line_no, raw in enumerate(raw_lines, start=1):
        if raw.strip() == b"":
            continue
        report.num_records += 1
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            report.add("invalid_utf8", line_no, str(e))
            continue
        line_key = line.rstrip("\r")
        if line_key in seen_lines:
            report.add(
                "duplicate_line",
                line_no,
                "first at line {}".format(seen_lines[line_key]),
            )
        else:
            seen_lines[line_key] = line_no
        parts = line.split(args.delimiter)
        if len(parts) != 2:
            report.add("bad_format", line_no, line[:200])
            continue
        img_name, label = parts
        img_path = (
            img_name if os.path.isabs(img_name) else os.path.join(data_dir, img_name)
        )
        images.add(os.path.normpath(img_path))

        if args.mode == "rec":
            prev = label_by_image.get(img_path)
            if prev is not None and prev != label.rstrip("\r"):
                report.add(
                    "conflicting_label", line_no, "{!r} vs {!r}".format(prev, label)
                )
            label_by_image[img_path] = label.rstrip("\r")

        img_hw = None
        if args.skip_images:
            pass
        elif not os.path.isfile(img_path):
            report.add("missing_image", line_no, img_path)
        else:
            img_hw, buf = _read_image_size(img_path)
            if img_hw is None:
                report.add("corrupted_image", line_no, img_path)
            else:
                h, w = img_hw
                if h < args.min_size or w < args.min_size:
                    report.add("tiny_image", line_no, "{}x{} {}".format(w, h, img_path))
                if args.mode == "rec":
                    ratio = w / float(h)
                    if ratio < 0.2 or ratio > args.max_aspect:
                        report.add(
                            "extreme_aspect", line_no, "{}x{} {}".format(w, h, img_path)
                        )
                    # PP-OCRv5 rec downsamples width by 8 after resizing to h=48
                    steps = int(48 * ratio) // 8
                    if steps and len(label) > steps:
                        report.add(
                            "text_too_long_for_width",
                            line_no,
                            "{} chars, ~{} steps: {}".format(
                                len(label), steps, img_path
                            ),
                        )
                if args.hash_images:
                    digest = hashlib.md5(buf).hexdigest()
                    other = hash_to_image.get(digest)
                    if other is not None and other != img_path:
                        report.add(
                            "duplicate_image_content",
                            line_no,
                            "{} == {}".format(img_path, other),
                        )
                    hash_to_image.setdefault(digest, img_path)

        if args.mode == "rec":
            check_text(label, line_no, report, charset, args)
        else:
            check_det_label(label.rstrip("\r"), line_no, report, charset, args, img_hw)

        if line_no not in report.bad_lines:
            if args.mode == "rec":
                label = normalize_uk_text(label.rstrip("\r")).strip()
            clean_lines.append(
                "{}{}{}".format(img_name, args.delimiter, label.rstrip("\r"))
            )

    return report, images, clean_lines


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate PaddleOCR label files for Ukrainian OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--label_file", required=True, nargs="+")
    parser.add_argument("--data_dir", default="", help="root joined with image paths")
    parser.add_argument("--mode", choices=["rec", "det"], default="rec")
    parser.add_argument("--dict", default=DEFAULT_DICT, help="'' disables the check")
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument("--delimiter", default="\t")
    parser.add_argument("--min_size", type=int, default=8)
    parser.add_argument("--max_aspect", type=float, default=40.0)
    parser.add_argument(
        "--hash_images", action="store_true", help="detect identical images"
    )
    parser.add_argument("--skip_images", action="store_true", help="only check labels")
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--report", help="write a JSON report")
    parser.add_argument(
        "--fix_output", help="write a cleaned label file (single input)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with code 1 when warnings are found (errors always fail)",
    )
    args = parser.parse_args(argv)

    charset = None
    if args.dict:
        chars = read_char_dict(args.dict)
        problems = validate_char_dict(chars)
        if problems["duplicates"]:
            print(
                "WARNING: dictionary {} has duplicates: {}".format(
                    args.dict, problems["duplicates"]
                )
            )
        charset = set(chars) | {" "}

    full = {}
    split_images = {}
    exit_code = 0
    for label_file in args.label_file:
        report, images, clean = check_label_file(
            label_file, args.data_dir, args, charset
        )
        split_images[label_file] = images
        full[label_file] = report
        if args.fix_output:
            if len(args.label_file) != 1:
                sys.exit("--fix_output supports a single --label_file")
            with open(args.fix_output, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(clean) + ("\n" if clean else ""))
            print(
                "cleaned label file: {} ({} of {} records kept)".format(
                    args.fix_output, len(clean), report.num_records
                )
            )

    files = list(split_images)
    for i, a in enumerate(files):
        for b in files[i + 1 :]:
            common = split_images[a] & split_images[b]
            for img in sorted(common)[: args.max_examples]:
                full[b].add("leak", 0, "{} also in {}".format(img, a))
            if len(common) > args.max_examples:
                full[b].counts["leak"] += len(common) - args.max_examples

    out = {}
    for label_file, report in full.items():
        d = report.to_dict()
        out[label_file] = d
        print("=" * 70)
        print(
            "{}: {} records, {} bad".format(
                label_file, d["num_records"], d["num_bad_records"]
            )
        )
        if not d["issues"]:
            print("  no issues found")
        for code, info in d["issues"].items():
            print(
                "  [{}] {}: {} - {}".format(
                    info["severity"].upper(), code, info["count"], info["description"]
                )
            )
            for ex in info["examples"][:3]:
                print("      line {}: {}".format(ex["line"], ex["detail"][:160]))
        if d["unknown_chars"]:
            print(
                "  unknown chars: {}".format(
                    ", ".join(
                        "{}x{}".format(k, v)
                        for k, v in list(d["unknown_chars"].items())[:20]
                    )
                )
            )
        if report.num_errors:
            exit_code = 1
        elif args.strict and report.counts:
            exit_code = 1

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("report written to {}".format(args.report))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
