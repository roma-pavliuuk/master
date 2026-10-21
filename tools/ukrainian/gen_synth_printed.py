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

"""Generate synthetic text-line images of PRINTED Ukrainian text.

Images imitate cropped text lines from photos and scans: different fonts and
sizes, colored / textured backgrounds, uneven illumination, small rotation and
perspective, blur, noise, JPEG artifacts, low resolution and table borders.
The output is a PaddleOCR recognition dataset (``images/`` + label file).

Synthetic printed text is NOT a replacement for real handwriting data.

Fonts are discovered in --font_dir and kept only if they contain glyphs for
the whole Ukrainian alphabet (including Ґ Є І Ї and the apostrophe ’).
Font families are split deterministically between ``train`` and ``test`` so
that the test set measures generalization to unseen fonts.

Example:
    python tools/ukrainian/gen_synth_printed.py --output_dir train_data/uk_printed \\
        --num 5000 --font_subset train --label_name train_all.txt
    python tools/ukrainian/gen_synth_printed.py --output_dir train_data/uk_printed \\
        --num 500 --font_subset test --label_name test.txt --seed 1
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from ppocr.utils.ukrainian import (
    UK_ALPHABET,
    normalize_uk_text,
    read_char_dict,
)  # noqa: E402

DEFAULT_CORPUS = os.path.join(__dir__, "data", "uk_corpus_sample.txt")
DEFAULT_DICT = os.path.join(
    __dir__, "..", "..", "ppocr", "utils", "dict", "ppocrv5_uk_dict.txt"
)
REQUIRED_GLYPHS = UK_ALPHABET + "0123456789’'«»№—"

MONTHS = [
    "січня",
    "лютого",
    "березня",
    "квітня",
    "травня",
    "червня",
    "липня",
    "серпня",
    "вересня",
    "жовтня",
    "листопада",
    "грудня",
]
STREETS = [
    "вул. Шевченка",
    "вул. Івана Франка",
    "просп. Перемоги",
    "вул. Ґонти",
    "бульв. Лесі Українки",
    "вул. Їжакевича",
    "пров. Європейський",
    "вул. Січових Стрільців",
    "площа Ринок",
    "вул. Об'їзна",
]
CITIES = [
    "Київ",
    "Львів",
    "Ірпінь",
    "Ізмаїл",
    "Житомир",
    "Чернівці",
    "Ужгород",
    "Запоріжжя",
    "Їжаківка",
    "Біла Церква",
    "Кам'янець-Подільський",
]


# --------------------------------------------------------------------------- #
# Text sampling
# --------------------------------------------------------------------------- #
def _template_text(rng: random.Random) -> str:
    kind = rng.randrange(7)
    if kind == 0:
        return "{:02d}.{:02d}.{}".format(
            rng.randint(1, 28), rng.randint(1, 12), rng.randint(1950, 2035)
        )
    if kind == 1:
        return "{} {} {} р.".format(
            rng.randint(1, 28), rng.choice(MONTHS), rng.randint(1991, 2035)
        )
    if kind == 2:
        value = "{:,}".format(rng.randint(1, 999999)).replace(",", " ")
        return "{},{:02d} {}".format(
            value, rng.randint(0, 99), rng.choice(["грн", "₴", "€", "$"])
        )
    if kind == 3:
        return "{}, {}".format(rng.choice(STREETS), rng.randint(1, 150))
    if kind == 4:
        return "м. {}, {:05d}".format(rng.choice(CITIES), rng.randint(1000, 99999))
    if kind == 5:
        return "+380 ({:02d}) {:03d}-{:02d}-{:02d}".format(
            rng.randint(39, 99),
            rng.randint(0, 999),
            rng.randint(0, 99),
            rng.randint(0, 99),
        )
    return "№ {}/{}-{}".format(
        rng.randint(1, 999), rng.randint(1, 99), rng.choice("АБВГҐДЕЄ")
    )


def sample_text(rng: random.Random, corpus, max_len: int) -> str:
    if rng.random() < 0.2:
        text = _template_text(rng)
    else:
        words = rng.choice(corpus).split()
        start = rng.randrange(len(words))
        n = rng.randint(1, max(1, len(words) - start))
        text = " ".join(words[start : start + n])
        r = rng.random()
        if r < 0.08:
            text = text.upper()
        elif r < 0.12:
            text = text.capitalize()
    text = normalize_uk_text(text).strip()
    while len(text) > max_len:
        text = " ".join(text.split()[:-1]) if " " in text else text[:max_len]
    return text


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
def _glyph_signature(font, ch):
    mask = font.getmask(ch)
    return mask.size, bytes(mask)


def font_covers(font_path: str, chars: str = REQUIRED_GLYPHS) -> bool:
    """True if every char renders differently from the font's .notdef glyph."""
    try:
        font = ImageFont.truetype(font_path, 32)
    except (OSError, ValueError):
        return False
    notdef = {_glyph_signature(font, c) for c in ("\U0010fffd", "", "￿")}
    for ch in chars:
        sig = _glyph_signature(font, ch)
        if sig in notdef or sig[0][0] == 0:
            return False
    return True


class GlyphCoverage(object):
    """Lazily cached per-font glyph coverage used to reject tofu (□) samples."""

    def __init__(self):
        self._fonts = {}
        self._cache = {}

    def covers(self, font_path: str, text: str) -> bool:
        if font_path not in self._fonts:
            font = ImageFont.truetype(font_path, 32)
            notdef = {
                _glyph_signature(font, c) for c in ("\U0010fffd", "\ue000", "\uffff")
            }
            self._fonts[font_path] = (font, notdef)
        font, notdef = self._fonts[font_path]
        for ch in set(text):
            if ch == " ":
                continue
            key = (font_path, ch)
            if key not in self._cache:
                sig = _glyph_signature(font, ch)
                self._cache[key] = sig not in notdef and sig[0][0] > 0
            if not self._cache[key]:
                return False
        return True


def discover_fonts(
    font_dirs, exclude_keywords=("emoji", "symbol", "dingbat", "math", "z003", "kacst")
):
    fonts = []
    for d in font_dirs:
        for root, _, files in os.walk(d):
            for name in sorted(files):
                if not name.lower().endswith((".ttf", ".otf")):
                    continue
                if any(k in name.lower() for k in exclude_keywords):
                    continue
                path = os.path.join(root, name)
                if font_covers(path):
                    fonts.append(path)
    return sorted(set(fonts))


def font_family(path: str) -> str:
    try:
        return ImageFont.truetype(path, 20).getname()[0]
    except OSError:
        return os.path.basename(path)


def split_fonts(fonts, test_ratio: float, subset: str):
    """Deterministic split of font families into train / test."""
    if subset == "all":
        return fonts
    families = sorted({font_family(f) for f in fonts})
    ranked = sorted(families, key=lambda f: hashlib.md5(f.encode("utf-8")).hexdigest())
    n_test = max(1, int(round(len(ranked) * test_ratio))) if len(ranked) > 1 else 0
    test_families = set(ranked[:n_test])
    return [f for f in fonts if (font_family(f) in test_families) == (subset == "test")]


# --------------------------------------------------------------------------- #
# Rendering and augmentation
# --------------------------------------------------------------------------- #
def _background(rng: random.Random, h: int, w: int, dark: bool):
    base = rng.randint(20, 70) if dark else rng.randint(180, 255)
    kind = rng.randrange(4)
    if kind == 0:
        img = np.full((h, w, 3), base, np.float32)
        img += np.array([rng.randint(-15, 15) for _ in range(3)], np.float32)
    elif kind == 1:  # linear gradient
        g = np.linspace(0, rng.uniform(-40, 40), w, dtype=np.float32)[None, :, None]
        img = np.full((h, w, 3), base, np.float32) + g
    elif kind == 2:  # paper texture
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, rng.uniform(4, 14), (h, w, 1)
        )
        noise = cv2.GaussianBlur(
            noise.astype(np.float32), (0, 0), rng.uniform(0.5, 2.0)
        )
        img = np.full((h, w, 3), base, np.float32) + (
            noise[..., None] if noise.ndim == 2 else noise
        )
    else:  # colored paper / sticker
        color = np.array([rng.randint(150, 255) for _ in range(3)], np.float32)
        img = np.ones((h, w, 3), np.float32) * (color * (0.3 if dark else 1.0))
    return np.clip(img, 0, 255)


def render_typewriter_line(text: str, font_path: str, rng: random.Random):
    """Typewriter-like line: monospace font, wide letter spacing (as in 1920s
    typed documents), slightly jumping baseline, uneven / worn ink and aged
    paper. The label stays the normal text without the extra spaces."""
    size = rng.randint(22, 56)
    font = ImageFont.truetype(font_path, size)
    tracking = size * rng.uniform(0.0, 0.6)  # extra space after every char
    word_gap = size * rng.uniform(0.0, 0.8)
    advances = []
    for ch in text:
        adv = font.getlength(ch) + tracking + (word_gap if ch == " " else 0)
        advances.append(adv)
    left, top, right, bottom = font.getbbox(text)
    th = bottom - top
    pad_x = int(size * rng.uniform(0.2, 0.8))
    pad_y = int(size * rng.uniform(0.2, 0.5))
    w = int(sum(advances) + 2 * pad_x)
    h = th + 2 * pad_y + 4
    # aged paper: yellowish / greyish, low contrast
    base = np.array(
        [rng.randint(200, 240), rng.randint(190, 230), rng.randint(150, 210)],
        np.float32,
    )
    bg = np.ones((h, w, 3), np.float32) * base
    bg += np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, rng.uniform(3, 12), (h, w, 1)
    )
    canvas = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    ribbon = rng.choice([(30, 30, 30), (50, 40, 35), (35, 35, 90), (70, 40, 90)])
    x = pad_x
    for ch, adv in zip(text, advances):
        if ch != " ":
            shade = rng.uniform(0.0, 0.45)  # worn / light strikes
            color = tuple(int(c + (base[i] - c) * shade) for i, c in enumerate(ribbon))
            dy = rng.randint(-2, 2)
            draw.text((x, pad_y - top + dy), ch, font=font, fill=color)
        x += adv
    img = np.asarray(canvas).copy()
    if rng.random() < 0.5:  # ink bleed or thin strokes
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k) if rng.random() < 0.6 else cv2.dilate(img, k)
    return img


def render_line(text: str, font_path: str, rng: random.Random):
    size = rng.randint(18, 64)
    font = ImageFont.truetype(font_path, size)
    left, top, right, bottom = font.getbbox(text)
    tw, th = right - left, bottom - top
    pad_x = int(size * rng.uniform(0.1, 0.8))
    pad_y = int(size * rng.uniform(0.1, 0.5))
    w, h = tw + 2 * pad_x, th + 2 * pad_y
    dark = rng.random() < 0.1
    bg = _background(rng, h, w, dark)
    if dark:
        color = tuple(rng.randint(200, 255) for _ in range(3))
    else:
        v = rng.randint(0, 80)
        color = (v + rng.randint(0, 30), v + rng.randint(0, 30), v + rng.randint(0, 60))

    canvas = Image.fromarray(bg.astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad_x - left, pad_y - top), text, font=font, fill=color)
    if rng.random() < 0.15:  # table borders / underline
        y = min(h - 1, pad_y + th + rng.randint(1, max(1, pad_y)))
        draw.line([(0, y), (w, y)], fill=color, width=rng.randint(1, 2))
        if rng.random() < 0.5:
            draw.line(
                [
                    (0, max(0, pad_y - rng.randint(1, max(1, pad_y)))),
                    (w, max(0, pad_y - 2)),
                ],
                fill=color,
                width=1,
            )
    return np.asarray(canvas)


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    img = img.astype(np.float32)
    # uneven illumination
    if rng.random() < 0.5:
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)
        cx, cy = rng.random(), rng.random()
        dist = np.sqrt(
            (x[None, :] - cx) ** 2 + ((y[:, None] - cy) * h / max(w, 1)) ** 2
        )
        mask = 1.0 - rng.uniform(0.15, 0.5) * dist / (dist.max() + 1e-6)
        img = img * mask[..., None]
    # small rotation (skew)
    if rng.random() < 0.5:
        angle = rng.uniform(-3.0, 3.0)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += nw / 2 - w / 2
        m[1, 2] += nh / 2 - h / 2
        border = tuple(float(v) for v in img[0, 0])
        img = cv2.warpAffine(
            img, m, (nw, nh), borderMode=cv2.BORDER_CONSTANT, borderValue=border
        )
        h, w = img.shape[:2]
    # slight perspective
    if rng.random() < 0.3:
        d = 0.03
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32(
            [
                [rng.uniform(0, d) * w, rng.uniform(0, d * 3) * h],
                [w - rng.uniform(0, d) * w, rng.uniform(0, d * 3) * h],
                [w - rng.uniform(0, d) * w, h - rng.uniform(0, d * 3) * h],
                [rng.uniform(0, d) * w, h - rng.uniform(0, d * 3) * h],
            ]
        )
        img = cv2.warpPerspective(
            img,
            cv2.getPerspectiveTransform(src, dst),
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )
    # low resolution scan
    if rng.random() < 0.3:
        f = rng.uniform(0.35, 0.8)
        small = cv2.resize(
            img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA
        )
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.4:
        img = img + np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, rng.uniform(2, 12), img.shape
        )
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 90)]
        )
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument(
        "--corpus",
        nargs="+",
        default=[DEFAULT_CORPUS],
        help="UTF-8 text files, one sentence per line",
    )
    parser.add_argument("--font_dir", nargs="+", default=["/usr/share/fonts"])
    parser.add_argument(
        "--fonts", nargs="*", help="explicit font files (skip discovery)"
    )
    parser.add_argument(
        "--font_subset", choices=["all", "train", "test"], default="all"
    )
    parser.add_argument("--test_font_ratio", type=float, default=0.25)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--dict", default=DEFAULT_DICT, help="drop samples with chars outside the dict"
    )
    parser.add_argument("--label_name", default="labels.txt")
    parser.add_argument("--image_subdir", default="images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list_fonts", action="store_true")
    parser.add_argument(
        "--typewriter_ratio",
        type=float,
        default=0.0,
        help="share of typewriter-style lines (monospace fonts, letter spacing, worn ink)",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    fonts = args.fonts or discover_fonts(args.font_dir)
    fonts = (
        split_fonts(fonts, args.test_font_ratio, args.font_subset)
        if not args.fonts
        else fonts
    )
    if args.list_fonts:
        for f in fonts:
            print("{}\t{}".format(font_family(f), f))
        return 0
    if not fonts:
        sys.exit(
            "no font with full Ukrainian coverage found in {}".format(args.font_dir)
        )

    mono_fonts = [f for f in fonts if "mono" in font_family(f).lower()]
    if args.typewriter_ratio > 0 and not mono_fonts:
        sys.exit("--typewriter_ratio needs monospace fonts (e.g. DejaVu Sans Mono)")
    corpus = []
    for path in args.corpus:
        with open(path, "r", encoding="utf-8") as f:
            corpus.extend(normalize_uk_text(line).strip() for line in f if line.strip())
    charset = set(read_char_dict(args.dict)) | {" "} if args.dict else None

    img_dir = os.path.join(args.output_dir, args.image_subdir)
    os.makedirs(img_dir, exist_ok=True)
    label_path = os.path.join(args.output_dir, args.label_name)
    prefix = os.path.splitext(args.label_name)[0]
    written = 0
    rejected = 0
    coverage = GlyphCoverage()
    with open(label_path, "w", encoding="utf-8", newline="\n") as fout:
        while written < args.num:
            text = sample_text(rng, corpus, args.max_text_length)
            if not text or (
                charset is not None and any(c not in charset for c in text)
            ):
                continue
            font_path = rng.choice(fonts)
            if not coverage.covers(font_path, text):
                rejected += 1
                if rejected > 100 * args.num:
                    sys.exit("fonts do not cover the corpus characters")
                continue
            if mono_fonts and rng.random() < args.typewriter_ratio:
                font_path = rng.choice(mono_fonts)
                if not coverage.covers(font_path, text):
                    continue
                img = augment(render_typewriter_line(text, font_path, rng), rng)
            else:
                img = augment(render_line(text, font_path, rng), rng)
            stem = os.path.splitext(os.path.basename(font_path))[0].replace(" ", "_")
            name = "{}__{}_{:06d}.jpg".format(stem, prefix, written)
            cv2.imwrite(
                os.path.join(img_dir, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
            fout.write("{}/{}\t{}\n".format(args.image_subdir, name, text))
            written += 1
    print(
        "{} images from {} fonts -> {} ({} text/font pairs rejected: missing glyphs)".format(
            written, len(fonts), label_path, rejected
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
