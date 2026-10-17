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

"""Render a synthetic PRINTED Ukrainian document page with known ground truth.

The page contains a title, a paragraph, a table with numbers and dates and
a single line, and is saved as PNG, as a 2-page PDF (second page rotated by a
small angle with uneven illumination) and as a JSON file with the line texts
and boxes. It is used by the examples and by the end-to-end pipeline check.

    python tools/ukrainian/examples/make_sample_document.py --output_dir output/uk_demo
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
]

TITLE = "Довідка про реєстрацію"
PARAGRAPH = [
    "Видана громадянці Ґонті Єлизаветі Іванівні про те,",
    "що вона зареєстрована за адресою: м. Київ,",
    "вул. Їжакевича, буд. 12, кв. 45, під'їзд 2.",
    "Об'єктно-орієнтоване програмування вивчала",
    "у 2019–2023 рр. в Українському університеті.",
]
TABLE = [
    ["№", "Документ", "Дата", "Сума, грн"],
    ["1", "Заява", "24.08.2024", "1 250,50"],
    ["2", "Квитанція", "01.09.2024", "499,00"],
    ["3", "Довідка", "15.03.2025", "0,00"],
]
FOOTER = "Їхній під’їзд відремонтовано. Україна, Європа."


def find_font(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("no font found, pass --font")


def render_page(font_path, width=1240, height=1754):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    lines = []

    def put(text, xy, size):
        font = ImageFont.truetype(font_path, size)
        draw.text(xy, text, font=font, fill=(20, 20, 20))
        l, t, r, b = draw.textbbox(xy, text, font=font)
        # annotation-style margin around the glyphs (detectors return padded boxes)
        pad = int(0.25 * (b - t))
        lines.append({"text": text, "box": [l - pad, t - pad, r + pad, b + pad]})

    y = 120
    put(TITLE, (360, y), 44)
    y += 110
    for line in PARAGRAPH:
        put(line, (120, y), 32)
        y += 56
    y += 50
    col_x = [120, 220, 560, 820, 1120]
    row_h = 64
    for r, row in enumerate(TABLE):
        for c, cell in enumerate(row):
            put(cell, (col_x[c] + 16, y + 14), 28)
        for c in range(len(col_x)):
            draw.line([(col_x[c], y), (col_x[c], y + row_h)], fill="black", width=2)
        draw.line([(col_x[0], y), (col_x[-1], y)], fill="black", width=2)
        y += row_h
    draw.line([(col_x[0], y), (col_x[-1], y)], fill="black", width=2)
    y += 90
    put(FOOTER, (120, y), 30)
    return img, lines


def degrade(img, angle=2.0):
    """Small skew, uneven illumination and noise, as in a phone photo."""
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor="white")
    arr = np.asarray(rotated).astype(np.float32)
    h, w = arr.shape[:2]
    gx = np.linspace(1.0, 0.6, w, dtype=np.float32)[None, :, None]
    gy = np.linspace(0.95, 1.0, h, dtype=np.float32)[:, None, None]
    arr = arr * gx * gy + np.random.default_rng(0).normal(0, 6, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="output/uk_demo")
    parser.add_argument("--font", default=None)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    img, lines = render_page(args.font or find_font(DEFAULT_FONTS))
    png = os.path.join(args.output_dir, "uk_printed_document.png")
    img.save(png)
    photo = degrade(img)
    photo.save(
        os.path.join(args.output_dir, "uk_printed_document_photo.jpg"), quality=85
    )
    pdf = os.path.join(args.output_dir, "uk_printed_document.pdf")
    img.save(pdf, save_all=True, append_images=[photo], resolution=150)
    with open(
        os.path.join(args.output_dir, "uk_printed_document.json"), "w", encoding="utf-8"
    ) as f:
        json.dump({"lines": lines}, f, ensure_ascii=False, indent=2)
    # PaddleOCR detection label format, usable by tools/ukrainian/eval_pipeline.py
    boxes = [
        {"transcription": ln["text"], "points": [[l, t], [r, t], [r, b], [l, b]]}
        for ln in lines
        for l, t, r, b in [ln["box"]]
    ]
    with open(
        os.path.join(args.output_dir, "uk_printed_document_det.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "uk_printed_document.png\t{}\n".format(
                json.dumps(boxes, ensure_ascii=False)
            )
        )
    print(
        "written: {}, {}, photo jpg, json and det label ({} lines)".format(
            png, pdf, len(lines)
        )
    )


if __name__ == "__main__":
    main()
