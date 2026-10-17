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

"""Before/after demo on annotated handwritten test pages (for slides / reports).

For every selected page of a detection label file it runs
  * the OFFICIAL pipeline: PaddleOCR(lang="uk") at default settings, and
  * OUR pipeline: fine-tuned detector + recognizer, handwritten settings,
and writes into --output_dir/<page>/:
  official_ocr_res_img.*, ours_ocr_res_img.*   visualizations (boxes + text)
  official.txt, ours.txt, reference.txt         text in reading order
plus summary.md with page CER / WER of both pipelines.

    python tools/ukrainian/examples/make_demo.py \\
        --det_label train_data/rukopys_raw/test/det_test.txt \\
        --data_dir train_data/rukopys_raw/test --num_pages 5 \\
        --rec_model_dir output/uk_handwritten_rec_gpu_v6/best_accuracy/inference \\
        --det_model_dir output/uk_handwritten_det_server/best_accuracy/inference \\
        --output_dir output/demo
"""

import argparse
import json
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../..")))

from uk_ocr_common import (
    build_uk_ocr,
    group_rows,
    lines_to_text,
    result_to_lines,
)  # noqa: E402

from ppocr.utils.ocr_error_rate import cer, normalize_for_eval, wer  # noqa: E402


def reference_text(regions):
    lines = []
    for r in regions:
        text = r.get("transcription", "")
        if text in ("###", "*", ""):
            continue
        xs = [p[0] for p in r["points"]]
        ys = [p[1] for p in r["points"]]
        lines.append(
            {
                "text": text,
                "poly": r["points"],
                "box": [min(xs), min(ys), max(xs), max(ys)],
            }
        )
    return lines_to_text(group_rows(lines))


def flat(text):
    return normalize_for_eval(text.replace("\t", " ").replace("\n", " "))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--det_label", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--rec_model_dir", required=True)
    parser.add_argument("--det_model_dir")
    parser.add_argument("--num_pages", type=int, default=5)
    parser.add_argument("--pages", nargs="*", help="explicit page file names")
    parser.add_argument("--output_dir", default="output/demo")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.det_label, "r", encoding="utf-8") as f:
        pages = [line.rstrip("\n").split("\t", 1) for line in f if line.strip()]
    if args.pages:
        pages = [p for p in pages if os.path.basename(p[0]) in set(args.pages)]
    else:
        pages = pages[: args.num_pages]

    official = build_uk_ocr("printed", device=args.device)  # == PaddleOCR(lang="uk")
    ours = build_uk_ocr(
        "handwritten",
        rec_model_dir=args.rec_model_dir,
        det_model_dir=args.det_model_dir,
        device=args.device,
    )
    rows = []
    for img_rel, label in pages:
        name = os.path.splitext(os.path.basename(img_rel))[0]
        out = os.path.join(args.output_dir, name)
        os.makedirs(out, exist_ok=True)
        ref = reference_text(json.loads(label))
        with open(os.path.join(out, "reference.txt"), "w", encoding="utf-8") as f:
            f.write(ref + "\n")
        row = {"page": name}
        for tag, ocr in (("official", official), ("ours", ours)):
            res = ocr.predict(os.path.join(args.data_dir, img_rel))[0]
            text = lines_to_text(result_to_lines(res))
            with open(os.path.join(out, tag + ".txt"), "w", encoding="utf-8") as f:
                f.write(text + "\n")
            res.save_to_img(out)
            for fn in os.listdir(out):
                if fn.endswith(
                    ("_ocr_res_img.jpg", "_ocr_res_img.png")
                ) and not fn.startswith(("official", "ours")):
                    os.replace(
                        os.path.join(out, fn),
                        os.path.join(
                            out, "{}_ocr_res_img{}".format(tag, os.path.splitext(fn)[1])
                        ),
                    )
            row[tag + "_cer"] = cer(flat(ref), flat(text))
            row[tag + "_wer"] = wer(flat(ref), flat(text))
        rows.append(row)
        print(
            "{page}: official CER {official_cer:.1%} WER {official_wer:.1%} | "
            "ours CER {ours_cer:.1%} WER {ours_wer:.1%}".format(**row)
        )

    with open(os.path.join(args.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(
            "| page | official CER | official WER | ours CER | ours WER |\n|---|---|---|---|---|\n"
        )
        for r in rows:
            f.write(
                "| {page} | {official_cer:.1%} | {official_wer:.1%} | {ours_cer:.1%} | {ours_wer:.1%} |\n".format(
                    **r
                )
            )
    print("written to", args.output_dir)


if __name__ == "__main__":
    main()
