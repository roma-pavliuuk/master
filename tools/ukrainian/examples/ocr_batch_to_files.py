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

"""Examples 3-5: process several images, save the text to files and print
the coordinates of every detected text region.

For every input page it writes into --output_dir:
    <name>.txt          recognized text in reading order (table cells: tabs)
    <name>.lines.json   [{"text", "score", "box": [x1, y1, x2, y2], "poly"}]
    <name>_res.json / <name>_ocr_res_img.*   native PaddleOCR result and visualization

    python tools/ukrainian/examples/ocr_batch_to_files.py \\
        output/uk_demo/uk_printed_document.png output/uk_demo/uk_printed_document_photo.jpg \\
        --output_dir output/uk_demo/results
"""

import argparse
import json
import os

from uk_ocr_common import build_uk_ocr, lines_to_text, result_to_lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", help="images, PDFs or directories")
    parser.add_argument("--output_dir", required=True)
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
        "--no_vis", action="store_true", help="do not save visualizations"
    )
    args = parser.parse_args()

    missing = [p for p in args.inputs if not os.path.exists(p)]
    if missing:
        parser.error("inputs not found: {}".format(", ".join(missing)))
    os.makedirs(args.output_dir, exist_ok=True)
    ocr = build_uk_ocr(
        args.mode,
        rec_model_dir=args.rec_model_dir,
        det_model_dir=args.det_model_dir,
        det_model_name=args.det_model_name,
        device=args.device,
    )

    for res in ocr.predict(args.inputs):
        name = os.path.splitext(os.path.basename(res["input_path"]))[0]
        if res["page_index"] is not None:
            name += "_page{}".format(res["page_index"])
        lines = result_to_lines(res)
        with open(
            os.path.join(args.output_dir, name + ".txt"), "w", encoding="utf-8"
        ) as f:
            f.write(lines_to_text(lines) + "\n")
        with open(
            os.path.join(args.output_dir, name + ".lines.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(lines, f, ensure_ascii=False, indent=2)
        res.save_to_json(args.output_dir)
        if not args.no_vis:
            res.save_to_img(args.output_dir)

        print("=== {}: {} text regions".format(name, len(lines)))
        for ln in lines:
            x1, y1, x2, y2 = ln["box"]
            print(
                "  [{:4d},{:4d},{:4d},{:4d}] {:.2f}  {}".format(
                    x1, y1, x2, y2, ln["score"], ln["text"]
                )
            )


if __name__ == "__main__":
    main()
