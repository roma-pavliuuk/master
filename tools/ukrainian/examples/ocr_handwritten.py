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

"""Example 2: recognize HANDWRITTEN Ukrainian text.

Requires a recognition model fine-tuned on real handwriting with
configs/rec/PP-OCRv5/multi_language/uk_handwritten_PP-OCRv5_mobile_rec.yml
(the official weights are trained on printed text only).

    python tools/ukrainian/examples/ocr_handwritten.py page.jpg \\
        --rec_model_dir output/uk_handwritten_rec_ppocrv5/best_accuracy/inference
    # recognize already cropped line images (no detection)
    python tools/ukrainian/examples/ocr_handwritten.py train_data/uk_handwritten/images \\
        --rec_model_dir output/uk_handwritten_rec_ppocrv5/best_accuracy/inference --lines
    # lexicon post-correction (tools/ukrainian/lexicon_correct.py)
    python tools/ukrainian/examples/ocr_handwritten.py page.jpg --rec_model_dir ... \
        --lexicon /usr/share/dict/ukrainian --lexicon_freq train_data/lexicon/uk_50k.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lexicon_correct import add_lexicon_args, corrector_from_args  # noqa: E402

from uk_ocr_common import (  # noqa: E402
    UK_REC_MODEL_NAME,
    build_uk_ocr,
    lines_to_text,
    result_to_lines,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="image, PDF or directory")
    parser.add_argument(
        "--rec_model_dir", help="fine-tuned handwritten recognition model"
    )
    parser.add_argument("--det_model_dir", help="optional fine-tuned detection model")
    parser.add_argument(
        "--det_model_name",
        help="detector architecture of --det_model_dir, e.g. PP-OCRv5_mobile_det "
        "(default PP-OCRv5_server_det)",
    )
    parser.add_argument(
        "--lines", action="store_true", help="inputs are single text-line crops"
    )
    parser.add_argument("--device", default="cpu")
    add_lexicon_args(parser)
    args = parser.parse_args()
    corrector = corrector_from_args(args)
    fix = corrector.correct if corrector else (lambda text: text)

    if args.lines:
        from paddleocr import TextRecognition

        rec = TextRecognition(
            model_name=UK_REC_MODEL_NAME,
            model_dir=args.rec_model_dir,
            device=args.device,
        )
        for res in rec.predict(args.input):
            print(
                "{}\t{}\t{:.3f}".format(
                    res["input_path"], fix(res["rec_text"]), res["rec_score"]
                )
            )
        return

    ocr = build_uk_ocr(
        "handwritten",
        rec_model_dir=args.rec_model_dir,
        det_model_dir=args.det_model_dir,
        det_model_name=args.det_model_name,
        device=args.device,
    )
    for res in ocr.predict(args.input):
        print("=== {}".format(res["input_path"]))
        lines = result_to_lines(res)
        for ln in lines:
            ln["text"] = fix(ln["text"])
        print(lines_to_text(lines))


if __name__ == "__main__":
    main()
