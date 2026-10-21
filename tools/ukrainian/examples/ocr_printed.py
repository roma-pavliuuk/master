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

"""Example 1: recognize PRINTED Ukrainian text (photo, scan or PDF).

    # official models (PaddleOCR(lang="uk"))
    python tools/ukrainian/examples/ocr_printed.py output/uk_demo/uk_printed_document.png
    # PDF pages are converted to images by the pipeline
    python tools/ukrainian/examples/ocr_printed.py output/uk_demo/uk_printed_document.pdf
    # fine-tuned Ukrainian printed model
    python tools/ukrainian/examples/ocr_printed.py page.jpg \\
        --rec_model_dir output/uk_printed_rec_ppocrv5/best_accuracy/inference
"""

import argparse

from uk_ocr_common import build_uk_ocr, lines_to_text, result_to_lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="image, PDF or directory")
    parser.add_argument("--rec_model_dir")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ocr = build_uk_ocr("printed", rec_model_dir=args.rec_model_dir, device=args.device)
    for res in ocr.predict(args.input):
        print("=== {} (page {})".format(res["input_path"], res["page_index"]))
        print(lines_to_text(result_to_lines(res)))


if __name__ == "__main__":
    main()
