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

"""Build ``ppocr/utils/dict/ppocrv5_uk_dict.txt`` from the eslav dictionary.

The Ukrainian dictionary is an order preserving subset of
``ppocrv5_eslav_dict.txt`` (the dictionary of ``eslav_PP-OCRv5_mobile_rec``),
so all output-layer rows of the pretrained eslav model can be transferred.

Usage:
    python tools/ukrainian/build_uk_dict.py
    python tools/ukrainian/build_uk_dict.py --check   # fail if the file is stale
"""

import argparse
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from ppocr.utils.ukrainian import (  # noqa: E402
    build_uk_char_list,
    read_char_dict,
    validate_char_dict,
)

DICT_DIR = os.path.join(__dir__, "..", "..", "ppocr", "utils", "dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base_dict", default=os.path.join(DICT_DIR, "ppocrv5_eslav_dict.txt")
    )
    parser.add_argument(
        "--output", default=os.path.join(DICT_DIR, "ppocrv5_uk_dict.txt")
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write, exit with code 1 if --output differs",
    )
    args = parser.parse_args()

    base = read_char_dict(args.base_dict)
    problems = validate_char_dict(base)
    if problems["duplicates"]:
        sys.exit("base dict has duplicates: {}".format(problems["duplicates"]))
    chars = build_uk_char_list(base)
    content = "\n".join(chars) + "\n"

    if args.check:
        with open(args.output, "r", encoding="utf-8", newline="") as f:
            if f.read() != content:
                sys.exit("{} is out of date".format(args.output))
        print(
            "{} is up to date ({} chars)".format(
                os.path.normpath(args.output), len(chars)
            )
        )
        return

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print("wrote {} chars to {}".format(len(chars), os.path.normpath(args.output)))


if __name__ == "__main__":
    main()
