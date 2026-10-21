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

"""Split a PaddleOCR label file into train / val / test label files.

Records of one group are never split between subsets, which prevents leakage
of the same page, writer or font into the test set:

* ``--group_by none``    every record is its own group
* ``--group_by dir``     parent directory of the image path
* ``--group_by prefix``  file name part before ``--group_sep`` (default "__"),
                         e.g. ``<page_id>__0003.jpg`` written by convert_rukopys.py
                         or ``<font>__000123.jpg`` written by gen_synth_printed.py

Example:
    python tools/ukrainian/split_dataset.py --label_file all.txt \\
        --output_dir train_data/uk_handwritten --ratios 0.8 0.1 0.1 \\
        --group_by prefix --seed 0
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import sys


def group_key(img_path: str, group_by: str, sep: str) -> str:
    if group_by == "none":
        return img_path
    if group_by == "dir":
        return os.path.dirname(img_path)
    name = os.path.basename(img_path)
    return name.split(sep, 1)[0] if sep in name else name


def split_records(records, ratios, group_by="none", sep="__", seed=0):
    """Split ``(img_path, line)`` records into len(ratios) lists by group."""
    if not ratios or any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must be non-negative and sum to > 0")
    groups = collections.OrderedDict()
    for img_path, line in records:
        groups.setdefault(group_key(img_path, group_by, sep), []).append(line)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    total = sum(len(v) for v in groups.values())
    targets = [r / float(sum(ratios)) * total for r in ratios]
    subsets = [[] for _ in ratios]
    idx = 0
    for k in keys:
        # fill subsets in order; the last subset takes the remainder
        while idx < len(subsets) - 1 and len(subsets[idx]) >= targets[idx]:
            idx += 1
        subsets[idx].extend(groups[k])
    return subsets


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label_file", required=True, nargs="+")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--names", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--group_by", choices=["none", "dir", "prefix"], default="none")
    parser.add_argument("--group_sep", default="__")
    parser.add_argument("--delimiter", default="\t")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if len(args.names) != len(args.ratios):
        parser.error("--names and --ratios must have the same length")

    records = []
    for label_file in args.label_file:
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if not line.strip():
                    continue
                records.append((line.split(args.delimiter, 1)[0], line))

    subsets = split_records(
        records, args.ratios, args.group_by, args.group_sep, args.seed
    )
    os.makedirs(args.output_dir, exist_ok=True)
    for name, lines in zip(args.names, subsets):
        out = os.path.join(args.output_dir, name + ".txt")
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        print("{}: {} records -> {}".format(name, len(lines), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
