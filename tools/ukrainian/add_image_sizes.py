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

"""Append image width and height to a recognition label file.

``MultiScaleDataSet`` with ``ds_width: true`` groups lines of similar aspect
ratio into one batch and resizes them to ``height * ratio`` instead of padding
every line to the maximum width, which makes training on long handwritten
lines considerably cheaper. It expects ``<image>\\t<text>\\t<w>\\t<h>`` lines.

    python tools/ukrainian/add_image_sizes.py --label_file train.txt \\
        --data_dir train_data/uk_handwritten --output train_wh.txt
"""

import argparse
import os
import sys

from PIL import Image


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    out, skipped = [], 0
    with open(args.label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                skipped += 1
                continue
            try:
                with Image.open(os.path.join(args.data_dir, parts[0])) as img:
                    w, h = img.size
            except OSError:
                skipped += 1
                continue
            out.append("{}\t{}\t{}\t{}".format(parts[0], parts[1], w, h))
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + ("\n" if out else ""))
    print("{} lines written to {}, {} skipped".format(len(out), args.output, skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
