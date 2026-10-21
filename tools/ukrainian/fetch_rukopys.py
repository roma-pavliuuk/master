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

"""Download an explicit, bounded subset of the RUKOPYS dataset.

RUKOPYS (Ukrainian Catholic University) is an open dataset of Ukrainian
handwritten pages with line-level boxes and transcriptions:
https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys

License: CC BY 4.0 according to the dataset card; the card also states that
some sources are shared under a data sharing agreement for academic research
and publication. Check the current dataset card before any use, and cite it.

Nothing is downloaded unless this script is run. ``--max_pages`` bounds the
size (pages are phone photos / scans of ~2 MB each; the full test split is
~0.7 GB, train ~2.5 GB, silver is auto-annotated and much larger).

Example:
    python tools/ukrainian/fetch_rukopys.py --split test --max_pages 40 \\
        --output_dir train_data/rukopys_raw
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

BASE_URL = (
    "https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys/resolve/main"
)


def _download(url, dst, timeout=60):
    import requests

    tmp = dst + ".part"
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    os.replace(tmp, dst)


def select_pages(records, max_pages, seed=0, sources=None):
    """Deterministic sample stratified by ``source`` (round robin)."""
    by_source = collections.OrderedDict()
    for rec in records:
        if sources and rec.get("source") not in sources:
            continue
        by_source.setdefault(rec.get("source"), []).append(rec)
    rng = random.Random(seed)
    for recs in by_source.values():
        rng.shuffle(recs)
    selected = []
    while len(selected) < max_pages and any(by_source.values()):
        for recs in by_source.values():
            if recs and len(selected) < max_pages:
                selected.append(recs.pop())
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=["train", "test", "silver"], required=True)
    parser.add_argument("--max_pages", type=int, required=True)
    parser.add_argument("--output_dir", default="train_data/rukopys_raw")
    parser.add_argument(
        "--sources", nargs="*", help="e.g. school dictation university archive"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    split_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
    meta_path = os.path.join(split_dir, "metadata.jsonl")
    if not os.path.isfile(meta_path):
        print("downloading {}".format(meta_path))
        _download("{}/{}/metadata.jsonl".format(BASE_URL, args.split), meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    pages = select_pages(records, args.max_pages, args.seed, args.sources)
    print(
        "RUKOPYS {}: {} pages in metadata, downloading {} (license: see dataset card)".format(
            args.split, len(records), len(pages)
        )
    )
    failed = 0
    for i, rec in enumerate(pages, start=1):
        dst = os.path.join(split_dir, rec["file_name"])
        if os.path.isfile(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            _download("{}/{}/{}".format(BASE_URL, args.split, rec["file_name"]), dst)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("failed {}: {}".format(rec["file_name"], e))
        if i % 10 == 0:
            print("  {}/{}".format(i, len(pages)))
    with open(
        os.path.join(split_dir, "selected_pages.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(rec["file_name"] for rec in pages) + "\n")
    print("done, {} failed".format(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
