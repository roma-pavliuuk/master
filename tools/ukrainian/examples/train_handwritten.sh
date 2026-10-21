#!/usr/bin/env bash
# Example 6b: fine-tune the HANDWRITTEN Ukrainian recognition model on real
# handwriting (here: a bounded RUKOPYS subset, CC BY 4.0, see docs/ukrainian_ocr.md).
#   bash tools/ukrainian/examples/train_handwritten.sh
#   USE_GPU=False bash tools/ukrainian/examples/train_handwritten.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
CONFIG=configs/rec/PP-OCRv5/multi_language/uk_handwritten_PP-OCRv5_mobile_rec.yml
USE_GPU=${USE_GPU:-True}
TRAIN_PAGES=${TRAIN_PAGES:-1330}  # whole train split, ~3 GB
TEST_PAGES=${TEST_PAGES:-40}

# 1. download a bounded subset (skipped for files that already exist)
python tools/ukrainian/fetch_rukopys.py --split train --max_pages "$TRAIN_PAGES" --output_dir train_data/rukopys_raw
python tools/ukrainian/fetch_rukopys.py --split test --max_pages "$TEST_PAGES" --output_dir train_data/rukopys_raw

# 2. pages -> line crops cut from text-detector polygons (same crops as at
#    inference); the official test split stays the test set
python tools/ukrainian/convert_rukopys.py --split_dir train_data/rukopys_raw/train --crop_source det --gt_fallback \
    --output_dir train_data/uk_handwritten --name train_all --det_label train_data/rukopys_raw/train/det_train.txt
python tools/ukrainian/convert_rukopys.py --split_dir train_data/rukopys_raw/test --crop_source det \
    --output_dir train_data/uk_handwritten --name test --det_label train_data/rukopys_raw/test/det_test.txt
# 3. train/val split by page (no lines of one page in both)
python tools/ukrainian/split_dataset.py --label_file train_data/uk_handwritten/train_all.txt \
    --output_dir train_data/uk_handwritten --names train val --ratios 0.95 0.05 --group_by prefix
python tools/ukrainian/check_dataset.py --data_dir train_data/uk_handwritten --max_text_length 75 \
    --label_file train_data/uk_handwritten/train.txt train_data/uk_handwritten/val.txt train_data/uk_handwritten/test.txt || true
# 4. width/height columns for aspect-ratio aware batches (ds_width)
python tools/ukrainian/add_image_sizes.py --label_file train_data/uk_handwritten/train.txt \
    --data_dir train_data/uk_handwritten --output train_data/uk_handwritten/train_wh.txt

# 5. fine-tune (resume: -o Global.checkpoints=./output/uk_handwritten_rec_ppocrv5/latest)
python tools/train.py -c "$CONFIG" -o Global.use_gpu="$USE_GPU" "$@"
