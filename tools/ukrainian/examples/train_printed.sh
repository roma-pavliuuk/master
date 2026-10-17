#!/usr/bin/env bash
# Example 6a: fine-tune the PRINTED Ukrainian recognition model.
#   bash tools/ukrainian/examples/train_printed.sh            # GPU
#   USE_GPU=False bash tools/ukrainian/examples/train_printed.sh   # CPU (very slow)
# Data: train_data/uk_printed/{train,val}.txt (see docs/ukrainian_ocr.md).
set -euo pipefail
cd "$(dirname "$0")/../../.."
CONFIG=configs/rec/PP-OCRv5/multi_language/uk_PP-OCRv5_mobile_rec.yml
USE_GPU=${USE_GPU:-True}

python tools/ukrainian/check_dataset.py --data_dir train_data/uk_printed \
    --label_file train_data/uk_printed/train.txt train_data/uk_printed/val.txt \
    --max_text_length 25

# resume an interrupted run with: -o Global.checkpoints=./output/uk_printed_rec_ppocrv5/latest
python tools/train.py -c "$CONFIG" -o Global.use_gpu="$USE_GPU" "$@"

# best model is exported to output/uk_printed_rec_ppocrv5/best_accuracy/inference
# (Global.uniform_output_enabled: true); manual export of any checkpoint:
#   python tools/export_model.py -c "$CONFIG" -o Global.use_gpu=False \
#       Global.pretrained_model=./output/uk_printed_rec_ppocrv5/best_accuracy/best_accuracy \
#       Global.save_inference_dir=./output/uk_printed_rec_ppocrv5/inference
