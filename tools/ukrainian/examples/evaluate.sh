#!/usr/bin/env bash
# Example 7: evaluate models on the independent TEST sets (never used for training).
#   bash tools/ukrainian/examples/evaluate.sh [printed_model_dir] [handwritten_model_dir]
set -euo pipefail
cd "$(dirname "$0")/../../.."
PRINTED_DIR=${1:-output/uk_printed_rec_ppocrv5/best_accuracy/inference}
HANDWRITTEN_DIR=${2:-output/uk_handwritten_rec_ppocrv5/best_accuracy/inference}

# baseline: official eslav_PP-OCRv5_mobile_rec
python tools/ukrainian/eval_rec.py --output_dir output/eval/official \
    --label_file printed=train_data/uk_printed/test.txt handwritten=train_data/uk_handwritten/test.txt

[ -d "$PRINTED_DIR" ] && python tools/ukrainian/eval_rec.py --model_dir "$PRINTED_DIR" \
    --output_dir output/eval/printed_finetuned --label_file printed=train_data/uk_printed/test.txt
[ -d "$HANDWRITTEN_DIR" ] && python tools/ukrainian/eval_rec.py --model_dir "$HANDWRITTEN_DIR" \
    --output_dir output/eval/handwritten_finetuned --label_file handwritten=train_data/uk_handwritten/test.txt

# full pipeline (detection + recognition) on annotated handwritten pages
REC_ARGS=()
[ -d "$HANDWRITTEN_DIR" ] && REC_ARGS=(--rec_model_dir "$HANDWRITTEN_DIR")
python tools/ukrainian/eval_pipeline.py --mode handwritten --max_pages 40 \
    --det_label train_data/rukopys_raw/test/det_test.txt --data_dir train_data/rukopys_raw/test \
    "${REC_ARGS[@]}" --output output/eval/pipeline_handwritten.json
