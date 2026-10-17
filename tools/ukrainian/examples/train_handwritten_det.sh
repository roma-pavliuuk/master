#!/bin/bash
# Fine-tune the MOBILE text detector on handwritten RUKOPYS pages (CPU, ~6 h for 10 epochs),
# then evaluate the full pipeline (fine-tuned detector + v5 recognizer) on the 40 test pages.
# Re-running the script resumes from the last finished epoch (checkpoint "latest").
# Extra arguments are passed to tools/train.py -o, e.g.
#   EPOCHS=2 bash tools/ukrainian/examples/train_handwritten_det.sh Optimizer.lr.warmup_epoch=0.2
# Requires: bash tools/ukrainian/examples/train_handwritten_det_prepare.sh (done once).
set -e
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=16 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
PY=.venv/bin/python
OUT=./output/uk_handwritten_det_mobile
EPOCHS=${EPOCHS:-10}
D=train_data/rukopys_raw/train
[ -s $D/det_train_refined_train.txt ] || { echo "run tools/ukrainian/examples/train_handwritten_det_prepare.sh first"; exit 1; }

RESUME=()
[ -f $OUT/latest/latest.pdparams ] && RESUME=(Global.checkpoints=$OUT/latest/latest)

$PY tools/train.py -c configs/det/PP-OCRv5/uk_handwritten_PP-OCRv5_mobile_det.yml -o \
  Global.use_gpu=False Global.epoch_num=$EPOCHS Global.eval_batch_step=[0,320] Global.print_batch_step=10 \
  Global.save_model_dir=$OUT Global.uniform_output_enabled=True \
  Global.pretrained_model=./pretrain_models/PP-OCRv5_mobile_det_pretrained \
  Train.loader.num_workers=6 "${RESUME[@]}" "$@" >> output/uk_handwritten_det_mobile.log 2>&1

# full pipeline on the 40 handwritten test pages (official server det + v5: page CER 38.9 %)
$PY tools/ukrainian/eval_pipeline.py --mode handwritten \
  --det_label train_data/rukopys_raw/test/det_test.txt --data_dir train_data/rukopys_raw/test \
  --det_model_name PP-OCRv5_mobile_det --det_model_dir $OUT/best_accuracy/inference \
  --rec_model_dir output/uk_handwritten_rec_cpu_v5/best_accuracy/inference \
  --output output/eval/pipeline_handwritten_v5_finetuned_mobile_det.json
echo "DONE"
