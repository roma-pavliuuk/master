#!/bin/bash
# Prepare data for handwritten text DETECTION fine-tuning (~2 h on CPU; can be
# interrupted at any time - a re-run continues from *.partial label files).
# 1) refined line polygons for all downloaded RUKOPYS train pages and test pages
# 2) train/val split by page (val = the 35 pages used for recognition validation)
# 3) baseline: full pipeline with the OFFICIAL mobile detector + v5 recognizer
set -e
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=16 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
PY=.venv/bin/python
D=train_data/rukopys_raw
if [ ! -s $D/train/det_train_refined.txt ]; then
  $PY tools/ukrainian/convert_rukopys.py --split_dir $D/train --output_dir output/tmp_det_prep --name unused \
      --crop_source det --refine_det_label --no_rec --resume --det_label $D/train/det_train_refined.txt
fi
if [ ! -s $D/test/det_test_refined.txt ]; then
  $PY tools/ukrainian/convert_rukopys.py --split_dir $D/test --output_dir output/tmp_det_prep --name unused \
      --crop_source det --refine_det_label --no_rec --resume --det_label $D/test/det_test_refined.txt
fi
$PY - <<'PY'
d = "train_data/rukopys_raw/train/"
page = lambda path: path.split("/")[-1].split("__")[0].rsplit(".", 1)[0]
val_pages = {l.split("\t")[0].split("/")[-1].split("__")[0]
             for l in open("train_data/uk_handwritten/val_det.txt", encoding="utf-8") if l.strip()}
lines = [l for l in open(d + "det_train_refined.txt", encoding="utf-8") if l.strip()]
# pages without any text line are useless for detection training
lines = [l for l in lines if '"points"' in l]
tr = [l for l in lines if page(l.split("\t")[0]) not in val_pages]
va = [l for l in lines if page(l.split("\t")[0]) in val_pages]
open(d + "det_train_refined_train.txt", "w", encoding="utf-8").writelines(tr)
open(d + "det_train_refined_val.txt", "w", encoding="utf-8").writelines(va)
print("det train pages", len(tr), "val pages", len(va))
PY
$PY tools/ukrainian/check_dataset.py --mode det --data_dir $D/train \
    --label_file $D/train/det_train_refined_train.txt $D/train/det_train_refined_val.txt --skip_images || true
# baseline: official mobile detector in the pipeline (server detector baseline: 38.9 % page CER)
$PY tools/ukrainian/eval_pipeline.py --mode handwritten \
  --det_label train_data/rukopys_raw/test/det_test.txt --data_dir train_data/rukopys_raw/test \
  --det_model_name PP-OCRv5_mobile_det \
  --rec_model_dir output/uk_handwritten_rec_cpu_v5/best_accuracy/inference \
  --output output/eval/pipeline_handwritten_v5_official_mobile_det.json
echo "PREPARE DONE"
