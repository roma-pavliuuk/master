#!/usr/bin/env bash
# Push v9 stages one after another; each stage must COMPLETE within 3 h, otherwise stop.
cd /home/roman/master-diplom/paddle/PaddleOCR/output/kaggle_stage
K=/home/roman/.kaggle-venv/bin/kaggle
for stage in 1 2 3; do
  slug=prozaik/uk-ocr-train-hw-rec-v9-e$stage
  st=$($K kernels status $slug 2>&1 | tail -1)
  if echo "$st" | grep -q COMPLETE; then echo "$(date '+%F %T') e$stage already complete"; continue; fi
  echo "$(date '+%F %T') push e$stage: $($K kernels push -p kernel_rec_v9_e$stage --accelerator NvidiaTeslaT4 2>&1 | tail -1)"
  start=$(date +%s); last=""
  while true; do
    sleep 120
    s=$($K kernels status $slug 2>&1 | tail -1 | sed 's/.*KernelWorkerStatus\.//; s/"//g')
    [ "$s" != "$last" ] && { echo "$(date '+%F %T') e$stage $s"; last=$s; }
    case "$s" in
      COMPLETE) break;;
      ERROR|CANCEL*) echo "stage $stage failed"; exit 1;;
      RUNNING) [ -z "${run_start:-}" ] && run_start=$(date +%s)
               if [ $(( $(date +%s) - run_start )) -gt 10800 ]; then echo "$(date '+%F %T') e$stage running > 3 h: stop chain"; exit 2; fi;;
    esac
  done
  unset run_start
  mkdir -p ../kaggle_v9_e${stage}_out
  $K kernels output $slug -p ../kaggle_v9_e${stage}_out -q
  grep -E "best metric" ../kaggle_v9_e${stage}_out/train_rec.log | tail -1 | sed 's/is_float16.*cer:/cer:/'
  sleep 60  # let Kaggle publish the output as a kernel source
done
echo "$(date '+%F %T') DONE"
