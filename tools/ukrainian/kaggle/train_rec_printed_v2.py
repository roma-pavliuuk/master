"""Kaggle GPU kernel: Ukrainian PRINTED recognizer v2: printed v1 fine-tuned on synthetic lines
(regular + typewriter style) and real printed lines of the RUKOPYS train split, in the
aspect-ratio aware regime of the handwritten config (lines up to 75 characters).

Inputs (private datasets): uk-ocr-code (paddleocr_uk_code.tar.gz), uk-ocr-handwritten-data
(uk_handwritten/{images,train_v4_wh.txt,val_det.txt,test_det.txt,test.txt}, weights/*.pdparams).
Optional kernel source: a previous version of this kernel -> resume from its output/latest.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # single GPU (2-GPU check failed in the probe)

EPOCHS = int(os.environ.get("EPOCHS", "8"))  # ~12 min/epoch on a T4 at batch 10
BATCH = 10  # batch 24 with AMP and batch 32/64 in fp32 ran out of the 15 GB of a T4
LR = 1e-4
TRAIN_HOURS = 6.5  # Kaggle kills GPU sessions after ~9 h WITHOUT saving /kaggle/working
WORK = "/kaggle/working"
OUT = os.path.join(WORK, "output", "uk_printed_rec_gpu_v2")


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    rc = subprocess.call(cmd, shell=True)
    if check and rc != 0:
        raise SystemExit("command failed ({}): {}".format(rc, cmd))
    return rc


def find(pattern):
    hits = sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True))
    hits += sorted(glob.glob("/tmp/blob/**/" + pattern, recursive=True))
    if not hits:
        raise SystemExit("not found in /kaggle/input: " + pattern)
    return hits[0]


# data is uploaded as one uncompressed tar named *.tar.bin (Kaggle did not
# finish processing the dataset of ~15k separate files)
for blob in glob.glob("/kaggle/input/**/*.tar.bin", recursive=True):
    os.makedirs("/tmp/blob", exist_ok=True)
    print("extracting", blob, flush=True)
    subprocess.check_call(["tar", "-xf", blob, "-C", "/tmp/blob"])


t0 = time.time()
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
# Kaggle extracts uploaded archives: the dataset holds ppocr/, tools/, configs/
src_repo = os.path.dirname(os.path.dirname(find("tools/train.py")))
repo = os.path.join(WORK, "repo")
shutil.copytree(src_repo, repo, dirs_exist_ok=True)
print("code from", src_repo, flush=True)
sh(sys.executable + " -m pip install -q paddlepaddle-gpu==3.2.0 "
   "-i https://www.paddlepaddle.org.cn/packages/stable/cu126/")
sh(sys.executable + " -m pip install -q -r {}/requirements.txt".format(repo))

data_dir = os.path.dirname(find("uk_printed_v2/train_wh.txt"))
init = find("weights/uk_printed_v1_best.pdparams")[: -len(".pdparams")]

# resume from a previous run of this kernel (attached as kernel source), if any
prev = sorted(glob.glob("/kaggle/input/**/uk_printed_rec_gpu_v2/latest/latest.pdparams", recursive=True))
extra = []
if prev:
    os.makedirs(OUT, exist_ok=True)
    src_dir = os.path.dirname(os.path.dirname(prev[0]))
    for sub in ("latest", "best_accuracy"):
        if os.path.isdir(os.path.join(src_dir, sub)):
            shutil.copytree(os.path.join(src_dir, sub), os.path.join(OUT, sub), dirs_exist_ok=True)
    extra.append("Global.checkpoints={}/latest/latest".format(OUT))
    print("resuming from", prev[0], flush=True)

os.chdir(repo)
iters_per_epoch = sum(1 for _ in open(os.path.join(data_dir, "train_wh.txt"), encoding="utf-8")) // BATCH
cmd = [
    sys.executable, "tools/train.py",
    "-c", "configs/rec/PP-OCRv5/multi_language/uk_handwritten_PP-OCRv5_mobile_rec.yml",
    "-o",
    "Global.use_gpu=True",
    "Global.epoch_num={}".format(EPOCHS),
    "Global.eval_batch_step=[0,{}]".format(iters_per_epoch),
    "Global.print_batch_step=50",
    "Global.save_model_dir={}".format(OUT),
    "Global.pretrained_model={}".format(init),
    "Optimizer.lr.learning_rate={:.8f}".format(LR),  # decimal: YAML reads 7e-05 as a string
    "Optimizer.lr.warmup_epoch=0.3",
    "Train.dataset.data_dir={}/".format(data_dir),
    "Train.dataset.label_file_list=[{}/train_wh.txt]".format(data_dir),
    "Train.sampler.first_bs={}".format(BATCH),
    "Train.loader.num_workers=4",
    "Eval.dataset.data_dir={}/".format(data_dir),
    "Eval.dataset.label_file_list=[{}/val.txt]".format(data_dir),
    "Eval.loader.num_workers=2",
] + extra
log = os.path.join(WORK, "train_rec.log")
rc = subprocess.call("timeout {}s ".format(int(TRAIN_HOURS * 3600)) + " ".join("'{}'".format(c) if "[" in c else c for c in cmd) + " 2>&1 | tee -a " + log, shell=True)
print("train exit", rc, "elapsed min", round((time.time() - t0) / 60, 1), flush=True)

# evaluate the best model on both test sets (same protocol as locally)
best = os.path.join(OUT, "best_accuracy", "inference")
if os.path.isdir(best):
    for name in ():  # real printed test is evaluated locally
        sh("{} tools/eval.py -c configs/rec/PP-OCRv5/multi_language/uk_handwritten_PP-OCRv5_mobile_rec.yml "
           "-o Global.use_gpu=True Global.pretrained_model={}/best_accuracy/best_accuracy "
           "Eval.dataset.data_dir={}/ 'Eval.dataset.label_file_list=[{}/{}.txt]' 2>&1 | tail -6 | tee -a {}"
           .format(sys.executable, OUT, data_dir, data_dir, name, os.path.join(WORK, "eval_" + name + ".log")), check=False)
shutil.rmtree(repo, ignore_errors=True)  # keep only outputs in /kaggle/working
print("total min", round((time.time() - t0) / 60, 1))
