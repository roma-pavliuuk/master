"""v8b: one more epoch from the v8 best weights (dataset uk-ocr-v8-weights), same data.

Kaggle GPU kernel: fine-tune the Ukrainian HANDWRITTEN recognizer v7 -> v8 on the v7 training lines
(manually annotated RUKOPYS train) plus RUKOPYS silver lines whose automatic transcription agrees with
the v7 prediction (tools/ukrainian/filter_silver.py).

Inputs (private datasets): uk-ocr-code, uk-ocr-handwritten-data-v7 (uk_handwritten_v7/{images,
train_wh.txt,val_det.txt,test_det.txt}), uk-ocr-silver-blob (one uncompressed tar *.tar.bin with
uk_handwritten_silver/{images,silver_train_wh.txt} and weights/uk_handwritten_v7_best.pdparams; a
dataset of tens of thousands of separate files is not processed by Kaggle in reasonable time).
The GPU quota left this week is ~2.6 h: training is capped at TRAIN_HOURS and the number of epochs
is chosen so that the cosine schedule ends inside that budget.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # single GPU (2-GPU check failed in the probe)

ITERS_BUDGET = 0  # v8b: exactly one more epoch (~33 min) - only ~70 min of GPU quota left
BATCH = 10  # batch 24 with AMP and batch 32/64 in fp32 ran out of the 15 GB of a T4
LR = 5e-5
TRAIN_HOURS = 0.62  # one epoch ~30 min + 2 evaluations; the session must end inside the weekly quota
WORK = "/kaggle/working"
OUT = os.path.join(WORK, "output", "uk_handwritten_rec_gpu_v8b")


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


t0 = time.time()
# not extracted into /kaggle/working: everything there becomes kernel output
for blob in glob.glob("/kaggle/input/**/*.tar.bin", recursive=True):
    os.makedirs("/tmp/blob", exist_ok=True)
    print("extracting", blob, flush=True)
    subprocess.check_call(["tar", "-xf", blob, "-C", "/tmp/blob"])
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
# Kaggle extracts uploaded archives: the dataset holds ppocr/, tools/, configs/
src_repo = os.path.dirname(os.path.dirname(find("tools/train.py")))
repo = os.path.join(WORK, "repo")
shutil.copytree(src_repo, repo, dirs_exist_ok=True)
print("code from", src_repo, flush=True)
# the wheel is downloaded by the CPU kernel uk-ocr-paddle-wheel (attached as kernel source): the
# Paddle mirror gives ~1 MB/s, i.e. ~20 min of GPU quota for the 1.9 GB wheel
wheels = glob.glob("/kaggle/input/**/paddlepaddle_gpu-3.2.0-*.whl", recursive=True)
if wheels:
    sh(sys.executable + " -m pip install -q " + wheels[0])
else:
    sh(sys.executable + " -m pip install -q paddlepaddle-gpu==3.2.0 "
       "-i https://www.paddlepaddle.org.cn/packages/stable/cu126/")
print("setup done, min", round((time.time() - t0) / 60, 1), flush=True)
sh(sys.executable + " -m pip install -q -r {}/requirements.txt".format(repo))

v7_dir = os.path.dirname(find("uk_handwritten_v7/train_wh.txt"))
silver_dir = os.path.dirname(find("uk_handwritten_silver/silver_train_wh.txt"))
init = find("uk_handwritten_v8_best.pdparams")[: -len(".pdparams")]
# one data_dir for both sources: images/ (v7) and silver_images/ (silver)
data_dir = "/tmp/v8data"
os.makedirs(data_dir, exist_ok=True)
for name, target in (("images", v7_dir + "/images"), ("silver_images", silver_dir + "/images")):
    if not os.path.exists(os.path.join(data_dir, name)):
        os.symlink(target, os.path.join(data_dir, name))
for name in ("val_det.txt", "test_det.txt"):
    shutil.copy(os.path.join(v7_dir, name), data_dir)
with open(os.path.join(data_dir, "train_wh.txt"), "w", encoding="utf-8") as fo:
    for line in open(os.path.join(v7_dir, "train_wh.txt"), encoding="utf-8"):
        fo.write(line)
    for line in open(os.path.join(silver_dir, "silver_train_wh.txt"), encoding="utf-8"):
        fo.write("silver_" + line)  # images/... -> silver_images/...

# resume from a previous run of this kernel (attached as kernel source), if any
prev = sorted(glob.glob("/kaggle/input/**/uk_handwritten_rec_gpu_v8b/latest/latest.pdparams", recursive=True))
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
n_train = sum(1 for _ in open(os.path.join(data_dir, "train_wh.txt"), encoding="utf-8"))
iters_per_epoch = n_train // BATCH
EPOCHS = max(1, ITERS_BUDGET // iters_per_epoch)  # -> 1
print("train lines", n_train, "iters/epoch", iters_per_epoch, "epochs", EPOCHS, flush=True)
cmd = [
    sys.executable, "tools/train.py",
    "-c", "configs/rec/PP-OCRv5/multi_language/uk_handwritten_PP-OCRv5_mobile_rec.yml",
    "-o",
    "Global.use_gpu=True",
    "Global.epoch_num={}".format(EPOCHS),
    "Global.eval_batch_step=[0,{}]".format(iters_per_epoch // 2),
    "Global.print_batch_step=50",
    "Global.save_model_dir={}".format(OUT),
    "Global.pretrained_model={}".format(init),
    "Optimizer.lr.learning_rate={:.8f}".format(LR),  # decimal: YAML reads 7e-05 as a string
    "Optimizer.lr.warmup_epoch=0.1",
    "Train.dataset.data_dir={}/".format(data_dir),
    "Train.dataset.label_file_list=[{}/train_wh.txt]".format(data_dir),
    "Train.sampler.first_bs={}".format(BATCH),
    "Train.loader.num_workers=4",
    "Eval.dataset.data_dir={}/".format(data_dir),
    "Eval.dataset.label_file_list=[{}/val_det.txt]".format(data_dir),
    "Eval.loader.num_workers=2",
] + extra
log = os.path.join(WORK, "train_rec.log")
# no "| tee": after a timeout the data loader workers survive the trainer, keep the pipe open and
# the kernel hangs forever (lost v8b run); write the log file directly and kill leftovers
rc = subprocess.call("timeout -k 60 {}s ".format(int(TRAIN_HOURS * 3600)) + " ".join("'{}'".format(c) if "[" in c else c for c in cmd) + " > " + log + " 2>&1", shell=True)
subprocess.call("pkill -9 -f 'tools/trai[n].py'", shell=True)
subprocess.call("grep -E 'best metric|Error|error' {} | tail -5".format(log), shell=True)
print("train exit", rc, "elapsed min", round((time.time() - t0) / 60, 1), flush=True)

# evaluate the best model on both test sets (same protocol as locally)
best = os.path.join(OUT, "best_accuracy", "inference")
if os.path.isdir(best):
    for name in ("test_det",):
        sh("{} tools/eval.py -c configs/rec/PP-OCRv5/multi_language/uk_handwritten_PP-OCRv5_mobile_rec.yml "
           "-o Global.use_gpu=True Global.pretrained_model={}/best_accuracy/best_accuracy "
           "Eval.dataset.data_dir={}/ 'Eval.dataset.label_file_list=[{}/{}.txt]' > {} 2>&1"
           .format(sys.executable, OUT, data_dir, data_dir, name, os.path.join(WORK, "eval_" + name + ".log")), check=False)
        subprocess.call("pkill -9 -f 'tools/eva[l].py'", shell=True)
shutil.rmtree(repo, ignore_errors=True)  # keep only outputs in /kaggle/working
print("total min", round((time.time() - t0) / 60, 1))
