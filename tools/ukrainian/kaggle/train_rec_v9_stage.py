"""Kaggle GPU kernel, one STAGE of v9 (sessions longer than ~3 h hung twice on Kaggle and lost their
output, so v9 is trained as 3 sessions of one epoch each, lr peak 5e-5 -> 3e-5 -> 1.5e-5, every stage
starting from the last weights of the previous one, attached as kernel source).

Ukrainian HANDWRITTEN recognizer v9 = v8 fine-tuned for 3 epochs on the manually annotated RUKOPYS train lines (repeated x2: the only dictation data) plus
all selected RUKOPYS silver lines: 16 261 (pages 1-2400, filtered with v7) + 57 426 (remaining
pages, filtered with v8; tools/ukrainian/filter_silver.py).

Inputs: datasets uk-ocr-code, uk-ocr-handwritten-data-v7, uk-ocr-silver-blob, uk-ocr-silver2-blob
(one uncompressed tar *.tar.bin each), uk-ocr-v8-weights; kernel source uk-ocr-paddle-wheel.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # single GPU (2-GPU check failed in the probe)

STAGE = 1  # 1, 2, 3: one epoch per Kaggle session
EPOCHS = 1  # ~11 300 iterations, ~95 min on a T4 at batch 10
BATCH = 10  # batch 24 with AMP and batch 32/64 in fp32 ran out of the 15 GB of a T4
LR = 0.00005  # stage 2: 0.00003, stage 3: 0.000015
GOLD_REPEAT = 2
TRAIN_HOURS = 2.3  # safety net only: one epoch needs ~1.6 h
WORK = "/kaggle/working"
OUT = os.path.join(WORK, "output", "uk_handwritten_rec_gpu_v9_e{}".format(STAGE))


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
silver2_dir = os.path.dirname(find("uk_handwritten_silver2/silver_train_wh.txt"))
init = (find("uk_handwritten_v8_best.pdparams") if STAGE == 1 else
        find("uk_handwritten_rec_gpu_v9_e{}/latest/latest.pdparams".format(STAGE - 1)))[: -len(".pdparams")]
print("init", init, flush=True)
# one data_dir for all sources: images/ (v7), silver_images/, silver2_images/
data_dir = "/tmp/v9data"
os.makedirs(data_dir, exist_ok=True)
for name, target in (("images", v7_dir + "/images"), ("silver_images", silver_dir + "/images"),
                     ("silver2_images", silver2_dir + "/images")):
    if not os.path.exists(os.path.join(data_dir, name)):
        os.symlink(target, os.path.join(data_dir, name))
for name in ("val_det.txt", "test_det.txt"):
    shutil.copy(os.path.join(v7_dir, name), data_dir)
with open(os.path.join(data_dir, "train_wh.txt"), "w", encoding="utf-8") as fo:
    gold = open(os.path.join(v7_dir, "train_wh.txt"), encoding="utf-8").read()
    fo.write(gold * GOLD_REPEAT)
    for line in open(os.path.join(silver_dir, "silver_train_wh.txt"), encoding="utf-8"):
        fo.write("silver_" + line)  # images/... -> silver_images/...
    for line in open(os.path.join(silver2_dir, "silver_train_wh.txt"), encoding="utf-8"):
        fo.write("silver2_" + line)

extra = []

os.chdir(repo)
n_train = sum(1 for _ in open(os.path.join(data_dir, "train_wh.txt"), encoding="utf-8"))
iters_per_epoch = n_train // BATCH
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
    "Optimizer.lr.warmup_epoch={}".format(0.2 if STAGE == 1 else 0.05),
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
