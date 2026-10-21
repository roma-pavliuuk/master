"""Kaggle GPU kernel: fine-tune PP-OCRv5_server_det on Ukrainian handwritten RUKOPYS pages."""
import glob, os, shutil, subprocess, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
EPOCHS = 30          # ~5.5 min/epoch at batch 8 on a T4 (timing run)
BATCH = 8            # fits in 15 GB (timing run)
LR = 0.0001          # decimal on purpose: YAML reads 1e-04 as a string
TRAIN_HOURS = 6.0    # keep the whole session well below Kaggle limits
WORK = "/kaggle/working"
PAGES = "/tmp/rukopys"  # not saved as kernel output
OUT = os.path.join(WORK, "output", "uk_handwritten_det_server")
T0 = time.time()


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    rc = subprocess.call(cmd, shell=True)
    if check and rc != 0:
        raise SystemExit("command failed ({}): {}".format(rc, cmd))
    return rc


def find(pattern):
    hits = sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True))
    if not hits:
        raise SystemExit("not found: " + pattern)
    return hits[0]


sh(sys.executable + " -m pip install -q paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/")
repo = os.path.join(WORK, "repo")
shutil.copytree(os.path.dirname(os.path.dirname(find("tools/train.py"))), repo, dirs_exist_ok=True)
sh(sys.executable + " -m pip install -q -r {}/requirements.txt huggingface_hub".format(repo))
labels = os.path.dirname(find("det_train_refined_train.txt"))
needed = sorted({l.split("\t", 1)[0] for f in ("det_train_refined_train.txt", "det_train_refined_val.txt")
                 for l in open(os.path.join(labels, f), encoding="utf-8") if l.strip()})
from huggingface_hub import snapshot_download
snapshot_download("UkrainianCatholicUniversity/rukopys", repo_type="dataset", local_dir=PAGES,
                  allow_patterns=["train/" + p for p in needed], max_workers=16)
data_dir = os.path.join(PAGES, "train")
print("pages", sum(os.path.isfile(os.path.join(data_dir, p)) for p in needed), "of", len(needed), flush=True)
sh("wget -q -O {}/server_det.pdparams https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams".format(WORK))

os.chdir(repo)
n_train = sum(1 for l in open(os.path.join(labels, "det_train_refined_train.txt"), encoding="utf-8") if l.strip())
iters_per_epoch = max(1, n_train // BATCH)
# validation pages are 3000x4000 px: without a size limit the server detector
# runs out of GPU memory, so evaluate with the longest side limited to 2048 px
import yaml
cfg = "configs/det/PP-OCRv5/uk_handwritten_PP-OCRv5_server_det_kaggle.yml"
with open("configs/det/PP-OCRv5/uk_handwritten_PP-OCRv5_server_det.yml", encoding="utf-8") as f:
    conf = yaml.safe_load(f)
for op in conf["Eval"]["dataset"]["transforms"]:
    if "DetResizeForTest" in op:
        op["DetResizeForTest"] = {"limit_side_len": 2048, "limit_type": "max"}
with open(cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(conf, f, allow_unicode=True, sort_keys=False)
# baseline: official detector on the validation pages
sh("{} tools/eval.py -c {} -o Global.use_gpu=True Global.pretrained_model={}/server_det "
   "Eval.dataset.data_dir={}/ 'Eval.dataset.label_file_list=[{}/det_train_refined_val.txt]' "
   "2>&1 | tee {}/det_eval_baseline.log".format(sys.executable, cfg, WORK, data_dir, labels, WORK), check=False)
opts = [
    "Global.use_gpu=True", "Global.epoch_num={}".format(EPOCHS),
    "Global.eval_batch_step=[0,{}]".format(iters_per_epoch), "Global.print_batch_step=20",
    "Global.save_model_dir={}".format(OUT), "Global.uniform_output_enabled=True",
    "Global.pretrained_model={}/server_det".format(WORK),
    "Optimizer.lr.learning_rate={:.8f}".format(LR), "Optimizer.lr.warmup_epoch=1",
    "Train.dataset.data_dir={}/".format(data_dir),
    "'Train.dataset.label_file_list=[{}/det_train_refined_train.txt]'".format(labels),
    "Train.loader.batch_size_per_card={}".format(BATCH), "Train.loader.num_workers=4",
    "Eval.dataset.data_dir={}/".format(data_dir),
    "'Eval.dataset.label_file_list=[{}/det_train_refined_val.txt]'".format(labels),
    "Eval.loader.num_workers=2",
]
rc = subprocess.call("timeout {}s {} tools/train.py -c {} -o {} 2>&1 | tee -a {}/det_train.log".format(
    int(TRAIN_HOURS * 3600), sys.executable, cfg, " ".join(opts), WORK), shell=True)
print("train exit", rc, "min", round((time.time() - T0) / 60, 1), flush=True)
shutil.rmtree(repo, ignore_errors=True)
print("total min", round((time.time() - T0) / 60, 1))
