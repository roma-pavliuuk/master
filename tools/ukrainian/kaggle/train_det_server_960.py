"""Kaggle GPU kernel: re-train the fine-tuned PP-OCRv5_server_det at the INFERENCE scale.

At inference handwritten pages are detected at 960 px (longest side). Here pages are
pre-scaled to 1280 px and randomly resized to 768-1280 px before 640x640 crops, so the
handwriting size seen in training matches inference; validation runs at 960 px."""
import glob, os, shutil, subprocess, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
EPOCHS = 30          # pages are small (<= 1280 px): faster than the full-size run
BATCH = 8            # fits in 15 GB (timing run)
LR = 0.00005         # decimal on purpose: YAML reads 5e-05 as a string
TRAIN_HOURS = 6.0    # keep the whole session well below Kaggle limits
WORK = "/kaggle/working"
PAGES = "/tmp/rukopys"  # not saved as kernel output
OUT = os.path.join(WORK, "output", "uk_handwritten_det_server_960")
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
# start from the detector fine-tuned at full resolution
init_weights = find("uk_det_server_v1.pdparams")[: -len(".pdparams")]

# pre-scale pages to <= 1280 px and rewrite the polygon labels
import json
import cv2
SMALL = "/tmp/rukopys1280"
for f in ("det_train_refined_train.txt", "det_train_refined_val.txt"):
    out_lines = []
    for line in open(os.path.join(labels, f), encoding="utf-8"):
        if not line.strip():
            continue
        rel, js = line.rstrip("\n").split("\t", 1)
        img = cv2.imread(os.path.join(data_dir, rel))
        if img is None:
            continue
        scale = min(1.0, 1280.0 / max(img.shape[:2]))
        dst = os.path.join(SMALL, rel)
        if not os.path.isfile(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst, small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        boxes = json.loads(js)
        for b in boxes:
            b["points"] = [[round(x * scale, 1), round(y * scale, 1)] for x, y in b["points"]]
        out_lines.append(rel + "\t" + json.dumps(boxes, ensure_ascii=False) + "\n")
    with open(os.path.join(WORK, f), "w", encoding="utf-8") as fo:
        fo.writelines(out_lines)
    print(f, len(out_lines), "pages pre-scaled", flush=True)
data_dir, labels = SMALL, WORK

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
        op["DetResizeForTest"] = {"limit_side_len": 960, "limit_type": "max"}
for op in conf["Train"]["dataset"]["transforms"]:
    if "IaaAugment" in op:
        for aug in op["IaaAugment"]["augmenter_args"]:
            if aug["type"] == "Resize":
                aug["args"]["size"] = [0.6, 1.0]  # pages are pre-scaled to 1280 px
with open(cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(conf, f, allow_unicode=True, sort_keys=False)
# baseline: official detector on the validation pages
sh("{} tools/eval.py -c {} -o Global.use_gpu=True Global.pretrained_model={} "
   "Eval.dataset.data_dir={}/ 'Eval.dataset.label_file_list=[{}/det_train_refined_val.txt]' "
   "2>&1 | tee {}/det_eval_baseline.log".format(sys.executable, cfg, init_weights, data_dir, labels, WORK), check=False)
opts = [
    "Global.use_gpu=True", "Global.epoch_num={}".format(EPOCHS),
    "Global.eval_batch_step=[0,{}]".format(iters_per_epoch), "Global.print_batch_step=20",
    "Global.save_model_dir={}".format(OUT), "Global.uniform_output_enabled=True",
    "Global.pretrained_model={}".format(init_weights),
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
