# Kaggle GPU kernels used for the Ukrainian OCR models

Scripts that ran on a free Kaggle T4 GPU (see `docs/ukrainian_ocr.md`):

| Script | Model | Inputs (private Kaggle datasets) |
|---|---|---|
| `train_rec_v6.py` | handwritten recognizer v6 (12 epochs from v5) | code (`ppocr/`, `tools/`, `configs/`), `uk_handwritten/` crops + labels, `weights/uk_handwritten_v5_best.pdparams` |
| `train_rec_v7.py` | handwritten recognizer v7 (6 epochs from v6) | code, `uk_handwritten_v7/` crops cut by the fine-tuned detector + labels, `weights/uk_handwritten_v6_best.pdparams` |
| `train_det_server.py` | `PP-OCRv5_server_det` fine-tuned on RUKOPYS pages | code, detection labels (`det_train_refined_{train,val}.txt`); pages are downloaded from Hugging Face inside the kernel |
| `train_rec_printed_v2.py` | printed recognizer v2 (8 epochs from printed v1) | code, one tar `*.tar.bin` with `uk_printed_v2/` (synthetic incl. typewriter style + real RUKOPYS train printed lines) and `weights/uk_printed_v1_best.pdparams` |
| `train_rec_v8.py` | handwritten recognizer v8 (from v7, + filtered RUKOPYS silver) | code, `uk_handwritten_v7/`, one tar `*.tar.bin` with `uk_handwritten_silver/` and `weights/uk_handwritten_v7_best.pdparams` |
| `train_rec_v8b.py` | handwritten recognizer v8b: one more epoch from v8 | as v8 + `uk_handwritten_v8_best.pdparams`; kernel source `uk-ocr-paddle-wheel` |
| `train_rec_v9.py` | handwritten recognizer v9 (3 epochs from v8, train x2 + all selected silver) | as v8 + `uk-ocr-silver2-blob`; kernel source `uk-ocr-paddle-wheel` |
| `train_det_server_960.py` | server detector re-trained on pages downscaled to <= 1280 px (run 18.10: no gain in the pipeline) | code, detection labels, `weights/uk_det_server_v1.pdparams`; kernel source `uk-ocr-paddle-wheel` |
| `get_paddle_wheel.py` | CPU-only kernel: downloads the `paddlepaddle-gpu` wheel (3 min, no GPU quota) | — |

Run with the Kaggle CLI (token in `~/.kaggle/access_token`, never in the repo):

```bash
mkdir kernel && cp tools/ukrainian/kaggle/train_rec_v7.py kernel/train_rec.py
cp tools/ukrainian/kaggle/kernel-metadata.rec_v7.json kernel/kernel-metadata.json   # edit id / datasets
kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
kaggle kernels status <user>/<kernel>
kaggle kernels output <user>/<kernel> -p output/kaggle_run
```

Notes: Kaggle extracts uploaded archives in datasets; numbers passed to
`tools/train.py -o` must be decimal (`0.00007`, YAML reads `7e-05` as a string);
batch 10 fits a T4 for the handwritten recognizer, batch 8 for the server
detector; outputs are saved only when the kernel finishes normally, so every
script limits its training time.

* A dataset of ~15 000 separate small files was never processed by Kaggle
  (status stayed unavailable for > 1 h). Upload one uncompressed tar with a
  `.tar.bin` extension instead (it is not auto-extracted) and extract it in the
  kernel into `/tmp`; everything under `/kaggle/working` becomes kernel output.
* Pushing a new version does **not** stop a running one (at most 2 GPU sessions)
  and `kaggle kernels output` returns only the latest version: never push over a
  running version; delete the kernel and push a new slug instead.
* Installing `paddlepaddle-gpu` from the Paddle mirror took up to ~20 min of
  GPU time (1.9 GB at ~1 MB/s): download the wheel once in a CPU kernel
  (`get_paddle_wheel.py`) and attach that kernel as `kernel_sources`.
* Do not pipe training through `| tee` together with `timeout`: after the
  timeout the data loader workers keep the pipe open and the kernel never
  finishes (it kept running ~11 h). Redirect to a file and `pkill` leftovers.
  Time of a deleted kernel was not charged to the quota.
* Check the weekly GPU quota before long runs (it is not always 30 h):
  `kagglesdk` `get_accelerator_quota_statistics`.
