# Kaggle GPU kernels used for the Ukrainian OCR models

Scripts that ran on a free Kaggle T4 GPU (see `docs/ukrainian_ocr.md`):

| Script | Model | Inputs (private Kaggle datasets) |
|---|---|---|
| `train_rec_v6.py` | handwritten recognizer v6 (12 epochs from v5) | code (`ppocr/`, `tools/`, `configs/`), `uk_handwritten/` crops + labels, `weights/uk_handwritten_v5_best.pdparams` |
| `train_rec_v7.py` | handwritten recognizer v7 (6 epochs from v6) | code, `uk_handwritten_v7/` crops cut by the fine-tuned detector + labels, `weights/uk_handwritten_v6_best.pdparams` |
| `train_det_server.py` | `PP-OCRv5_server_det` fine-tuned on RUKOPYS pages | code, detection labels (`det_train_refined_{train,val}.txt`); pages are downloaded from Hugging Face inside the kernel |

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
