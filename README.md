# MAST

**MAST** (Masked Augmentation for time Series with missing-aware Tokens) is a
self-supervised pretraining method for multivariate time series forecasting.

The pipeline has two stages:

1. **Pretraining** (self-supervised, no labels). Every window is corrupted twice — once in
   the time domain and once in the frequency domain — and the model has to reconstruct the
   clean signal from both views, while the latent of the frequency view is additionally
   trained to predict the latent of the time view.
2. **Downstream forecasting.** The pretrained backbone is reused for prediction, either with
   **linear probing** (backbone frozen, only the head is trained) or with **end-to-end
   finetuning**.

Everything in this repository runs on **ETTh1** with a single command.

---

## Repository layout

```text
MAST/
├── pretrain.py              # stage 1: self-supervised pretraining
├── forecast.py              # stage 2: linear probing or finetuning + test
├── scripts/
│   ├── pretrain_etth1.sh
│   ├── linear_probe_etth1.sh
│   └── finetune_etth1.sh
├── mast/                    # the library
│   ├── model.py             # MAST encoder, pretraining/prediction heads, learnable tokens
│   ├── masking.py           # patching + the time/frequency masking objective
│   ├── learner.py           # training loop and callback hooks
│   ├── pipeline.py          # shared wiring for the two entry points
│   ├── data.py              # ETTh1 dataset and data loaders
│   ├── cli.py               # hyper-parameter definitions shared by both stages
│   ├── metrics.py           # MSE / MAE / RMSE
│   ├── basics.py  utils.py
│   ├── callback/            # core, tracking, one-cycle LR, RevIN
│   └── layers/              # attention, positional encoding, RevIN, building blocks
└── dataset/ETT-small/ETTh1.csv
```

---

## Environment

Tested with:

| package | version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.5.1 (CUDA 12.4) |
| NumPy | 1.26.4 |
| pandas | 2.0.3 |
| scikit-learn | 1.7.2 |

Install the dependencies with:

```bash
pip install -r requirements.txt
```

MAST is plain PyTorch — no `tsai`, no `pytorch-lightning`. CPU works out of the box (the
scripts fall back to it automatically), a GPU is only needed for speed. Use
`--require_cuda` if you would rather fail than silently train on CPU.

---

## Dataset

ETTh1 is the hourly *Electricity Transformer Temperature* dataset: **17,420 hourly steps**
over 7 columns (`date`, `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`).

**This repository already ships the file**, so you can skip the download:

```text
dataset/ETT-small/ETTh1.csv      (2.6 MB, md5 8381763947c85f4be6ac456c508460d6)
```

If you prefer to fetch it yourself, download it from
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) (the ETT datasets live
in its [`dataset/ETT-small`](https://github.com/thuml/Time-Series-Library/tree/main/dataset/ETT-small)
folder):

```bash
mkdir -p dataset/ETT-small
curl -L -o dataset/ETT-small/ETTh1.csv \
  https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTh1.csv
```

If that direct link is unavailable, follow the dataset instructions in the
[tslib README](https://github.com/thuml/Time-Series-Library#datasets) and place the CSV at
the same path.

**The path matters.** `mast/data.py` resolves the dataset relative to the repository root:

```text
<repo root>/dataset/ETT-small/ETTh1.csv
```

If the file is missing, both entry points exit with a `FileNotFoundError` telling you
exactly where it looked.

The split is the standard one used by TSLib and PatchTST — 12 months train, 4 months
validation, 4 months test, with the scaler fitted on the training split only.

---

## Quickstart

Run all commands from the repository root:

```bash
git clone https://github.com/zltututu/MAST.git
cd MAST
pip install -r requirements.txt
```

### 1. Pretraining

```bash
bash scripts/pretrain_etth1.sh
```

which is equivalent to:

```bash
python pretrain.py \
  --context_points 336 \
  --target_points 96 \
  --patch_len 16 \
  --stride 8 \
  --batch_size 64 \
  --n_epochs 20 \
  --lr 1e-3 \
  --patch_mask_ratio 0.2 \
  --point_mask_ratio 0.3 \
  --freq_mask_min 0.0 \
  --freq_mask_max 0.7 \
  --save_dir saved_models \
  --save_name mast_pretrain_etth1
```

Output: `saved_models/mast_pretrain_etth1.pth` and
`saved_models/mast_pretrain_etth1_losses.csv`.

### 2a. Downstream forecasting — linear probing

The backbone is frozen; only the prediction head is trained.

```bash
bash scripts/linear_probe_etth1.sh
```

```bash
python forecast.py \
  --finetune_mode linear_probe \
  --pretrained_model saved_models/mast_pretrain_etth1.pth \
  --context_points 336 --target_points 96 \
  --patch_len 16 --stride 8 \
  --batch_size 64 --n_epochs 20 --lr 1e-3 \
  --save_dir saved_models --save_name mast_linear_probe_etth1
```

### 2b. Downstream forecasting — end-to-end finetuning

The head is trained for `--freeze_epochs` epochs first, then the whole network is unfrozen.

```bash
bash scripts/finetune_etth1.sh
```

```bash
python forecast.py \
  --finetune_mode finetune \
  --pretrained_model saved_models/mast_pretrain_etth1.pth \
  --context_points 336 --target_points 96 \
  --patch_len 16 --stride 8 \
  --batch_size 64 --n_epochs 20 --freeze_epochs 5 --lr 5e-5 \
  --save_dir saved_models --save_name mast_finetune_etth1
```

Both stage-2 commands print the test scores and write
`saved_models/<save_name>_acc.csv`:

```text
[linear_probe] test MSE: 0.391311   MAE: 0.409339   RMSE: 0.625549
[finetune]     test MSE: 0.384749   MAE: 0.405192   RMSE: 0.620281
```

The reported scores always come from the checkpoint with the best validation loss, not
from the last epoch.

---

## Reference results

ETTh1, multivariate (7 variates in, 7 out), look-back 336 → horizon 96, seed 2021,
single NVIDIA L40. Wall-clock for the whole three-command pipeline is under 20 minutes.

| stage | test MSE | test MAE |
| --- | --- | --- |
| linear probing (20 epochs) | 0.3913 | 0.4093 |
| end-to-end finetuning (20 epochs) | 0.3847 | 0.4052 |

Pretraining for 20 epochs takes about 6 minutes and drives the validation loss from
1.026 to 0.618. Longer pretraining keeps improving it, so treat these numbers as a
readily reproducible baseline rather than a tuned result.

---

## Main arguments

Shared by `pretrain.py` and `forecast.py`; the two stages must agree on the architecture
hyper-parameters, otherwise the checkpoint will not line up with the downstream model.

| argument | default | meaning |
| --- | --- | --- |
| `--context_points` | 336 | look-back window length |
| `--target_points` | 96 | forecast horizon |
| `--patch_len` / `--stride` | 16 / 8 | patch size and stride |
| `--batch_size` | 64 | batch size |
| `--n_layers` / `--n_heads` | 3 / 16 | Transformer depth and heads |
| `--d_model` / `--d_ff` | 128 / 256 | model / feed-forward width |
| `--dropout` / `--head_dropout` | 0.2 / 0.2 | dropout rates |
| `--n_epochs` | 20 | epochs |
| `--seed` | 2021 | RNG seed |

Pretraining only:

| argument | default | meaning |
| --- | --- | --- |
| `--patch_mask_ratio` | 0.2 | fraction of patches removed in the time view |
| `--point_mask_ratio` | 0.3 | fraction of points removed inside surviving patches |
| `--freq_mask_min` / `--freq_mask_max` | 0.0 / 0.7 | range the frequency mask ratio is drawn from |
| `--mask_seed` | None | seed for the masking RNG |
| `--lr` | 1e-3 | peak learning rate |

Forecasting only:

| argument | default | meaning |
| --- | --- | --- |
| `--pretrained_model` | *required* | checkpoint written by `pretrain.py` |
| `--finetune_mode` | `linear_probe` | `linear_probe` or `finetune` |
| `--freeze_epochs` | 5 | `finetune` only: head-only epochs before unfreezing |
| `--handle_missing` | 1 | substitute NaN inputs with the learned `patch_token` |

Run `python pretrain.py --help` or `python forecast.py --help` for the full list.

---

## How the pretraining objective works

For every window, `TimeFreqSequentialMaskCB` (`mast/masking.py`) builds two corrupted
views and the batch is the concatenation of both:

* **Time view** — `hybrid_masking` removes a fraction of whole patches *and* a fraction of
  the points inside the surviving patches. Every removed position is replaced by the
  learnable `patch_token`, and the same map is fed to the model's
  missing-state-aware patch dropping module, which randomly zeroes out patches whose
  missingness is high.
* **Frequency view** — the time-masked signal has a random subset of its rFFT coefficients
  zeroed (`frequency_masking`), with the ratio drawn per sample from
  `[freq_mask_min, freq_mask_max]`.

The loss combines three terms:

1. **Reconstruction** — MSE between the head output and the clean window, computed only on
   the positions that were masked on purpose.
2. **Latent prediction** — the projected frequency-view latent, conditioned on an embedding
   of the frequency mask, is trained to predict the projected time-view latent (which is
   detached, so the gradient only flows one way).
3. **Covariance regularisation** — keeps the projected latents from collapsing.

Two learnable memory patches are prepended to the sequence during encoding and dropped
again before the head, so the positional encoding shape stays identical between the two
stages.

---

## Acknowledgements

This repository is derived from and would not exist without these projects:

* [PatchTST](https://github.com/yuqinie98/PatchTST) — the channel-independent patching
  Transformer encoder, the training loop / callback design, and the linear-probing
  protocol.
* [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) — the ETTh1
  dataset, its train/val/test split and the data loading conventions.
* [PyPOTS](https://github.com/WenjieDu/PyPOTS) — the missing-value perspective that MAST's
  learnable mask token builds on.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
