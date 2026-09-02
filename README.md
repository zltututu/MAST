# MAST

**MAST** (Masked Augmentation for time Series with missing-aware Tokens) is a
self-supervised pretraining method for multivariate time series, covering both
**forecasting** and **classification**.

The pipeline has two stages:

1. **Pretraining** (self-supervised, no labels). Every window is corrupted twice — once in
   the time domain and once in the frequency domain — and the model has to reconstruct the
   clean signal from both views, while the latent of the frequency view is additionally
   trained to predict the latent of the time view.
2. **Downstream.** The pretrained backbone is reused either with **linear probing**
   (backbone frozen, only the head is trained) or with **end-to-end finetuning**, for
   forecasting (`forecast.py`) or classification (`classify.py`).

The repository ships with **ETTh1** so you can run the full pipeline out of the box. All
other [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library)
forecasting datasets and the 10 TSLib classification (UEA) datasets are supported as
well — you only have to download them from TSLib and place them under `dataset/` (see
[Datasets](#datasets)).

---

## Repository layout

```text
MAST/
├── pretrain.py              # stage 1: self-supervised pretraining
├── forecast.py              # stage 2: forecasting via linear probing or finetuning
├── classify.py              # stage 2: classification via linear probing or finetuning
├── scripts/
│   ├── pretrain_etth1.sh  linear_probe_etth1.sh  finetune_etth1.sh
│   ├── pretrain_weather.sh  linear_probe_weather.sh  finetune_weather.sh
│   └── pretrain_uea.sh  linear_probe_uea.sh  finetune_uea.sh
├── mast/                    # the library
│   ├── model.py             # MAST encoder, pretraining/prediction/classification heads
│   ├── masking.py           # patching + the time/frequency masking objective
│   ├── learner.py           # training loop and callback hooks
│   ├── pipeline.py          # shared wiring for the entry points
│   ├── data.py              # TSLib forecasting CSVs + UEA classification loaders
│   ├── cli.py               # hyper-parameter definitions shared by all stages
│   ├── metrics.py           # MSE / MAE / RMSE / accuracy / precision / recall / F1
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

MAST is plain PyTorch — no `tsai`, no `pytorch-lightning`, no `sktime`. CPU works out of
the box (the scripts fall back to it automatically), a GPU is only needed for speed. Use
`--require_cuda` if you would rather fail than silently train on CPU.

---

## Datasets

All datasets come from
[Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library). TSLib
distributes the preprocessed archives on
[Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing),
[Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) and
[Hugging Face](https://huggingface.co/datasets/thuml/Time-Series-Library). Download what
you need and place the files under `dataset/` so that the paths below match.

### Forecasting datasets

| dataset | `--data` | `--root_path` | `--data_path` |
| --- | --- | --- | --- |
| ETTh1 (shipped) | `ETTh1` | `dataset/ETT-small` | `ETTh1.csv` |
| ETTh2 | `ETTh2` | `dataset/ETT-small` | `ETTh2.csv` |
| ETTm1 | `ETTm1` | `dataset/ETT-small` | `ETTm1.csv` |
| ETTm2 | `ETTm2` | `dataset/ETT-small` | `ETTm2.csv` |
| weather | `custom` | `dataset/weather` | `weather.csv` |
| electricity | `custom` | `dataset/electricity` | `electricity.csv` |
| traffic | `custom` | `dataset/traffic` | `traffic.csv` |
| exchange_rate | `custom` | `dataset/exchange_rate` | `exchange_rate.csv` |
| illness (ILI) | `custom` | `dataset/illness` | `national_illness.csv` |

ETTh1 already ships with this repository (`2.6 MB`, md5
`8381763947c85f4be6ac456c508460d6`). The remaining ETT files are small and can be fetched
directly:

```bash
curl -L -o dataset/ETT-small/ETTm1.csv \
  https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTm1.csv
```

The four ETT datasets use the standard 12/4/4-month train/val/test split; every
`--data custom` dataset uses a 70/10/20 ratio split. In both cases the scaler is fitted
on the training split only. `--target` is only relevant with `--features S` (univariate);
for the ETT and weather datasets the target column is `OT`.

**The path matters.** `mast/data.py` looks for `<root_path>/<data_path>`. If the file is
missing, the entry point exits with a `FileNotFoundError` telling you exactly where it
looked.

### Classification datasets (UEA)

The 10 UEA classification datasets distributed with TSLib are supported:

`EthanolConcentration`, `FaceDetection`, `Handwriting`, `Heartbeat`, `JapaneseVowels`,
`PEMS-SF`, `SelfRegulationSCP1`, `SelfRegulationSCP2`, `SpokenArabicDigits`,
`UWaveGestureLibrary`.

They are stored in the sktime `.ts` format, one folder per dataset containing
`<Name>_TRAIN.ts` and `<Name>_TEST.ts`. Download them from the TSLib dataset links above
(e.g. the Hugging Face mirror) and place them as:

```text
dataset/<Name>/<Name>_TRAIN.ts
dataset/<Name>/<Name>_TEST.ts
```

For classification, pass `--data UEA --root_path dataset/<Name>` and use `classify.py`.
`--context_points 0` auto-detects the series length (series are padded to the length of
the longest one across both splits).

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

Both stage-2 commands print the test scores and write `saved_models/<save_name>_acc.csv`:

The reported scores always come from the checkpoint with the best validation loss, not
from the last epoch.

### 3. Forecasting on any other TSLib dataset

Point `--data custom` (or the matching ETT flag) at the right folder and file. For
example, weather:

```bash
python pretrain.py \
  --data custom --root_path dataset/weather --data_path weather.csv \
  --context_points 336 --target_points 96 --patch_len 16 --stride 8 \
  --batch_size 64 --n_epochs 20 --lr 1e-3 \
  --patch_mask_ratio 0.2 --point_mask_ratio 0.3 --freq_mask_min 0.0 --freq_mask_max 0.7 \
  --save_dir saved_models --save_name mast_pretrain_weather

python forecast.py \
  --data custom --root_path dataset/weather --data_path weather.csv \
  --finetune_mode linear_probe \
  --pretrained_model saved_models/mast_pretrain_weather.pth \
  --context_points 336 --target_points 96 --patch_len 16 --stride 8 \
  --batch_size 64 --n_epochs 20 --lr 1e-3 \
  --save_dir saved_models --save_name mast_linear_probe_weather
```

The same commands are wrapped in `scripts/{pretrain,linear_probe,finetune}_weather.sh`.

### 4. Classification on a UEA dataset

Classification follows the TSLib protocol: pretrain self-supervised on the train split,
then train a classification head with cross-entropy (the UEA TEST split doubles as the
validation set). Using EthanolConcentration as an example:

```bash
python pretrain.py \
  --data UEA --root_path dataset/EthanolConcentration \
  --context_points 0 --patch_len 12 --stride 12 --batch_size 16 --n_epochs 20 --lr 1e-3 \
  --patch_mask_ratio 0.2 --point_mask_ratio 0.3 --freq_mask_min 0.0 --freq_mask_max 0.7 \
  --save_dir saved_models --save_name mast_pretrain_uea

python classify.py \
  --data UEA --root_path dataset/EthanolConcentration \
  --finetune_mode linear_probe \
  --pretrained_model saved_models/mast_pretrain_uea.pth \
  --context_points 0 --patch_len 12 --stride 12 --batch_size 16 --n_epochs 20 --lr 1e-3 \
  --save_dir saved_models --save_name mast_linear_probe_uea
```

`classify.py` prints and writes accuracy / precision / recall / F1 (macro). The same
commands are wrapped in `scripts/{pretrain,linear_probe,finetune}_uea.sh`, which read
`UEA_DIR` (default `dataset/EthanolConcentration`) so you can point them at any other UEA
dataset.

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

* [PatchTST](https://github.com/yuqinie98/PatchTST)
* [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library)
* [PyPOTS](https://github.com/WenjieDu/PyPOTS)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
