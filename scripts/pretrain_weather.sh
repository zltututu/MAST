#!/usr/bin/env bash
# Stage 1 - self-supervised pretraining of MAST on weather (any non-ETT TSLib
# forecasting CSV works the same way: --data custom + the right --root_path/--data_path).
#
# Requires dataset/weather/weather.csv (see the README for the download link).
# Writes saved_models/mast_pretrain_weather.pth plus a *_losses.csv next to it.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" pretrain.py \
  --data custom \
  --root_path dataset/weather \
  --data_path weather.csv \
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
  --save_name mast_pretrain_weather
