#!/usr/bin/env bash
# Stage 2a - downstream forecasting with linear probing.
#
# The pretrained backbone is frozen, only the prediction head is trained.
# Requires scripts/pretrain_etth1.sh to have been run first.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
PRETRAINED="${PRETRAINED:-saved_models/mast_pretrain_etth1.pth}"

"$PYTHON_BIN" forecast.py \
  --finetune_mode linear_probe \
  --pretrained_model "$PRETRAINED" \
  --context_points 336 \
  --target_points 96 \
  --patch_len 16 \
  --stride 8 \
  --batch_size 64 \
  --n_epochs 20 \
  --lr 1e-3 \
  --save_dir saved_models \
  --save_name mast_linear_probe_etth1
