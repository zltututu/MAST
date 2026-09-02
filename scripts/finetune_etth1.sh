#!/usr/bin/env bash
# Stage 2b - downstream forecasting with end-to-end finetuning.
#
# The head is trained for --freeze_epochs first, then the whole network is unfrozen.
# Requires scripts/pretrain_etth1.sh to have been run first.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
PRETRAINED="${PRETRAINED:-saved_models/mast_pretrain_etth1.pth}"

"$PYTHON_BIN" forecast.py \
  --finetune_mode finetune \
  --pretrained_model "$PRETRAINED" \
  --context_points 336 \
  --target_points 96 \
  --patch_len 16 \
  --stride 8 \
  --batch_size 64 \
  --n_epochs 20 \
  --freeze_epochs 5 \
  --lr 5e-5 \
  --save_dir saved_models \
  --save_name mast_finetune_etth1
