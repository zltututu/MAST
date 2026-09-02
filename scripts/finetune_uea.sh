#!/usr/bin/env bash
# Stage 2b - downstream classification on a UEA dataset with end-to-end finetuning.
#
# The whole network is unfrozen from the start (freeze_epochs 0, following the
# TSLib classification protocol); use --freeze_epochs to warm up the head first.
# Requires scripts/pretrain_uea.sh to have been run first.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
UEA_DIR="${UEA_DIR:-dataset/EthanolConcentration}"
PRETRAINED="${PRETRAINED:-saved_models/mast_pretrain_uea.pth}"

"$PYTHON_BIN" classify.py \
  --data UEA \
  --root_path "$UEA_DIR" \
  --finetune_mode finetune \
  --pretrained_model "$PRETRAINED" \
  --context_points 0 \
  --patch_len 12 \
  --stride 12 \
  --batch_size 16 \
  --n_epochs 20 \
  --freeze_epochs 0 \
  --lr 5e-5 \
  --save_dir saved_models \
  --save_name mast_finetune_uea
