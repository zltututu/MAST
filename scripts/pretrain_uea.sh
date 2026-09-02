#!/usr/bin/env bash
# Stage 1 - self-supervised pretraining of MAST on a UEA classification dataset
# (EthanolConcentration by default; point UEA_DIR at any of the other UEA folders).
#
# Requires dataset/<Name>/<Name>_TRAIN.ts and <Name>_TEST.ts (see the README for
# the download links). --context_points 0 auto-detects the padded series length.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
UEA_DIR="${UEA_DIR:-dataset/EthanolConcentration}"

"$PYTHON_BIN" pretrain.py \
  --data UEA \
  --root_path "$UEA_DIR" \
  --context_points 0 \
  --patch_len 12 \
  --stride 12 \
  --batch_size 16 \
  --n_epochs 20 \
  --lr 1e-3 \
  --patch_mask_ratio 0.2 \
  --point_mask_ratio 0.3 \
  --freq_mask_min 0.0 \
  --freq_mask_max 0.7 \
  --save_dir saved_models \
  --save_name mast_pretrain_uea
