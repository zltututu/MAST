#!/usr/bin/env bash
# Stage 2a - downstream classification on a UEA dataset with linear probing.
#
# The pretrained backbone is frozen, only the classification head is trained.
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
  --finetune_mode linear_probe \
  --pretrained_model "$PRETRAINED" \
  --context_points 0 \
  --patch_len 12 \
  --stride 12 \
  --batch_size 16 \
  --n_epochs 20 \
  --lr 1e-3 \
  --save_dir saved_models \
  --save_name mast_linear_probe_uea
