#!/usr/bin/env bash
set -euo pipefail

TR="outputs/coco_co_train/cooccur_oriented.csv"
EV="outputs/coco_co_val/cooccur_oriented.csv"
OUT="outputs/coco_compare_ts"

mkdir -p "$OUT"

python scripts/compare_cooccur_splits.py \
  --train_csv "$TR" \
  --eval_csv "$EV" \
  --out_dir "$OUT" \
  --metric ts \
  --scatter_k 20
