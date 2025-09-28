#!/usr/bin/env bash
set -euo pipefail

CSV="outputs/coco_co_train/cooccur_oriented.csv"
OUT="outputs/coco_anchor_profiles_ts"

mkdir -p "$OUT"

python scripts/plot_anchor_profiles.py \
  --co_csv "$CSV" \
  --out_dir "$OUT" \
  --metric ts \
  --top_k 15

