#!/usr/bin/env bash
set -euo pipefail

CSV="outputs/coco_co_train/cooccur_oriented.csv"
OUT_JSON="outputs/coco_pairs/pairs_ts_flat.json"

mkdir -p "$(dirname "$OUT_JSON")"

python scripts/select_pairs.py \
  --co_csv "$CSV" \
  --dataset coco \
  --metric ts \
  --top_typical 100 \
  --top_atypical 100 \
  --max_per_anchor 3 \
  --out_json "$OUT_JSON"

