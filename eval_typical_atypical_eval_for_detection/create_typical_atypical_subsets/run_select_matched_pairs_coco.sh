#!/usr/bin/env bash
set -euo pipefail

CSV="outputs/coco_co_train/cooccur_oriented.csv"
OUT_MATCHED="outputs/coco_pairs/matched_ts.json"
OUT_FLAT="outputs/coco_pairs/matched_ts_flat.json"

mkdir -p "$(dirname "$OUT_MATCHED")"

python scripts/select_matched_pairs.py \
  --co_csv "$CSV" \
  --dataset coco \
  --metric ts \
  --k_typ 2 \
  --k_aty 2 \
  --out_matched_json "$OUT_MATCHED" \
  --out_pairs_flat_json "$OUT_FLAT"

