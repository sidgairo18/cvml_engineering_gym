#!/usr/bin/env bash
set -euo pipefail

CSV="outputs/lvis_co_train/cooccur_oriented.csv"
OUT_JSON="outputs/lvis_pairs/pairs_ts_flat.json"

mkdir -p "$(dirname "$OUT_JSON")"

python scripts/select_pairs.py \
  --co_csv "$CSV" \
  --dataset lvis \
  --metric ts \
  --top_typical 200 \
  --top_atypical 200 \
  --max_per_anchor 3 \
  --out_json "$OUT_JSON"

