#!/usr/bin/env bash
set -e

# COCO — simple top/bottom selection (PMI by default; switch --metric to nc or ts if desired)
python scripts/selected_pairs.py \
  --co_csv outputs/coco_co/cooccur_oriented.csv \
  --dataset coco \
  --metric pmi \
  --top_typical 20 \
  --top_atypical 20 \
  --out_json outputs/coco_co/coco_pairs_selected.json

# COCO — matched per anchor (recommended)
python scripts/selected_matched_pairs.py \
  --co_csv outputs/coco_co/cooccur_oriented.csv \
  --dataset coco \
  --metric pmi \
  --k 20 \
  --pairs_per_anchor_typ 3 \
  --pairs_per_anchor_aty 3 \
  --min_plausibility 0.0 \
  --out_matched_json outputs/coco_co/coco_matched_pairs.json \
  --out_pairs_flat_json outputs/coco_co/coco_pairs_flat.json

