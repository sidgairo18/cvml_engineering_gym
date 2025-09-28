#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"
OUT_DIR="outputs/coco_co_val"
ANN="$BASE_DIR/coco/annotations/instances_val2017.json"

mkdir -p "$OUT_DIR"

python scripts/compute_cooccur.py \
  --ann "$ANN" \
  --out_dir "$OUT_DIR" \
  --alpha 1.0 \
  --auto_min --q_class 0.10 --q_pair 0.10 --floor_class 20 --floor_pair 5 \
  --support_T 15 --support_slope 5.0 \
  --save_unordered \
  --plot_topk 20 --plot_metric pmi \
  --plot_heatmap "$OUT_DIR/heatmap_pmi_top80.png" \
  --heatmap_max_cats 80 \
  --heatmap_sample_seed 3407

