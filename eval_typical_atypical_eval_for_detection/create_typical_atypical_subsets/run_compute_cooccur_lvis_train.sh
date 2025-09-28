#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"
OUT_DIR="outputs/lvis_co_train"
ANN="$BASE_DIR/lvis/annotations/lvis_v1_train.json"

mkdir -p "$OUT_DIR"

python scripts/compute_cooccur.py \
  --ann "$ANN" \
  --out_dir "$OUT_DIR" \
  --alpha 1.0 \
  --auto_min --q_class 0.10 --q_pair 0.10 --floor_class 10 --floor_pair 3 \
  --support_T 5 --support_slope 2.0 \
  --save_unordered \
  --plot_topk 20 --plot_metric pmi \
  --plot_heatmap "$OUT_DIR/heatmap_pmi_top80.png" \
  --heatmap_max_cats 80 \
  --heatmap_sample_seed 3407
