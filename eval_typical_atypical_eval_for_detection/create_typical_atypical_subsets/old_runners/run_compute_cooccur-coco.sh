#!/usr/bin/env bash
set -e

# Adjust this to your COCO root
BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"

python scripts/compute_cooccur.py \
  --ann "$BASE_DIR/coco/annotations/instances_train2017.json" \
  --out_dir outputs/coco_co \
  --min_area 1.0 \
  \
  # ===== Dataset-aware minima (COCO: moderate tail) =====
  --auto_min \
  --q_class 0.10 --q_pair 0.10 \
  --floor_class 30 --floor_pair 10 \
  \
  # ===== PMI smoothing (kept standard) =====
  --alpha 1.0 \
  \
  # ===== Score choice (COCO): ZLOR evidence + raw NC direction =====
  --ts_evidence zlor \
  --ts_direction nc_raw \
  --ts_support_T 15 \
  --ts_support_slope 10 \
  \
  # ===== Outputs & plots =====
  --save_unordered \
  --plot_topk 20 \
  --plot_metric ts \
  --plot_heatmap outputs/coco_co/ts_heatmap_top40.png \
  --heatmap_max_cats 40 \
  --heatmap_pad nan \
  --heatmap_sample_seed 3407

