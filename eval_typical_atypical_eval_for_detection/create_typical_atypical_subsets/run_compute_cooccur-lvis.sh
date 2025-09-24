#!/usr/bin/env bash
set -e

BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"

python scripts/compute_cooccur.py \
  --ann "$BASE_DIR/lvis/annotations/lvis_v1_train.json" \
  --dataset lvis \
  --out_dir outputs/lvis_co \
  --alpha 1.0 \
  --auto_min --q_class 0.10 --q_pair 0.10 --floor_class 10 --floor_pair 5 \
  --plot_topk 20 \
  --plot_heatmap outputs/lvis_co/pmi_heatmap_80-classes.png \
  --heatmap_max_cats 80 \
  --heatmap_pad nan \
  --heatmap_sample_seed 3407

