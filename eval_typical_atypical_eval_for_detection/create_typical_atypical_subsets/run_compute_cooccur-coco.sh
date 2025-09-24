#!/usr/bin/env bash
set -e

BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"

python scripts/compute_cooccur.py \
  --ann "$BASE_DIR/coco/annotations/instances_train2017.json" \
  --dataset coco \
  --out_dir outputs/coco_co \
  --alpha 1.0 \
  --auto_min --q_class 0.10 --q_pair 0.10 --floor_class 20 --floor_pair 5 \
  --plot_topk 20 \
  --plot_heatmap outputs/coco_co/pmi_heatmap_20-classes.png \
  --heatmap_max_cats 20 \
  --heatmap_pad nan \
  --heatmap_sample_seed 3407

