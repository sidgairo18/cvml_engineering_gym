#!/usr/bin/env bash
set -e

BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"

python -m scripts.group_images_by_anchor \
  --mode original \
  --images_root "$BASE_DIR/coco/train2017" \
  --manifests_dir outputs/coco_subsets_original \
  --ann_json_typical outputs/coco_subsets_original/typical.json \
  --ann_json_atypical outputs/coco_subsets_original/atypical.json \
  --out_dir viz/by_anchor_original_drawn \
  --draw

