#!/usr/bin/env bash
set -e

BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"

# Example: build original-image control subsets for COCO using matched pairs
python -m scripts.build_subset_original_pairs-new-img-ids \
  --dataset coco \
  --images_root "$BASE_DIR/coco/train2017" \
  --ann "$BASE_DIR/coco/annotations/instances_train2017.json" \
  --pairs_json outputs/coco_co/coco_pairs_flat.json \
  --out_dir outputs/coco_subsets_original \
  --per_pair_per_type 20

