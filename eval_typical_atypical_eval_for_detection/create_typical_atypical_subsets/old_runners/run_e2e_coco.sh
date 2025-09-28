#!/usr/bin/env bash
set -e

BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"

# 1) Compute oriented co-occurrence / PMI / NC / TS
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

# 2) Create matched pairs (change --metric to nc/ts to try alternatives)
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

# 3) Build CONTROL subsets from original images (keeps original image IDs version)
python -m scripts.build_subset_original_pairs-orig-img-ids \
  --dataset coco \
  --images_root "$BASE_DIR/coco/train2017" \
  --ann "$BASE_DIR/coco/annotations/instances_train2017.json" \
  --pairs_json outputs/coco_co/coco_pairs_flat.json \
  --out_dir outputs/coco_subsets_original \
  --per_pair_per_type 20

# 4) Optional: visualize grouped by anchor (draws boxes if eval JSONs are given)
python -m scripts.group_images_by_anchor \
  --mode original \
  --images_root "$BASE_DIR/coco/train2017" \
  --manifests_dir outputs/coco_subsets_original \
  --ann_json_typical outputs/coco_subsets_original/typical.json \
  --ann_json_atypical outputs/coco_subsets_original/atypical.json \
  --out_dir outputs/coco_viz/by_anchor_original_drawn \
  --draw \
  --draw_all

