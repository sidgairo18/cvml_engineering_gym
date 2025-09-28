#!/usr/bin/env bash
set -e

BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"

# 1) Compute oriented co-occurrence
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

# 2) Create matched pairs
python scripts/selected_matched_pairs.py \
  --co_csv outputs/lvis_co/cooccur_oriented.csv \
  --dataset lvis \
  --metric pmi \
  --k 20 \
  --pairs_per_anchor_typ 3 \
  --pairs_per_anchor_aty 3 \
  --min_plausibility 0.0 \
  --out_matched_json outputs/lvis_co/lvis_matched_pairs.json \
  --out_pairs_flat_json outputs/lvis_co/lvis_pairs_flat.json

# 3) Build CONTROL subsets from original LVIS images (keeps original image IDs)
python -m scripts.build_subset_original_pairs-orig-img-ids \
  --dataset lvis \
  --images_root "$BASE_DIR/lvis/train2017" \
  --ann "$BASE_DIR/lvis/annotations/lvis_v1_train.json" \
  --pairs_json outputs/lvis_co/lvis_pairs_flat.json \
  --out_dir outputs/lvis_subsets_original \
  --per_pair_per_type 20

# 4) Optional: visualize grouped by anchor
python -m scripts.group_images_by_anchor \
  --mode original \
  --images_root "$BASE_DIR/lvis/train2017" \
  --manifests_dir outputs/lvis_subsets_original \
  --ann_json_typical outputs/lvis_subsets_original/typical.json \
  --ann_json_atypical outputs/lvis_subsets_original/atypical.json \
  --out_dir outputs/lvis_viz/by_anchor_original_drawn \
  --draw \
  --draw_all

