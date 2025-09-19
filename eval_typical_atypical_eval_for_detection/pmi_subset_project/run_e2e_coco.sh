BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"
python scripts/compute_pmi.py \
    --ann $BASE_DIR/coco/annotations/instances_train2017.json \
    --dataset coco \
    --min_images 200 \
    --min_pair 20 \
    --out_dir outputs/coco_pmi \
    --plot_topk 20 \
    --plot_heatmap outputs/coco_pmi/pmi_heatmap_20-classes.png \
    --heatmap_max_cats 20 \
    --heatmap_pad nan

exit


# 1) After computing PMI (pmi.csv), create matched pairs:
python scripts/select_matched_pairs.py \
  --pmi_csv outputs/coco_pmi/pmi.csv \
  --dataset coco \
  --k 20 \
  --pairs_per_anchor_typ 3 \
  --pairs_per_anchor_aty 3 \
  --min_plausibility 0.0 \
  --out_matched_json outputs/coco_pmi/coco_matched_pairs.json \
  --out_pairs_flat_json outputs/coco_pmi/coco_pairs_flat.json

python -m scripts.build_subset_original_pairs-new-img-ids \
  --dataset coco \
  --images_root $BASE_DIR/coco/train2017 \
  --ann $BASE_DIR/coco/annotations/instances_train2017.json \
  --pairs_json outputs/coco_pmi/coco_pairs_flat.json \
  --out_dir outputs/coco_subsets_original \
  --per_pair_per_type 20

python -m scripts.group_images_by_anchor \
  --mode original \
  --images_root $BASE_DIR/coco/train2017 \
  --manifests_dir outputs/coco_subsets_original \
  --ann_json_typical outputs/coco_subsets_original/typical.json \
  --ann_json_atypical outputs/coco_subsets_original/atypical.json \
  --out_dir outputs/coco_viz/by_anchor_original_drawn \
  --draw \
  --draw_all     # omit --draw_all to show only anchor & partner
