BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"
python scripts/compute_pmi.py \
    --ann $BASE_DIR/lvis/annotations/lvis_v1_train.json \
    --dataset lvis \
    --min_images 200 \
    --min_pair 20 \
    --out_dir outputs/lvis_pmi \
    --plot_topk 20 \
    --plot_heatmap outputs/lvis_pmi/pmi_heatmap_80-classes.png \
    --heatmap_max_cats 80 \
    --heatmap_pad nan
exit
# 1) After computing PMI (pmi.csv), create matched pairs:
python scripts/select_matched_pairs.py \
  --pmi_csv outputs/lvis_pmi/pmi.csv \
  --dataset lvis \
  --k 20 \
  --pairs_per_anchor_typ 3 \
  --pairs_per_anchor_aty 3 \
  --min_plausibility 0.0 \
  --out_matched_json outputs/lvis_pmi/lvis_matched_pairs.json \
  --out_pairs_flat_json outputs/lvis_pmi/lvis_pairs_flat.json

python -m scripts.build_subset_original_pairs-orig-img-ids \
  --dataset lvis \
  --images_root $BASE_DIR/lvis/train2017 \
  --ann $BASE_DIR/lvis/annotations/lvis_v1_train.json \
  --pairs_json outputs/lvis_pmi/lvis_pairs_flat.json \
  --out_dir outputs/lvis_subsets_original \
  --per_pair_per_type 20

python -m scripts.group_images_by_anchor \
  --mode original \
  --images_root $BASE_DIR/lvis/train2017 \
  --manifests_dir outputs/lvis_subsets_original \
  --ann_json_typical outputs/lvis_subsets_original/typical.json \
  --ann_json_atypical outputs/lvis_subsets_original/atypical.json \
  --out_dir outputs/lvis_viz/by_anchor_original_drawn \
  --draw \
  --draw_all     # omit --draw_all to show only anchor & partner
