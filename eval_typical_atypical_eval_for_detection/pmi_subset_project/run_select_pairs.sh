#python scripts/select_pairs.py \
#    --pmi_csv outputs/coco_pmi/pmi.csv \
#    --cat_map_json outputs/coco_pmi/category_map.json \
#    --dataset coco \
#    --top_typical 20 \
#    --top_atypical 20 \
#    --out_json outputs/coco_pairs_selected.json
#
# 1) After computing PMI (pmi.csv), create matched pairs:
python scripts/select_matched_pairs.py \
  --pmi_csv outputs/coco_pmi/pmi.csv \
  --dataset coco \
  --k 20 \
  --pairs_per_anchor_typ 3 \
  --pairs_per_anchor_aty 3 \
  --min_plausibility 0.0 \
  --out_matched_json outputs/coco_matched_pairs.json \
  --out_pairs_flat_json outputs/coco_pairs_flat.json
