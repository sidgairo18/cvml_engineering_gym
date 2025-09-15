python scripts/select_pairs.py \
    --pmi_csv outputs/coco_pmi/pmi.csv \
    --cat_map_json outputs/coco_pmi/category_map.json \
    --dataset coco \
    --top_typical 20 \
    --top_atypical 20 \
    --out_json outputs/coco_pairs_selected.json
