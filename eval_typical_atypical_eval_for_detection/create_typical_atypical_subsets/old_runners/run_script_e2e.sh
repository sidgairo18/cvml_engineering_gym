# 1) Build oriented co-occurrence
python -m scripts.compute_cooccur \
  --ann /path/to/instances_train2017.json \
  --out_dir outputs/coco_co \
  --auto_min --q_class 0.10 --q_pair 0.10 --floor_class 20 --floor_pair 5 \
  --alpha 1.0 --save_unordered

# 2a) Flat selection (per anchor, top typical & atypical)
python -m scripts.selected_pairs \
  --co_csv outputs/coco_co/cooccur_oriented.csv \
  --dataset coco \
  --top_k_typ 1 --top_k_aty 1 \
  --min_pair_aty 3 \
  --tiebreak_pmi \
  --out_json outputs/coco_co/coco_pairs_selected.json

# 2b) Matched sets (anchor has both sides)
python -m scripts.selected_matched_pairs \
  --co_csv outputs/coco_co/cooccur_oriented.csv \
  --dataset coco \
  --k 20 \
  --pairs_per_anchor_typ 1 \
  --pairs_per_anchor_aty 1 \
  --min_pair_aty 3 \
  --tiebreak_pmi \
  --out_matched_json outputs/coco_co/coco_matched_pairs.json \
  --out_pairs_flat_json outputs/coco_co/coco_pairs_flat.json

