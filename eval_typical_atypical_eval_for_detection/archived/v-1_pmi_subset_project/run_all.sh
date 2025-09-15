# 1) Compute PMI pairs
python augment_eval/scripts/pmi_pairs.py --ds COCO
python augment_eval/scripts/pmi_pairs.py --ds LVIS

# 2) (Optional) Merge with LLM suggestions
python augment_eval/scripts/pair_select_merge.py --ds COCO
python augment_eval/scripts/pair_select_merge.py --ds LVIS

# 3) Build Real-Paste subsets (COCO and LVIS)
python augment_eval/scripts/build_subset_real_paste.py --ds COCO --n_per_pair 200
python augment_eval/scripts/build_subset_real_paste.py --ds LVIS --n_per_pair 200

# 4) Build Diffusion subsets (implement your backend first)
python augment_eval/scripts/generate_subset_diffusion.py --per_pair 200

# 5) Train/evaluate your Co-DETR recipes on these subsets and save predictions to preds_root/
# 6) Compute pair-conditioned metrics and Typicality gaps
python augment_eval/scripts/eval_protocol.py \
  --subset_root augment_eval/subsets/real_paste/COCO \
  --preds_root  /path/to/preds/COCO \
  --out augment_eval/outputs/eval_COCO.json

