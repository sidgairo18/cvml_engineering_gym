python -m scripts.compare_cooccur_splits \
  --co_csv_train outputs/coco_train/cooccur_oriented.csv \
  --co_csv_eval  outputs/coco_val/cooccur_oriented.csv \
  --metric nc \
  --out_dir outputs/coco_compare_nc \
  --topk_overlap 5 \
  --plot_anchors 15

