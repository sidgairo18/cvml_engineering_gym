cat outputs/shards/val2017_llava7b_two_gated_shard_*.jsonl > outputs/val2017_llava7b_two_gated_merged.jsonl
cat outputs/shards/val2017_llava0.5b_two_gated_shard_*.jsonl > outputs/val2017_llava0.5b_two_gated_merged.jsonl

python compute_metrics_from_jsonl.py \
  --ann_file /BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco/annotations/instances_val2017.json \
  --pred_jsonl outputs/val2017_llava7b_two_gated_merged.jsonl \
  --topk 15 \
  --min_support 5 \
  --save_csv outputs/per_class_val2017_llava7b_two_gated.csv \
  --save_json outputs/per_class_val2017_llava7b_two_gated.json

python compute_metrics_from_jsonl.py \
  --ann_file /BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco/annotations/instances_val2017.json \
  --pred_jsonl outputs/val2017_llava0.5b_two_gated_merged.jsonl \
  --topk 15 \
  --min_support 5 \
  --save_csv outputs/per_class_val2017_llava0.5b_two_gated.csv \
  --save_json outputs/per_class_val2017_llava0.5b_two_gated.json
