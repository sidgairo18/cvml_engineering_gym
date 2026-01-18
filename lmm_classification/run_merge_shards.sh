cat outputs/shards/atypical_llava7b_two_gated_shard_*.jsonl > outputs/atypical_llava7b_two_gated_merged.jsonl
cat outputs/shards/typical_llava7b_two_gated_shard_*.jsonl > outputs/typical_llava7b_two_gated_merged.jsonl
cat outputs/shards/atypical_llava0.5b_two_gated_shard_*.jsonl > outputs/atypical_llava0.5b_two_gated_merged.jsonl
cat outputs/shards/typical_llava0.5b_two_gated_shard_*.jsonl > outputs/typical_llava0.5b_two_gated_merged.jsonl

python compute_metrics_from_jsonl.py \
  --ann_file ./coco_typicality_annotations/atypical.json \
  --pred_jsonl outputs/atypical_llava7b_two_gated_merged.jsonl \
  --topk 15 \
  --min_support 5 \
  --save_csv outputs/per_class_atypical_llava7b_two_gated.csv \
  --save_json outputs/per_class_atypical_llava7b_two_gated.json

python compute_metrics_from_jsonl.py \
  --ann_file ./coco_typicality_annotations/typical.json \
  --pred_jsonl outputs/typical_llava7b_two_gated_merged.jsonl \
  --topk 15 \
  --min_support 5 \
  --save_csv outputs/per_class_typical_llava7b_two_gated.csv \
  --save_json outputs/per_class_typical_llava7b_two_gated.json

python compute_metrics_from_jsonl.py \
  --ann_file ./coco_typicality_annotations/atypical.json \
  --pred_jsonl outputs/atypical_llava0.5b_two_gated_merged.jsonl \
  --topk 15 \
  --min_support 5 \
  --save_csv outputs/per_class_atypical_llava0.5b_two_gated.csv \
  --save_json outputs/per_class_atypical_llava0.5b_two_gated.json

python compute_metrics_from_jsonl.py \
  --ann_file ./coco_typicality_annotations/typical.json \
  --pred_jsonl outputs/typical_llava0.5b_two_gated_merged.jsonl \
  --topk 15 \
  --min_support 5 \
  --save_csv outputs/per_class_typical_llava0.5b_two_gated.csv \
  --save_json outputs/per_class_typical_llava0.5b_two_gated.json
