COCO_ROOT="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco"
#python llava_onevision_coco_multilabel.py \
#  --coco_root /path/to/coco \
#  --split val2017 \
#  --model_id llava-hf/llava-onevision-qwen2-7b-ov-hf \
#  --mode two_gated \
#  --k 10 \
#  --batch_size 2 \
#  --sanity_check \
#  --sanity_out sanity_outputs \
#  --out_jsonl sanity_outputs/sanity_k10.jsonl \
#  --shuffle --seed 123
#  --ann_file "$COCO_ROOT/annotations/instances_val2017.json" \

python llava_onevision_coco_multilabel.py \
  --img_dir "$COCO_ROOT/train2017" \
  --ann_file "./coco_typicality_annotations/typical.json" \
  --model_id llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --mode two_gated \
  --k 48 \
  --batch_size 8 \
  --sanity_check \
  --sanity_out sanity_outputs \
  --out_jsonl sanity_outputs/sanity_k48.jsonl \
  --shuffle  --seed 123
