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

python llava_onevision_coco_multilabel.py \
  --img_dir "$COCO_ROOT/val2017" \
  --ann_file "$COCO_ROOT/annotations/instances_val2017.json" \
  --model_id llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  --mode two_gated \
  --k 300 \
  --batch_size 16 \
  --sanity_check \
  --sanity_out sanity_outputs \
  --out_jsonl sanity_outputs/sanity_k300.jsonl \
  #--shuffle --seed 123
