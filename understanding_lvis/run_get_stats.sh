python get_overview-stats_lvis_coco.py --dataset coco \
  --ann_json /BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco/annotations/instances_val2017.json \
  --save_csv coco_val_stats.csv \
  --save_summary coco_val_summary.json

python get_overview-stats_lvis_coco.py --dataset lvis \
  --ann_json /BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/lvis/annotations/lvis_v1_val.json \
  --save_csv lvis_val_stats.csv \
  --save_summary lvis_val_summary.json
