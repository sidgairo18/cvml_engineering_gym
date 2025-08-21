python report.py \
  --full_train_json /BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/lvis/annotations/lvis_v1_train.json \
  --mini_json lvis_mini_balanced_25k_fixed.json \
  --save_fig mini_report.png \
  --out_csv mini_report.csv \
  --out_per_class_csv mini_per_class.csv \
  --class_bar_fig class_bars_seed1337 \
  --class_bar_seed 1337 \
  --class_bar_log
