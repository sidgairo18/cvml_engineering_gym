# command to create a report of dataset stats.
python -m mini_lvis_balanced.report \
  --train_json /path/to/lvis_v1_train.json \
  --subset_json /path/to/lvis_mini_balanced_25k.json \
  --alpha 0.6 \
  --area_small 32 --area_medium 96 \
  --save_prefix reports/lvis_mini_balanced_25k \
  --export_csv
