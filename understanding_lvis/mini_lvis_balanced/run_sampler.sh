#python main.py \
#--train_json /BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/lvis/annotations/lvis_v1_train.json \
#--save_file_name lvis_mini_balanced_25k \
#--total_images 25000 \
#--alpha 0.6 \
#--tier_bonus 1.0,1.0,0.6 \
#--lambda_tier 3.0 \
#--lambda_cap 2.0 \
#--rare_cap 2 \
#--seed 1337 \
#--debug

python main.py \
--train_json /BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/lvis/annotations/lvis_v1_train.json \
--save_file_name lvis_mini_balanced_25k_fixed \
--total_images 25000 \
--alpha 0.6 \
--tier_bonus 1.0,1.0,0.7 \
--lambda_tier 5.0 \
--lambda_cap 0.4 \
--rare_cap 1 \
--seed 1337 \
--debug

# --lambda_tier 5.0 \ #makes drifting off the tier mix costly.
# --lambda_cap 0.5 \ #keeps per-class rare caps gentle (it’s a soft penalty).
#--rare_cap 1 \ #ensures breadth (many rare classes show up) without letting any single rare class dominate.
