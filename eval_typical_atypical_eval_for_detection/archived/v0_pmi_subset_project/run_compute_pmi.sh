BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"
python scripts/compute_pmi.py \
    --ann $BASE_DIR/coco/annotations/instances_train2017.json \
    --dataset coco \
    --min_images 100 \
    --min_pair 20 \
    --top_k 300 \
    --out_dir outputs/coco_pmi
