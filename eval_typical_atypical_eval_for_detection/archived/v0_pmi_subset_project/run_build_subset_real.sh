BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"
python scripts/build_subset_real_paste.py \
    --dataset coco \
    --images_root $BASE_DIR/coco/train2017 \
    --ann $BASE_DIR/coco/annotations/instances_train2017.json \
    --pairs_json outputs/coco_pairs_selected.json \
    --out_dir outputs/coco_subsets_real_paste \
    --per_pair_per_type 200 \
    --max_workers 8
