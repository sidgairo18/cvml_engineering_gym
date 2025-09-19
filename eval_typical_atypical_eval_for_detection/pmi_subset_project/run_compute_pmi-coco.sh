BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"
python scripts/compute_pmi.py \
    --ann $BASE_DIR/coco/annotations/instances_train2017.json \
    --dataset coco \
    --min_images 20 \
    --min_pair 5 \
    --out_dir outputs/coco_pmi \
    --plot_topk 20 \
    --plot_heatmap outputs/coco_pmi/pmi_heatmap_20-classes.png \
    --heatmap_max_cats 20 \
    --heatmap_pad nan
