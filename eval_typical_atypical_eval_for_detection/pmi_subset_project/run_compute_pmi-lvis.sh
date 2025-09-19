BASE_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS"
python scripts/compute_pmi.py \
    --ann $BASE_DIR/lvis/annotations/lvis_v1_train.json \
    --dataset lvis \
    --min_images 10 \
    --min_pair 5 \
    --out_dir outputs/lvis_pmi \
    --plot_topk 20 \
    --plot_heatmap outputs/lvis_pmi/pmi_heatmap_80-classes.png \
    --heatmap_max_cats 80 \
    --heatmap_pad nan
