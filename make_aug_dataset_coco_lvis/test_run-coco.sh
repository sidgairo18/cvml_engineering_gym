export PYTHONPATH=$PYTHONPATH:/BS/generative_modelling_for_image_understanding/work/my_projects/github_repos/cvml_engineering_gym/make_aug_dataset_coco_lvis/Inpaint-Anything
BASE="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS" 
COCO_IMG_DIR=$BASE"/coco/train2017"                                                           
COCO_ANN=$BASE"/coco/annotations/instances_train2017.json"                                    
OUT_DIR="./out/coco_sam_lama"                                              
# ================================================================                            
                                                                                              
# Optional: pick GPU                                                                          
#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"                                      
                                                                                              
# Helps with fragmentation on long runs                                                       
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  

python make_inpainted_dataset_inpainting_anything.py \
  --dataset coco \
  --images-dir $COCO_IMG_DIR \
  --ann-file $COCO_ANN \
  --out-dir ./out/coco_sam_lama \
  --split-name train2017 \
  --strategy sizebased --k-remove 1 --area-max 0.30 --dilate 5 \
  --max-images 10 \
  --lama-config ./lama/configs/prediction/default.yaml \
  --lama-ckpt ./pretrained_models/big-lama \
  --save-preview --preview-alpha 0.55

