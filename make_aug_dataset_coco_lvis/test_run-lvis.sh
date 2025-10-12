BASE="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS" 
LVIS_IMG_DIR=$BASE"/lvis/train2017"                                                           
LVIS_ANN=$BASE"/lvis/annotations/lvis_v1_train.json"                                    
OUT_DIR="./out/lvis_sam_lama"                                              
# ================================================================                            

                                                                                              
# Optional: pick GPU                                                                          
#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"                                      
                                                                                              
# Helps with fragmentation on long runs                                                       
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  

python make_inpainted_dataset_inpainting_anything.py \
  --dataset lvis \
  --images-dir $LVIS_IMG_DIR \
  --ann-file $LVIS_ANN \
  --out-dir $OUT_DIR \
  --split-name train2017 \
  --strategy random --k-remove 1 --area-max 0.30 --dilate 5 \
  --max-images 5 \
  --lama-config ./lama/configs/prediction/default.yaml \
  --lama-ckpt ./pretrained_models/big-lama \
  --sam-refine --sam-model-type vit_h --sam-ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
  --save-preview
