#!/bin/bash                                                                                   
#SBATCH -D ./                                                                                 
#SBATCH -J coco_inp_aug_gen
#SBATCH -t 48:00:00                                                                           
#SBATCH --cpus-per-task=8                                                                     
#SBATCH --mem=110GB                                                                           
#SBATCH --gres gpu:1
#SBATCH --nodes=1                                                                             
#SBATCH --ntasks-per-node=1                                                                   
#SBATCH -p gpu20                                                                  
                                                                                              
# load conda environment                                                                      
source /home/sgairola/.bashrc
eval "$(conda shell.bash hook)"                                                               
mamba activate inpaint-dset
                                                                                              
# run your script here                                                                        
cd /BS/generative_modelling_for_image_understanding/work/my_projects/github_repos/cvml_engineering_gym/make_aug_dataset_coco_lvis/Inpaint-Anything

#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/BS/generative_modelling_for_image_understanding/work/my_projects/github_repos/cvml_engineering_gym/make_aug_dataset_coco_lvis/Inpaint-Anything

# Adjust these to your paths
IMAGES_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco/train2017"
ANN_FILE="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco/annotations/instances_train2017.json"

OUT_DIR="./augmented_dataset/out/coco_sam_lama_all_full"
SPLIT="train2017"

# Weights
SAM_CKPT="./pretrained_models/sam_vit_h_4b8939.pth"
LAMA_CFG="./lama/configs/prediction/default.yaml"
LAMA_CKPT="./pretrained_models/big-lama"

python make_inpainted_dataset_inpainting_anything.py \
  --dataset coco \
  --images-dir "${IMAGES_DIR}" \
  --ann-file "${ANN_FILE}" \
  --out-dir "${OUT_DIR}" \
  --split-name "${SPLIT}" \
  --strategy all \
  --dilate 7 \
  --restore-overlaps-after --restore-margin 0 \
  --lama-ckpt ./pretrained_models/big-lama \
  --lama-config ./lama/configs/prediction/default.yaml
