#!/bin/bash
#SBATCH -D ./
#SBATCH -J llava_coco_val_single
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=110GB
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu17
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# --- conda ---
source /home/sgairola/.bashrc
eval "$(conda shell.bash hook)"
mamba activate inpaint-dset

# --- project ---
cd /BS/generative_modelling_for_image_understanding/work/my_projects/github_repos/cvml_engineering_gym
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p logs outputs

# --- paths (val2017) ---
ANN_FILE="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco/annotations/instances_val2017.json"
IMG_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco/val2017"

# --- run ---
python llava_onevision_coco_multilabel.py \
  --ann_file "$ANN_FILE" \
  --img_dir "$IMG_DIR" \
  --model_id "llava-hf/llava-onevision-qwen2-7b-ov-hf" \
  --mode two_gated \
  --batch_size 2 \
  --k -1 \
  --verbose_every 200 \
  --out_jsonl "outputs/val2017_llava7b_two_gated_full.jsonl"

