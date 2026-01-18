#!/bin/bash
#SBATCH -D ./
#SBATCH -J 0.5bllava_coco_val_array
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=110GB
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu17
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# >>> set your array size here <<<
#SBATCH --array=0-4

#set -euo pipefail

# create logs director
mkdir logs

# --- conda ---
source /home/sgairola/.bashrc
eval "$(conda shell.bash hook)"
mamba activate llava_coco

# --- project ---
cd /BS/generative_modelling_for_image_understanding/work/my_projects/github_repos/cvml_engineering_gym/lmm_classification
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p outputs/shards

# --- paths (val2017) ---
ANN_FILE="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco/annotations/instances_val2017.json"
IMG_DIR="/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco/val2017"

# --- sharding config ---
# For val2017 = 5000 images.
# With --array=0-9 => 10 shards => 500 images per shard.
SHARD_SIZE=1000

START_IDX=$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))
END_IDX=$((START_IDX + SHARD_SIZE))

OUT_JSONL="outputs/shards/val2017_llava0.5b_two_gated_shard_${SLURM_ARRAY_TASK_ID}.jsonl"

python llava_onevision_coco_multilabel.py \
  --ann_file "$ANN_FILE" \
  --img_dir "$IMG_DIR" \
  --model_id "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
  --mode two_gated \
  --batch_size 8 \
  --k -1 \
  --start_idx "$START_IDX" \
  --end_idx "$END_IDX" \
  --verbose_every 100 \
  --out_jsonl "$OUT_JSONL"
