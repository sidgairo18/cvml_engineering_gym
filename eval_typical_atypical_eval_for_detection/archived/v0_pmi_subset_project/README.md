# Co-occurrence Typical vs Atypical Subsets (COCO & LVIS)

This project provides **container-free** Python scripts to:
1. Compute **PMI** (Pointwise Mutual Information) between category pairs on COCO/LVIS.
2. Select **typical** (high PMI) and **atypical** (low PMI but plausible) pairs.
3. Build **evaluation subsets**:
   - **Method 1 (Real-Paste):** Compose images by pasting GT-masked instances into backgrounds with IoA/occlusion control.
   - **Method 2 (Instance Diffusion):** Prepare prompts & manifests for diffusion-based synthesis / inpainting (plug your IA pipeline).

Everything is **Co-DETR friendly** (COCO-style annotations).

## Quick Start

### 0) Environment (conda/mamba)
```bash
mamba create -n pmi_subsets python=3.10 -y
mamba activate pmi_subsets
pip install numpy pandas opencv-python pillow pycocotools tqdm matplotlib
```

> Optional (quality filters): `pip install clip-anytorch open_clip_torch torch torchvision`

### 1) Compute PMI
```bash
# COCO
python scripts/compute_pmi.py   --ann /path/to/coco/annotations/instances_train2017.json   --dataset coco   --min_images 100 --min_pair 20 --top_k 300   --out_dir outputs/coco_pmi

# LVIS
python scripts/compute_pmi.py   --ann /path/to/lvis/annotations/lvis_v1_train.json   --dataset lvis   --min_images 50 --min_pair 10 --top_k 500   --out_dir outputs/lvis_pmi
```

### 2) Select pairs (typical & atypical)
```bash
python scripts/select_pairs.py   --pmi_csv outputs/coco_pmi/pmi.csv   --cat_map_json outputs/coco_pmi/category_map.json   --dataset coco   --top_typical 20 --top_atypical 20   --llm_pairs_json optional_llm_pairs.json   --out_json outputs/coco_pairs_selected.json
```

### 3) Build subsets — Method 1 (Real-Paste)
```bash
python scripts/build_subset_real_paste.py   --dataset coco   --images_root /path/to/coco/train2017   --ann /path/to/coco/annotations/instances_train2017.json   --pairs_json outputs/coco_pairs_selected.json   --out_dir outputs/coco_subsets_real_paste   --per_pair_per_type 200   --max_workers 8
```

### 4) Build subsets — Method 2 (Instance Diffusion)
This script **prepares** prompts/manifests; connect it to your IA pipeline.
```bash
python scripts/build_subset_diffusion.py   --dataset coco   --pairs_json outputs/coco_pairs_selected.json   --out_dir outputs/coco_subsets_diffusion   --per_pair_per_type 200
```

The diffusion pipeline should write back images + masks and fill `manifest.jsonl` produced by the script.

## Outputs
- `outputs/*/pmi.csv`                           — PMI table
- `outputs/*/pairs_selected.json`               — final pair list with scene templates
- `outputs/*/subsets_real_paste/annotations`    — COCO-style JSON per pair-type
- `outputs/*/subsets_diffusion/prompts.jsonl`   — prompts and target relations
- `outputs/*/subsets_diffusion/manifest.jsonl`  — (to be filled by your IA pipeline)

## Notes
- LVIS uses COCO image ids; scripts handle both via `--dataset` flag.
- IoA thresholds control occlusion: keep pasted object visible (IoA <= 0.4) and drop fully occluded existing objects (IoA >= 0.95).
- Atypical pairs are filtered for **physical plausibility** via simple heuristics or optional LLM scores.
