# Co-occurrence Typical vs Atypical Subsets (COCO & LVIS)

Container-free Python scripts to:
1. Compute **PMI** (Pointwise Mutual Information) between category pairs on COCO/LVIS.
2. Select **typical** (high PMI) and **atypical** (low PMI but plausible) pairs — optionally matched per **anchor** class.
3. Build **evaluation subsets**:
   - **Control (Original Images):** Use only images where pairs co-occur in the dataset.
   - **Method 1 (Real-Paste):** Compose images by pasting GT-masked instances into backgrounds with IoA control.
   - **Method 2 (Instance Diffusion):** Prepare prompts & manifests for diffusion/inpainting-based synthesis.

All outputs are **COCO/LVIS instances-style** so you can run standard AP evaluation.

## Env (conda/mamba)
```bash
mamba create -n pmi_subsets python=3.10 -y
mamba activate pmi_subsets
pip install numpy pandas opencv-python pillow pycocotools tqdm matplotlib
# optional quality filters:
# pip install clip-anytorch open_clip_torch torch torchvision
```

## 1) Compute PMI
```bash
# COCO
python -m scripts.compute_pmi   --ann /path/to/coco/annotations/instances_train2017.json   --dataset coco   --min_images 100 --min_pair 20   --out_dir outputs/coco_pmi   --plot_topk 20

# LVIS
python -m scripts.compute_pmi   --ann /path/to/lvis/annotations/lvis_v1_train.json   --dataset lvis   --min_images 50 --min_pair 10   --out_dir outputs/lvis_pmi   --plot_topk 20
```

## 2) Select pairs
### (a) Simple top/bottom by PMI
```bash
python -m scripts.select_pairs   --pmi_csv outputs/coco_pmi/pmi.csv   --cat_map_json outputs/coco_pmi/category_map.json   --dataset coco   --top_typical 20 --top_atypical 20   --out_json outputs/coco_pairs_selected.json
```

### (b) Matched per anchor (recommended for bias gaps)
```bash
python -m scripts.select_matched_pairs   --pmi_csv outputs/coco_pmi/pmi.csv   --dataset coco   --k 20   --pairs_per_anchor_typ 1   --pairs_per_anchor_aty 1   --out_matched_json outputs/coco_matched_pairs.json   --out_pairs_flat_json outputs/coco_pairs_flat.json
```

## 3) CONTROL: Original-image subsets (no synthesis)
```bash
python -m scripts.build_subset_original_pairs   --dataset coco   --images_root /path/to/coco/train2017   --ann /path/to/coco/annotations/instances_train2017.json   --pairs_json outputs/coco_pairs_flat.json   --out_dir outputs/coco_subsets_original   --per_pair_per_type 200
# Evaluate typical.json vs atypical.json with your standard pipeline
```

## 4) Method 1: Real-paste synthesis
```bash
python -m scripts.build_subset_real_paste   --dataset coco   --images_root /path/to/coco/train2017   --ann /path/to/coco/annotations/instances_train2017.json   --pairs_json outputs/coco_pairs_flat.json   --out_dir outputs/coco_subsets_real_paste   --per_pair_per_type 200
# Export to COCO/LVIS instances for eval
python -m scripts.make_eval_json   --dataset coco   --images_root outputs/coco_subsets_real_paste/images   --categories_json /path/to/coco/annotations/instances_train2017.json   --out_json outputs/coco_eval_typical.json   --source real_paste   --subset_type typical
```

## 5) Method 2: Diffusion prompts/manifest
```bash
python -m scripts.build_subset_diffusion   --dataset coco   --pairs_json outputs/coco_pairs_flat.json   --out_dir outputs/coco_subsets_diffusion   --per_pair_per_type 200
# Run your diffusion/inpainting pipeline, fill manifest.jsonl, then:
python -m scripts.make_eval_json   --dataset coco   --images_root outputs/coco_subsets_diffusion/images   --masks_root  outputs/coco_subsets_diffusion/masks   --categories_json /path/to/coco/annotations/instances_train2017.json   --diffusion_manifest outputs/coco_subsets_diffusion/manifest.jsonl   --out_json outputs/coco_eval_typical_diffusion.json   --source diffusion   --subset_type typical
```

## PMI Top/Bottom Plot
`compute_pmi.py` can create a combined bar chart of Top-K and Bottom-K pairs:
```
--plot_topk K
```
It saves `pmi_top_bottom_plot_kK.png` in the output directory.
