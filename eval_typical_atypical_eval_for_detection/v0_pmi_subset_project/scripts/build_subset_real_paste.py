#!/usr/bin/env python3
import argparse, os, json, random
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from scripts.utils.coco_lvis import DatasetAPI
from scripts.utils.masks import bbox_from_mask, compute_ioa
from scripts.utils.placement import place_relation, translate_mask
from scripts.utils.blending import paste_with_feather

def sample_instances(api, class_name, max_trials=100):
    # Returns (img, ann) where ann is one instance of class_name
    for _ in range(max_trials):
        # iterate until we find a valid instance
        for img, anns in api.iter_instances(cat_name=class_name):
            cand = [a for a in anns if api.cat_id_to_name[a['category_id']] == class_name]
            if not cand: continue
            ann = random.choice(cand)
            return img, ann
    return None, None

def crop_instance(api, img, ann):
    img_path = api.image_path(img)
    I = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if I is None: return None, None, None
    mask = api.ann_to_mask(img['height'], img['width'], ann)
    if mask is None: return None, None, None
    x,y,w,h = bbox_from_mask(mask)
    crop = I[y:y+h, x:x+w].copy()
    m = mask[y:y+h, x:x+w].copy()
    return crop, m, (x,y,w,h)

def pick_background(api, exclude_classes, max_trials=200):
    for _ in range(max_trials):
        # random image with no exclude classes present
        img_id = random.choice(list(api.coco.imgs.keys()))
        ann_ids = api.coco.getAnnIds(imgIds=[img_id])
        anns = api.coco.loadAnns(ann_ids)
        present = set(api.cat_id_to_name[a['category_id']] for a in anns if a.get('iscrowd',0)==0)
        if present.isdisjoint(set(exclude_classes)):
            return api.coco.loadImgs([img_id])[0]
    return None

def build_one(bg_api, obj_api, pair, per_type=200, out_dir='out', ioa_place=0.4, drop_cover=0.95, occl_bins=(0,0.1,0.3,0.5)):
    """Compose per pair (A,B) for typical and atypical."""
    a, b = pair['a'], pair['b']
    relation = pair.get('relation','next to')
    types = ['typical','atypical']
    os.makedirs(out_dir, exist_ok=True)
    ann_dir = os.path.join(out_dir, 'annotations')
    img_dir = os.path.join(out_dir, 'images')
    Path(ann_dir).mkdir(parents=True, exist_ok=True)
    Path(img_dir).mkdir(parents=True, exist_ok=True)

    manifests = {t: [] for t in types}

    for typ in types:
        for idx in tqdm(range(per_type), desc=f"{a}-{b} [{typ}]"):
            # choose background
            ex_cls = [] if typ=='typical' else [a,b]
            bg_img = pick_background(bg_api, ex_cls) or pick_background(bg_api, [])
            bg_path = bg_api.image_path(bg_img)
            BG = cv2.imread(bg_path, cv2.IMREAD_COLOR)
            H,W = BG.shape[:2]

            # sample A and B crops
            ia_img, ia_ann = sample_instances(obj_api, a)
            ib_img, ib_ann = sample_instances(obj_api, b)
            if ia_img is None or ib_img is None:
                continue
            crop_a, mask_a, box_a = crop_instance(obj_api, ia_img, ia_ann)
            crop_b, mask_b, box_b = crop_instance(obj_api, ib_img, ib_ann)
            if crop_a is None or crop_b is None: 
                continue

            # simple rescale to median sizes
            # (could add size bins + pose heuristics here)
            target_h_a = np.random.randint(64, 192)
            scale_a = target_h_a / max(1, mask_a.shape[0])
            crop_a = cv2.resize(crop_a, (int(crop_a.shape[1]*scale_a), int(crop_a.shape[0]*scale_a)))
            mask_a = cv2.resize(mask_a, (crop_a.shape[1], crop_a.shape[0]), interpolation=cv2.INTER_NEAREST)

            target_h_b = np.random.randint(64, 192)
            scale_b = target_h_b / max(1, mask_b.shape[0])
            crop_b = cv2.resize(crop_b, (int(crop_b.shape[1]*scale_b), int(crop_b.shape[0]*scale_b)))
            mask_b = cv2.resize(mask_b, (crop_b.shape[1], crop_b.shape[0]), interpolation=cv2.INTER_NEAREST)

            # placement
            pos_a, pos_b = place_relation(H, W, mask_a, mask_b, relation=relation, existing_masks=[], ioa_thresh=ioa_place)
            if pos_a is None or pos_b is None:
                continue

            # compose
            out = BG.copy()
            xA, yA = pos_a
            out, _ = paste_with_feather(out, crop_a, mask_a, xA, yA, feather_radius=7)
            xB, yB = pos_b
            out, _ = paste_with_feather(out, crop_b, mask_b, xB, yB, feather_radius=7)

            # final masks on canvas
            Ma = translate_mask(mask_a, xA, yA, H, W)
            Mb = translate_mask(mask_b, xB, yB, H, W)

            # occlusion handling: drop fully covered existing objects (not applicable here, we only have A,B)
            # compute boxes
            from scripts.utils.masks import bbox_from_mask
            boxA = bbox_from_mask(Ma); boxB = bbox_from_mask(Mb)
            if boxA is None or boxB is None:
                continue

            # save
            out_name = f"{a}_{b}_{typ}_{idx:06d}.jpg"
            cv2.imwrite(os.path.join(img_dir, out_name), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            manifests[typ].append({
                'file_name': out_name,
                'height': H, 'width': W,
                'objects': [
                    {'category': a, 'bbox': boxA, 'iscrowd': 0},
                    {'category': b, 'bbox': boxB, 'iscrowd': 0}
                ],
                'pair': {'a':a,'b':b,'type':typ,'relation':relation}
            })

    # write simple annotation JSONs per type
    for typ in manifests:
        ann_path = os.path.join(ann_dir, f"{a}_{b}_{typ}.json")
        with open(ann_path, 'w') as f:
            json.dump({'images_dir': 'images', 'samples': manifests[typ]}, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['coco','lvis'])
    ap.add_argument('--images_root', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--pairs_json', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--per_pair_per_type', type=int, default=200)
    ap.add_argument('--ioa_place', type=float, default=0.4)
    ap.add_argument('--drop_cover', type=float, default=0.95)
    ap.add_argument('--max_workers', type=int, default=8)
    args = ap.parse_args()

    with open(args.pairs_json, 'r') as f:
        pairs_data = json.load(f)
    pairs = pairs_data['pairs']

    # background and object APIs can be same (COCO/LVIS images overlap)
    api_bg = DatasetAPI(args.dataset, args.images_root, args.ann)
    api_obj = api_bg

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    for p in pairs:
        build_one(api_bg, api_obj, p, per_type=args.per_pair_per_type, out_dir=out, ioa_place=args.ioa_place, drop_cover=args.drop_cover)

if __name__ == '__main__':
    main()
