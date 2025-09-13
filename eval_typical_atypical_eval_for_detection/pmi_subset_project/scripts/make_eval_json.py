#!/usr/bin/env python3
import argparse, os, json, glob
import numpy as np
import cv2
from pycocotools import mask as mask_utils

def load_category_mapping(categories_json):
    with open(categories_json,'r') as f:
        src = json.load(f)
    cats = src['categories']
    name2id = {c['name']: c['id'] for c in cats}
    return cats, name2id

def encode_mask_binary(mask_arr):
    m = (mask_arr > 0).astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(m))
    rle['counts'] = rle['counts'].decode('utf-8')
    area = float(mask_utils.area(rle))
    ys, xs = np.where(m>0)
    if len(xs)==0 or len(ys)==0:
        bbox = [0,0,0,0]
    else:
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        bbox = [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)]
    return rle, bbox, area

def gather_real_paste_manifests(ann_dir):
    files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    entries = []
    for fp in files:
        with open(fp,'r') as f:
            data = json.load(f)
        for s in data.get('samples', []):
            entries.append(s)
    return entries

def gather_diffusion_manifest(manifest_jsonl):
    entries = []
    with open(manifest_jsonl,'r') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('status','').upper()=='DONE':
                entries.append(obj)
    return entries

def make_coco_json(dataset, images_root, masks_root, categories_json, out_json, source, subset_type=None, diffusion_manifest=None):
    cats, name2id = load_category_mapping(categories_json)
    images, annotations = [], []
    ann_id, img_id = 1, 1

    if source == 'real_paste':
        entries = gather_real_paste_manifests(os.path.join(images_root, "..", "annotations"))
    elif source == 'diffusion':
        if diffusion_manifest is None:
            raise ValueError("Need --diffusion_manifest for source=diffusion")
        entries = gather_diffusion_manifest(diffusion_manifest)
    else:
        raise ValueError("source must be real_paste or diffusion")

    for s in entries:
        if subset_type is not None:
            if 'pair' in s and s['pair'].get('type') != subset_type:
                continue
            if 'pair' not in s and s.get('pair_type') and s['pair_type'] != subset_type:
                continue

        img_rel = s.get('file_name') or s.get('image_path')
        if img_rel is None:
            img_rel = s.get('img_rel') or s.get('img_path')
        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(images_root, img_rel)

        I = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if I is None:
            alt = os.path.join(images_root, 'images', os.path.basename(img_rel))
            I = cv2.imread(alt, cv2.IMREAD_COLOR)
            img_path = alt
            if I is None:
                continue
        H, W = I.shape[:2]
        images.append(dict(id=img_id, file_name=os.path.basename(img_path), height=H, width=W))

        if source == 'real_paste':
            mask_paths = s.get('mask_paths', {})
            for i, obj in enumerate(s['objects']):
                cat_name = obj['category']; cat_id = name2id.get(cat_name)
                if cat_id is None: continue
                suffix = 'A' if i==0 else 'B'
                mrel = mask_paths.get(suffix)
                mpath = mrel if (mrel and os.path.isabs(mrel)) else (mrel and os.path.join(os.path.dirname(images_root), mrel))
                if not mpath or not os.path.exists(mpath):
                    # fallback guess
                    mpath = os.path.join(os.path.dirname(images_root), 'masks', os.path.basename(img_path).replace('.jpg', f'_{suffix}.png'))
                M = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if M is None: 
                    continue
                rle, bbox, area = encode_mask_binary(M)
                annotations.append(dict(id=ann_id, image_id=img_id, category_id=int(cat_id),
                                        iscrowd=0, bbox=bbox, area=area, segmentation=rle))
                ann_id += 1
        else:
            masks = s.get('masks', [])
            for mobj in masks:
                cat_name = mobj['category']; cat_id = name2id.get(cat_name)
                if cat_id is None: continue
                mpath = mobj['mask_path']
                if not os.path.isabs(mpath):
                    mpath = os.path.join(masks_root, mpath)
                M = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if M is None: continue
                rle, bbox, area = encode_mask_binary(M)
                annotations.append(dict(id=ann_id, image_id=img_id, category_id=int(cat_id),
                                        iscrowd=0, bbox=bbox, area=area, segmentation=rle))
                ann_id += 1
        img_id += 1

    coco_json = dict(images=images, annotations=annotations, categories=cats)
    if dataset == 'lvis':
        for im in coco_json['images']:
            im['not_exhaustive_category_ids'] = []
            im['neg_category_ids'] = []

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(coco_json, f)
    print(f"[OK] Wrote {out_json} with {len(images)} images / {len(annotations)} annotations.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['coco','lvis'])
    ap.add_argument('--images_root', required=True)
    ap.add_argument('--masks_root', default=None)
    ap.add_argument('--categories_json', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--source', required=True, choices=['real_paste','diffusion'])
    ap.add_argument('--subset_type', default=None, choices=[None,'typical','atypical'])
    ap.add_argument('--diffusion_manifest', default=None)
    args = ap.parse_args()
    if args.masks_root is None:
        args.masks_root = os.path.join(os.path.dirname(args.images_root), 'masks')
    make_coco_json(args.dataset, args.images_root, args.masks_root, args.categories_json, args.out_json,
                   args.source, args.subset_type, args.diffusion_manifest)
