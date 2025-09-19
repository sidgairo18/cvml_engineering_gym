#!/usr/bin/env python3
# allow running as module or script
import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse, json, random
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO

def find_images_with_both(coco, cat_id_a, cat_id_b, min_area=1.0, require_noncrowd=True):
    imgs_a = set(coco.getImgIds(catIds=[cat_id_a]))
    imgs_b = set(coco.getImgIds(catIds=[cat_id_b]))
    cand = list(imgs_a.intersection(imgs_b))
    result = []
    for img_id in cand:
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        has_a = False; has_b = False
        for a in anns:
            if require_noncrowd and a.get('iscrowd',0)==1: continue
            if a.get('area',1.0) < min_area: continue
            if a['category_id']==cat_id_a: has_a = True
            elif a['category_id']==cat_id_b: has_b = True
            if has_a and has_b:
                result.append(img_id); break
    return result

def collect_annotations_for_images(coco, image_ids, only_target_ids=None, min_area=1.0, require_noncrowd=True):
    images = []; annotations = []
    imgid_new = {}
    next_img_id = 1
    next_ann_id = 1
    for img_id in image_ids:
        im = coco.loadImgs([img_id])[0]
        #new_id = next_img_id; next_img_id += 1
        #imgid_new[img_id] = new_id
        #images.append(dict(id=new_id, file_name=im['file_name'], height=im['height'], width=im['width']))
        # keep original COCO id
        images.append(dict(id=im['id'], file_name=im['file_name'],
                           height=im['height'], width=im['width']))
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        for a in anns:
            if require_noncrowd and a.get('iscrowd',0)==1: continue
            if a.get('area',1.0) < min_area: continue
            if only_target_ids is not None and a['category_id'] not in only_target_ids: continue
            bbox = a.get('bbox'); segm = a.get('segmentation')
            if bbox is None or segm is None: continue
            #annotations.append(dict(id=next_ann_id, image_id=new_id, category_id=int(a['category_id']),
            #                        iscrowd=int(a.get('iscrowd',0)), bbox=[float(x) for x in bbox],
            #                        area=float(a.get('area',0.0)), segmentation=segm))
            annotations.append(dict(id=next_ann_id,image_id=im['id'],category_id=int(a['category_id']),
                                    iscrowd=int(a.get('iscrowd',0)),bbox=[float(x) for x in bbox],
                                    area=float(a.get('area',0.0)),segmentation=segm
                                    ))
            next_ann_id += 1
    return images, annotations

def main():
    ap = argparse.ArgumentParser(description="Build typical/atypical subsets using original images only.")
    ap.add_argument('--dataset', required=True, choices=['coco','lvis'])
    ap.add_argument('--images_root', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--pairs_json', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--per_pair_per_type', type=int, default=200)
    ap.add_argument('--min_area', type=float, default=1.0)
    ap.add_argument('--only_target_cats', action='store_true')
    ap.add_argument('--seed', type=int, default=3407)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    coco = COCO(args.ann)
    with open(args.pairs_json, 'r') as f:
        pairs_data = json.load(f)
    pairs = pairs_data.get('pairs', [])

    cat_name_to_id = {c['name']: c['id'] for c in coco.loadCats(coco.getCatIds())}

    images_typ, anns_typ = [], []
    images_aty, anns_aty = [], []
    manifest_typ, manifest_aty = [], []

    for p in pairs:
        a = p['a']; b = p['b']; typetag = p.get('type','typical')
        if a not in cat_name_to_id or b not in cat_name_to_id: continue
        id_a = cat_name_to_id[a]; id_b = cat_name_to_id[b]

        cand_imgs = find_images_with_both(coco, id_a, id_b, min_area=args.min_area, require_noncrowd=True)
        if not cand_imgs: continue
        random.shuffle(cand_imgs)
        chosen = cand_imgs[:args.per_pair_per_type]

        only_ids = {id_a, id_b} if args.only_target_cats else None
        imgs, anns = collect_annotations_for_images(coco, chosen, only_target_ids=only_ids, min_area=args.min_area)

        if typetag == 'typical':
            offset_img = len(images_typ)
            idmap = {}
            for im in imgs:
                old = im['id']; new = offset_img + len(idmap) + 1
                idmap[old] = new; im['id'] = new; images_typ.append(im)
            for an in anns:
                an['image_id'] = idmap[an['image_id']]
                an['id'] = len(anns_typ) + 1
                anns_typ.append(an)
            for img_id in chosen:
                manifest_typ.append(dict(orig_img_id=int(img_id), a=a, b=b, type='typical', match_id=p.get('match_id')))
        else:
            offset_img = len(images_aty)
            idmap = {}
            for im in imgs:
                old = im['id']; new = offset_img + len(idmap) + 1
                idmap[old] = new; im['id'] = new; images_aty.append(im)
            for an in anns:
                an['image_id'] = idmap[an['image_id']]
                an['id'] = len(anns_aty) + 1
                anns_aty.append(an)
            for img_id in chosen:
                manifest_aty.append(dict(orig_img_id=int(img_id), a=a, b=b, type='atypical', match_id=p.get('match_id')))

    cats = coco.loadCats(coco.getCatIds())
    out_typ = dict(images=images_typ, annotations=anns_typ, categories=cats)
    out_aty = dict(images=images_aty, annotations=anns_aty, categories=cats)

    with open(os.path.join(args.out_dir, 'typical.json'), 'w') as f:
        json.dump(out_typ, f)
    with open(os.path.join(args.out_dir, 'atypical.json'), 'w') as f:
        json.dump(out_aty, f)
    with open(os.path.join(args.out_dir, 'manifest_typical.jsonl'), 'w') as f:
        for m in manifest_typ:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(args.out_dir, 'manifest_atypical.jsonl'), 'w') as f:
        for m in manifest_aty:
            f.write(json.dumps(m) + "\n")

    print(f"[OK] typical.json: {len(images_typ)} images, {len(anns_typ)} annotations")
    print(f"[OK] atypical.json: {len(images_aty)} images, {len(anns_aty)} annotations")

if __name__ == '__main__':
    main()
