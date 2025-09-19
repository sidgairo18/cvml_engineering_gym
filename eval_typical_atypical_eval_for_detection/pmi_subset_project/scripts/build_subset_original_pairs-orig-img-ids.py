#!/usr/bin/env python3
# Build typical/atypical subsets using ORIGINAL COCO/LVIS images (no synthesis).
# Option B: Keep ORIGINAL image IDs in the output JSON and DEDUPLICATE
# images/annotations across pairs within each subset (typical/atypical).

import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse, json, random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from pycocotools.coco import COCO


def _infer_file_name(im: dict, dataset: str) -> str:
    """
    Robustly infer a filename for an image record.
    - If 'file_name' exists and is non-empty, use it.
    - Else, try to parse from 'coco_url'.
    - Else, fall back to zero-padded COCO-style name: 000000XXXXXXXX.jpg
      (LVIS images come from COCO, so this is correct.)
    """
    fn = im.get("file_name")
    if isinstance(fn, str) and fn.strip():
        return fn
    url = im.get("coco_url")
    if isinstance(url, str) and url.strip():
        base = os.path.basename(url)
        if base:
            return base
    # final fallback: COCO naming
    return f"{int(im['id']):012d}.jpg"


def find_images_with_both(
    coco: COCO,
    cat_id_a: int,
    cat_id_b: int,
    min_area: float = 1.0,
    require_noncrowd: bool = True,
) -> List[int]:
    """Return image ids that contain at least one instance of both categories."""
    imgs_a = set(coco.getImgIds(catIds=[cat_id_a]))
    imgs_b = set(coco.getImgIds(catIds=[cat_id_b]))
    cand = list(imgs_a.intersection(imgs_b))
    result: List[int] = []
    for img_id in cand:
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        has_a = False
        has_b = False
        for a in anns:
            if require_noncrowd and a.get('iscrowd', 0) == 1:
                continue
            if a.get('area', 1.0) < min_area:
                continue
            if a['category_id'] == cat_id_a:
                has_a = True
            elif a['category_id'] == cat_id_b:
                has_b = True
            if has_a and has_b:
                result.append(img_id)
                break
    return result


def collect_annotations_for_images(
    coco: COCO,
    image_ids: List[int],
    only_target_ids: Optional[Set[int]] = None,
    min_area: float = 1.0,
    require_noncrowd: bool = True,
    dataset: str = "coco",
) -> Tuple[List[dict], List[dict]]:
    """
    Collect images and (filtered) annotations for given image ids.

    Returns:
      images: list of COCO/LVIS image dicts with ORIGINAL 'id' and a valid 'file_name'
      annotations: list of dicts with fields:
         - image_id (ORIGINAL)
         - category_id, iscrowd, bbox, area, segmentation
         - orig_ann_id (the ORIGINAL annotation id)  <-- used for dedup
      NOTE: We DO NOT assign 'id' to annotations here; we add them later once deduped.
    """
    images: List[dict] = []
    annotations: List[dict] = []
    for img_id in image_ids:
        im = coco.loadImgs([img_id])[0]
        images.append(dict(
            id=im['id'],
            file_name=_infer_file_name(im, dataset),
            height=im['height'],
            width=im['width'],
        ))
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        for a in anns:
            if require_noncrowd and a.get('iscrowd', 0) == 1:
                continue
            if a.get('area', 1.0) < min_area:
                continue
            if only_target_ids is not None and a['category_id'] not in only_target_ids:
                continue
            bbox = a.get('bbox')
            segm = a.get('segmentation')
            if bbox is None or segm is None:
                continue
            annotations.append(dict(
                image_id=im['id'],
                category_id=int(a['category_id']),
                iscrowd=int(a.get('iscrowd', 0)),
                bbox=[float(x) for x in bbox],
                area=float(a.get('area', 0.0)),
                segmentation=segm,
                orig_ann_id=int(a['id']),
            ))
    return images, annotations


def main():
    ap = argparse.ArgumentParser(description="Build typical/atypical subsets using original images only (keep original image IDs).")
    ap.add_argument('--dataset', required=True, choices=['coco', 'lvis'])
    ap.add_argument('--images_root', required=True, help="(Not used for writing; kept for parity/QA tooling.)")
    ap.add_argument('--ann', required=True, help="Path to COCO/LVIS annotations (train json).")
    ap.add_argument('--pairs_json', required=True, help="Flat pairs json produced by select_matched_pairs.py (key 'pairs').")
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--per_pair_per_type', type=int, default=200)
    ap.add_argument('--min_area', type=float, default=1.0)
    ap.add_argument('--only_target_cats', action='store_true', help="If set, include only A/B categories in annotations; else include all.")
    ap.add_argument('--seed', type=int, default=3407)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    coco = COCO(args.ann)
    with open(args.pairs_json, 'r') as f:
        pairs_data = json.load(f)
    pairs = pairs_data.get('pairs', [])

    cat_name_to_id: Dict[str, int] = {c['name']: c['id'] for c in coco.loadCats(coco.getCatIds())}

    images_typ: List[dict] = []
    anns_typ: List[dict] = []
    images_aty: List[dict] = []
    anns_aty: List[dict] = []

    # Dedup structures per subset
    seen_imgs_typ: Set[int] = set()     # original image ids
    seen_imgs_aty: Set[int] = set()
    seen_anns_typ: Set[int] = set()     # original annotation ids
    seen_anns_aty: Set[int] = set()

    manifest_typ: List[dict] = []
    manifest_aty: List[dict] = []

    for p in pairs:
        a = p['a']; b = p['b']; typetag = p.get('type', 'typical')
        if a not in cat_name_to_id or b not in cat_name_to_id:
            continue
        id_a = cat_name_to_id[a]; id_b = cat_name_to_id[b]

        cand_imgs = find_images_with_both(coco, id_a, id_b, min_area=args.min_area, require_noncrowd=True)
        if not cand_imgs:
            continue
        random.shuffle(cand_imgs)
        chosen = cand_imgs[:args.per_pair_per_type]

        only_ids = {id_a, id_b} if args.only_target_cats else None
        imgs, anns = collect_annotations_for_images(
            coco, chosen,
            only_target_ids=only_ids,
            min_area=args.min_area,
            require_noncrowd=True,
            dataset=args.dataset,
        )

        if typetag == 'typical':
            for im in imgs:
                if im['id'] not in seen_imgs_typ:
                    images_typ.append(im)
                    seen_imgs_typ.add(im['id'])
            for an in anns:
                orig_ann_id = an.get('orig_ann_id')
                if orig_ann_id is None or orig_ann_id in seen_anns_typ:
                    continue
                if an['image_id'] not in seen_imgs_typ:
                    continue
                an_out = dict(an)
                del an_out['orig_ann_id']
                an_out['id'] = len(anns_typ) + 1
                anns_typ.append(an_out)
                seen_anns_typ.add(orig_ann_id)
            for img_id in chosen:
                manifest_typ.append(dict(orig_img_id=int(img_id), a=a, b=b, type='typical', match_id=p.get('match_id')))
        else:
            for im in imgs:
                if im['id'] not in seen_imgs_aty:
                    images_aty.append(im)
                    seen_imgs_aty.add(im['id'])
            for an in anns:
                orig_ann_id = an.get('orig_ann_id')
                if orig_ann_id is None or orig_ann_id in seen_anns_aty:
                    continue
                if an['image_id'] not in seen_imgs_aty:
                    continue
                an_out = dict(an)
                del an_out['orig_ann_id']
                an_out['id'] = len(anns_aty) + 1
                anns_aty.append(an_out)
                seen_anns_aty.add(orig_ann_id)
            for img_id in chosen:
                manifest_aty.append(dict(orig_img_id=int(img_id), a=a, b=b, type='atypical', match_id=p.get('match_id')))

    cats = coco.loadCats(coco.getCatIds())
    out_typ = dict(images=images_typ, annotations=anns_typ, categories=cats)
    out_aty = dict(images=images_aty, annotations=anns_aty, categories=cats)

    # LVIS requires these image fields (even if empty)
    if args.dataset == 'lvis':
        for im in out_typ['images']:
            im['not_exhaustive_category_ids'] = []
            im['neg_category_ids'] = []
        for im in out_aty['images']:
            im['not_exhaustive_category_ids'] = []
            im['neg_category_ids'] = []

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

