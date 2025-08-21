#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset statistics for COCO2017 and LVIS v1.

Global stats:
  - total images
  - number of classes
  - number of annotations (total)
  - average instances per image
  - average classes per image (unique categories per image)
  - average instances per class per image (overall mean of per-class: ann_count / img_count)

Per-class stats:
  - category_id, name, (tier for LVIS), ann_count, img_count,
    prevalence (img_count / total_images), avg_inst_per_img_for_class (ann_count / max(1,img_count))

Extras:
  - COCO-style S/M/L bucket counts (32^2, 96^2)
  - LVIS tier summaries (common/frequent/rare)

Usage (COCO):
  python dataset_stats.py --dataset coco \
    --ann_json /path/to/annotations/instances_train2017.json \
    --save_csv coco_train_stats.csv --save_summary coco_train_summary.json

Usage (LVIS):
  python dataset_stats.py --dataset lvis \
    --ann_json /path/to/lvis_v1_train.json \
    --save_csv lvis_train_stats.csv --save_summary lvis_train_summary.json
"""

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

# APIs
from pycocotools.coco import COCO
try:
    from lvis import LVIS
    _HAS_LVIS = True
except Exception:
    _HAS_LVIS = False


# -------------------------------
# Core computations (API-agnostic)
# -------------------------------

def _size_bucket(area, areaRng=(32**2, 96**2, 1e5**2)):
    s, m, l = areaRng
    if area < s:  return "S"
    if area < m:  return "M"
    return "L"

def _per_image_unique_classes(api, anns_by_img, is_lvis=False):
    """Return average number of unique classes per image."""
    # category_id key is 'category_id' for both APIs
    uniq_counts = []
    for img_id, anns in anns_by_img.items():
        cats = {a["category_id"] for a in anns}
        uniq_counts.append(len(cats))
    return float(np.mean(uniq_counts)) if uniq_counts else 0.0

def _collect_common(api, is_lvis=False):
    """Collect basic structures from either COCO or LVIS API object."""
    if is_lvis:
        imgs = api.imgs                     # {id: img_dict}
        anns = {a['id']: a for a in api.anns.values()}   # {id: ann_dict}
        cats = {c['id']: c for c in api.cats.values()}   # {id: cat_dict}
        img_ann_map = defaultdict(list)
        for img_id, ann_list in api.img_ann_map.items():
            # lvis stores list of dicts with 'id'; turn into ann dicts
            for x in ann_list:
                aid = x['id']
                img_ann_map[img_id].append(anns[aid])
                #img_ann_map[img_id].append(anns[aid])
        cat_img_map = {cid: set(api.cat_img_map.get(cid, [])) for cid in cats.keys()}

    else:
        imgs = api.imgs
        cats = {c['id']: c for c in api.loadCats(api.getCatIds())}
        # Build ann dicts and mappings
        ann_ids_all = list(api.anns.keys())
        anns = {aid: api.anns[aid] for aid in ann_ids_all}
        img_ann_map = defaultdict(list)
        for img_id, ann_ids in api.imgToAnns.items():
            img_ann_map[img_id] = [api.anns[a['id']] for a in ann_ids]
        # For COCO, we can infer cat_img_map
        cat_img_map = defaultdict(set)
        for img_id, ann_list in img_ann_map.items():
            for a in ann_list:
                cat_img_map[a["category_id"]].add(img_id)

    return imgs, anns, cats, img_ann_map, cat_img_map

def compute_stats(api, areaRng=(32**2, 96**2, 1e5**2), is_lvis=False, include_tiers=True):
    """
    Compute global + per-class stats for a COCO/LVIS-like API.
    """
    imgs, anns, cats, img_ann_map, cat_img_map = _collect_common(api, is_lvis=is_lvis)

    total_images = len(imgs)
    num_classes  = len(cats)

    # per-class annotation counts
    ann_per_class = Counter()
    size_buckets  = Counter()  # global S/M/L
    for a in anns.values():
        ann_per_class[a["category_id"]] += 1
        # bbox area
        x, y, w, h = a["bbox"]
        if w >= 1 and h >= 1:
            area = w * h
            size_buckets[_size_bucket(area, areaRng)] += 1

    total_annotations = int(sum(ann_per_class.values()))

    # per-class image counts
    img_per_class = {cid: len(cat_img_map.get(cid, [])) for cid in cats.keys()}

    # averages
    avg_instances_per_image = (total_annotations / total_images) if total_images else 0.0
    avg_classes_per_image   = _per_image_unique_classes(api, img_ann_map, is_lvis=is_lvis)

    # per-class average instances given presence
    per_class_avg_inst_per_img = {
        cid: (ann_per_class.get(cid, 0) / img_per_class[cid]) if img_per_class[cid] > 0 else 0.0
        for cid in cats.keys()
    }
    # global mean of that quantity
    overall_mean_inst_per_class_image = (
        float(np.mean([v for v in per_class_avg_inst_per_img.values()])) if per_class_avg_inst_per_img else 0.0
    )

    # per-class prevalence
    prevalence = {
        cid: (img_per_class[cid] / total_images if total_images else 0.0) for cid in cats.keys()
    }

    # LVIS tiers
    tier_per_cat = {}
    tier_summ = None
    if is_lvis and include_tiers:
        # LVIS category has 'frequency' in {'c','f','r'}
        tier_per_cat = {cid: cats[cid].get("frequency", "f") for cid in cats.keys()}
        tier_summ = {
            "num_classes": Counter([tier_per_cat[cid] for cid in cats.keys()]),
            "num_annotations": Counter(),
            "num_images": Counter()
        }
        for cid in cats.keys():
            t = tier_per_cat[cid]
            tier_summ["num_annotations"][t] += ann_per_class.get(cid, 0)
            tier_summ["num_images"][t] += img_per_class.get(cid, 0)

    # assemble per-class table
    per_class_rows = []
    for cid, cinfo in cats.items():
        row = {
            "category_id": int(cid),
            "name": cinfo.get("name", str(cid)),
            "ann_count": int(ann_per_class.get(cid, 0)),
            "img_count": int(img_per_class.get(cid, 0)),
            "prevalence": float(prevalence[cid]),
            "avg_instances_per_img_for_class": float(per_class_avg_inst_per_img[cid]),
        }
        if is_lvis:
            row["tier"] = tier_per_cat.get(cid, "f")
        per_class_rows.append(row)

    # sort table by ann_count desc
    per_class_rows.sort(key=lambda r: r["ann_count"], reverse=True)

    summary = {
        "total_images": total_images,
        "num_classes": num_classes,
        "total_annotations": total_annotations,
        "avg_instances_per_image": float(avg_instances_per_image),
        "avg_classes_per_image": float(avg_classes_per_image),
        "overall_mean_instances_per_class_per_image": float(overall_mean_inst_per_class_image),
        "size_buckets": dict(size_buckets),  # {"S":..,"M":..,"L":..}
    }
    if is_lvis and tier_summ is not None:
        # make plain ints
        summary["tiers"] = {
            "num_classes": {k:int(v) for k,v in tier_summ["num_classes"].items()},
            "num_annotations": {k:int(v) for k,v in tier_summ["num_annotations"].items()},
            "num_images": {k:int(v) for k,v in tier_summ["num_images"].items()},
        }

    return summary, per_class_rows


# -------------------------------
# Formatting / I/O
# -------------------------------

def save_csv(rows, path, include_tier=False):
    import csv
    cols = ["category_id", "name", "ann_count", "img_count",
            "prevalence", "avg_instances_per_img_for_class"]
    if include_tier:
        cols.insert(2, "tier")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            rr = {k: r.get(k, "") for k in cols}
            w.writerow(rr)

def pretty_print_summary(title, summary):
    print(f"\n=== {title} ===")
    print(f"total_images: {summary['total_images']}")
    print(f"num_classes: {summary['num_classes']}")
    print(f"total_annotations: {summary['total_annotations']}")
    print(f"avg_instances_per_image: {summary['avg_instances_per_image']:.4f}")
    print(f"avg_classes_per_image:   {summary['avg_classes_per_image']:.4f}")
    print(f"overall_mean_instances_per_class_per_image: "
          f"{summary['overall_mean_instances_per_class_per_image']:.4f}")
    if "size_buckets" in summary:
        sb = summary["size_buckets"]
        print(f"size_buckets S/M/L: {sb.get('S',0)} / {sb.get('M',0)} / {sb.get('L',0)}")
    if "tiers" in summary:
        t = summary["tiers"]
        print("tiers.num_classes:", t["num_classes"])
        print("tiers.num_annotations:", t["num_annotations"])
        print("tiers.num_images:", t["num_images"])


# -------------------------------
# CLI
# -------------------------------

def parse_args():
    p = argparse.ArgumentParser("Dataset statistics for COCO2017 / LVIS v1")
    p.add_argument("--dataset", choices=["coco", "lvis"], required=True,
                   help="Which dataset API to use for the annotation JSON.")
    p.add_argument("--ann_json", required=True, help="Path to annotation JSON (instances_* for COCO, lvis_v1_*.json for LVIS).")
    p.add_argument("--save_csv", type=str, help="Optional: path to write per-class CSV.")
    p.add_argument("--save_summary", type=str, help="Optional: path to write global summary JSON.")
    p.add_argument("--area_small", type=float, default=32.0, help="Small size (COCO) sqrt area threshold.")
    p.add_argument("--area_medium", type=float, default=96.0, help="Medium size (COCO) sqrt area threshold.")
    return p.parse_args()

def main():
    args = parse_args()
    areaRng = (args.area_small**2, args.area_medium**2, 1e5**2)

    if args.dataset == "coco":
        api = COCO(args.ann_json)
        summary, rows = compute_stats(api, areaRng=areaRng, is_lvis=False)
        pretty_print_summary("COCO stats", summary)
        if args.save_csv:
            save_csv(rows, args.save_csv, include_tier=False)
            print(f"[OK] wrote per-class CSV -> {args.save_csv}")
        if args.save_summary:
            with open(args.save_summary, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[OK] wrote summary JSON -> {args.save_summary}")

    else:  # LVIS
        if not _HAS_LVIS:
            raise RuntimeError("lvis package not available. Install lvis-api first: pip install lvis")
        api = LVIS(args.ann_json)
        summary, rows = compute_stats(api, areaRng=areaRng, is_lvis=True, include_tiers=True)
        pretty_print_summary("LVIS stats", summary)
        if args.save_csv:
            save_csv(rows, args.save_csv, include_tier=True)
            print(f"[OK] wrote per-class CSV -> {args.save_csv}")
        if args.save_summary:
            with open(args.save_summary, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[OK] wrote summary JSON -> {args.save_summary}")

if __name__ == "__main__":
    main()

