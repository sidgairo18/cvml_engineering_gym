#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
from pycocotools.coco import COCO

from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
)


def build_label_index(coco: COCO) -> Tuple[List[int], List[str], Dict[int, int], Dict[str, int]]:
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cats_sorted = sorted(cats, key=lambda x: x["id"])
    cat_ids_sorted = [c["id"] for c in cats_sorted]
    coco_labels = [c["name"] for c in cats_sorted]
    catid_to_index = {cid: i for i, cid in enumerate(cat_ids_sorted)}
    label_to_idx = {name: i for i, name in enumerate(coco_labels)}
    return cat_ids_sorted, coco_labels, catid_to_index, label_to_idx


def read_jsonl_predictions(pred_jsonl: str) -> Dict[int, Dict[str, Any]]:
    """
    Returns dict: img_id -> record
    If duplicates exist (e.g., overlapping shards), last occurrence wins.
    """
    by_img: Dict[int, Dict[str, Any]] = {}
    with open(pred_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "img_id" not in rec:
                continue
            img_id = int(rec["img_id"])
            by_img[img_id] = rec
    return by_img


def gt_multihot_for_img_ids(coco: COCO, img_ids: List[int], catid_to_index: Dict[int, int], num_classes: int) -> np.ndarray:
    Y = np.zeros((len(img_ids), num_classes), dtype=np.int32)
    for i, iid in enumerate(img_ids):
        ann_ids = coco.getAnnIds(imgIds=[iid])
        anns = coco.loadAnns(ann_ids)
        present = {a["category_id"] for a in anns}
        for cid in present:
            if cid in catid_to_index:
                Y[i, catid_to_index[cid]] = 1
    return Y


def pred_multihot_from_records(
    img_ids: List[int],
    records: Dict[int, Dict[str, Any]],
    label_to_idx: Dict[str, int],
    num_classes: int,
    key: str = "pred_labels",
) -> np.ndarray:
    Y = np.zeros((len(img_ids), num_classes), dtype=np.int32)
    for i, iid in enumerate(img_ids):
        rec = records.get(iid, {})
        labs = rec.get(key, []) or []
        for lab in labs:
            if lab in label_to_idx:
                Y[i, label_to_idx[lab]] = 1
    return Y


def compute_per_class_stats(Y_true: np.ndarray, Y_pred: np.ndarray, class_names: List[str]) -> List[Dict[str, Any]]:
    """
    Computes TP/FP/FN, precision/recall/F1, AP (degenerate) per class.
    AP is computed using binary predictions as "scores" -> useful for quick comparisons,
    but NOT a proper detection-style mAP.
    """
    out = []
    C = Y_true.shape[1]
    for c in range(C):
        yt = Y_true[:, c].astype(int)
        yp = Y_pred[:, c].astype(int)

        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())

        support = int(yt.sum())
        pred_pos = int(yp.sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # AP requires at least one positive in GT to be meaningful
        if support > 0:
            ap = float(average_precision_score(yt, yp))
        else:
            ap = float("nan")

        out.append({
            "class": class_names[c],
            "support": support,
            "pred_pos": pred_pos,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "ap_bin": ap,
        })
    return out


def print_topk(per_class: List[Dict[str, Any]], metric: str, k: int, descending: bool = True, min_support: int = 1):
    xs = [r for r in per_class if (r["support"] >= min_support)]
    xs = [r for r in xs if not (isinstance(r.get(metric), float) and np.isnan(r.get(metric)))]
    xs.sort(key=lambda r: r.get(metric, -1.0), reverse=descending)
    print(f"\nTop {k} by {metric} (min_support={min_support}):")
    for r in xs[:k]:
        print(f"  {r['class']:<18} {metric}={r[metric]:.4f}  sup={r['support']}  pred={r['pred_pos']}  tp={r['tp']} fp={r['fp']} fn={r['fn']}")

    xs.sort(key=lambda r: r.get(metric, 1e9), reverse=False)
    print(f"\nBottom {k} by {metric} (min_support={min_support}):")
    for r in xs[:k]:
        print(f"  {r['class']:<18} {metric}={r[metric]:.4f}  sup={r['support']}  pred={r['pred_pos']}  tp={r['tp']} fp={r['fp']} fn={r['fn']}")


def save_csv(per_class: List[Dict[str, Any]], out_csv: str):
    import csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = list(per_class[0].keys()) if per_class else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_class:
            w.writerow(r)


def save_json(per_class: List[Dict[str, Any]], out_json: str):
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(per_class, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", type=str, required=True, help="COCO-format annotation JSON (instances_*.json)")
    parser.add_argument("--pred_jsonl", type=str, required=True, help="Merged JSONL predictions file")
    parser.add_argument("--pred_key", type=str, default="pred_labels", help="Which JSONL field to use for predicted labels")
    parser.add_argument("--min_support", type=int, default=1, help="Minimum GT positives required to be included in top/bottom lists")
    parser.add_argument("--topk", type=int, default=10, help="How many classes to show in top/bottom lists")
    parser.add_argument("--save_csv", type=str, default="", help="Optional CSV path for per-class breakdown")
    parser.add_argument("--save_json", type=str, default="", help="Optional JSON path for per-class breakdown")
    args = parser.parse_args()

    assert os.path.isfile(args.ann_file), args.ann_file
    assert os.path.isfile(args.pred_jsonl), args.pred_jsonl

    coco = COCO(args.ann_file)
    _, coco_labels, catid_to_index, label_to_idx = build_label_index(coco)

    records = read_jsonl_predictions(args.pred_jsonl)
    img_ids = sorted(records.keys())
    if len(img_ids) == 0:
        print("No predictions found in JSONL.")
        return

    print("Annotation:", args.ann_file)
    print("Pred JSONL :", args.pred_jsonl)
    print("Num predicted images:", len(img_ids))
    print("Num classes:", len(coco_labels))

    Y_true = gt_multihot_for_img_ids(coco, img_ids, catid_to_index, num_classes=len(coco_labels))
    Y_pred = pred_multihot_from_records(img_ids, records, label_to_idx, num_classes=len(coco_labels), key=args.pred_key)

    # Overall metrics
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(Y_true, Y_pred, average="micro", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(Y_true, Y_pred, average="macro", zero_division=0)

    # Macro mAP (degenerate using bin preds as scores) over classes that appear in GT
    aps = []
    for c in range(Y_true.shape[1]):
        if Y_true[:, c].sum() == 0:
            continue
        aps.append(float(average_precision_score(Y_true[:, c], Y_pred[:, c])))
    map_macro = float(np.mean(aps)) if aps else 0.0

    print("\n=== Overall ===")
    print(f"Micro P/R/F1: {p_micro:.4f} / {r_micro:.4f} / {f1_micro:.4f}")
    print(f"Macro P/R/F1: {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
    print(f"Macro mAP (bin-scores): {map_macro:.4f}")
    print("(Note: AP/mAP here uses binary preds as scores; for real mAP you need per-class confidence scores.)")

    # Per-class breakdown
    per_class = compute_per_class_stats(Y_true, Y_pred, coco_labels)

    print_topk(per_class, metric="f1", k=args.topk, descending=True, min_support=args.min_support)
    print_topk(per_class, metric="ap_bin", k=args.topk, descending=True, min_support=args.min_support)

    if args.save_csv:
        save_csv(per_class, args.save_csv)
        print("\nWrote per-class CSV:", args.save_csv)

    if args.save_json:
        save_json(per_class, args.save_json)
        print("Wrote per-class JSON:", args.save_json)


if __name__ == "__main__":
    main()

'''
1) Merge shards (as you already planned)
cat outputs/shards/val2017_llava7b_two_gated_shard_*.jsonl > outputs/val2017_llava7b_two_gated_merged.jsonl

2) Compute global metrics + per-class breakdown
python compute_metrics_from_jsonl.py \
  --ann_file /BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco/annotations/instances_val2017.json \
  --pred_jsonl outputs/val2017_llava7b_two_gated_merged.jsonl \
  --topk 15 \
  --min_support 5 \
  --save_csv outputs/per_class_val2017_llava7b_two_gated.csv \
  --save_json outputs/per_class_val2017_llava7b_two_gated.json

'''
