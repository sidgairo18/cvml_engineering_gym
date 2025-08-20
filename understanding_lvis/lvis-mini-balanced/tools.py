# mini_lvis_balanced/tools.py
import json
import numpy as np
from collections import defaultdict
from lvis import LVIS
import matplotlib.pyplot as plt

def write_lvis_subset(lvis: LVIS, imgs_best_sample, out_json_path):
    subset = {
        "info": lvis.dataset.get("info", {}),
        "licenses": lvis.dataset.get("licenses", []),
        "categories": lvis.dataset.get("categories", []),
        "images": [],
        "annotations": [],
    }
    sel = set(imgs_best_sample.keys())
    for im_id in sel:
        subset["images"].append(lvis.imgs[im_id])
        for aid in lvis.img_ann_map[im_id]:
            subset["annotations"].append(lvis.anns[aid])
    with open(out_json_path, "w") as f:
        json.dump(subset, f)
    return out_json_path

def kept_categories(imgs_best_sample):
    s = set()
    for anns in imgs_best_sample.values():
        for a in anns:
            s.add(a["category_id"])
    return sorted(s)

def prune_gt(in_json, out_json, kept_cat_ids, drop_empty_images=False):
    kept = set(kept_cat_ids)
    with open(in_json, "r") as f:
        data = json.load(f)
    cats = [c for c in data["categories"] if c["id"] in kept]
    anns = [a for a in data["annotations"] if a["category_id"] in kept]
    if drop_empty_images:
        has = set(a["image_id"] for a in anns)
        imgs = [im for im in data["images"] if im["id"] in has]
    else:
        imgs = data["images"]
    out = {"info": data.get("info",{}), "licenses": data.get("licenses", []),
           "images": imgs, "annotations": anns, "categories": cats}
    with open(out_json, "w") as f:
        json.dump(out, f)
    return out_json

def prune_preds(in_pred, out_pred, kept_cat_ids):
    kept = set(kept_cat_ids)
    with open(in_pred, "r") as f:
        preds = json.load(f)
    pruned = [p for p in preds if p.get("category_id") in kept]
    with open(out_pred, "w") as f:
        json.dump(pruned, f)
    return out_pred

def plot_tier_bars(train_stats, val_stats, save="lvis_tiers_train_val.png"):
    tiers = ["c","f","r"]; names = ["Common","Frequent","Rare"]
    def _bars(ax, key, title):
        tr = [train_stats[key][t] for t in tiers]
        va = [val_stats[key][t] for t in tiers]
        x = np.arange(3)
        ax.bar(x-0.2, tr, width=0.4, label="Train")
        ax.bar(x+0.2, va, width=0.4, label="Val")
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_title(title); ax.legend()
    fig, axs = plt.subplots(1,3, figsize=(16,5))
    _bars(axs[0], "num_classes", "Number of Classes")
    _bars(axs[1], "num_annotations", "Number of Annotations")
    _bars(axs[2], "num_images", "Number of Images")
    fig.suptitle("LVIS v1 — Tier Distributions (Train vs Val)", fontsize=14)
    fig.tight_layout(); fig.savefig(save, dpi=200); plt.close(fig)
    return save

