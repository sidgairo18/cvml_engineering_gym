#!/usr/bin/env python3
"""
Compute oriented co-occurrence tables (NC and PMI) for COCO/LVIS.

Outputs:
  - <out_dir>/cooccur_oriented.csv
      columns:
        anchor_id,anchor,partner_id,partner,
        n_img,count_anchor,count_partner,count_both,
        nc,pmi

  - <out_dir>/cooccur_unordered.csv  (optional compact form with nc_ab/nc_ba and pmi)

  - <out_dir>/nc_thresholds.json     (per-class percentiles for NC)

Notes
-----
- NC(a->b) = P(b | a) = count_both / count_anchor
- PMI(a,b) = log( P(a,b) / (P(a)P(b)) ), with Laplace smoothing:
    pa  = (n_a  + alpha) / (N + alpha * V)
    pb  = (n_b  + alpha) / (N + alpha * V)
    pab = (n_ab + alpha) / (N + alpha * (V*V))
- Use --min_class and --min_pair to filter very sparse classes/pairs.
- Use --auto_min to derive dataset-aware minima from the distribution
  (quantile-based) and then clamp with hard floors.

Python 3.8-compatible.
"""
import argparse, json, math, os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_annotations(ann_path: str) -> Tuple[Dict[int, dict], Dict[int, str], List[dict]]:
    with open(ann_path, "r") as f:
        ann = json.load(f)
    imgs = {int(im["id"]): im for im in ann["images"]}
    cats = {int(c["id"]): c["name"] for c in ann["categories"]}
    anns = ann["annotations"]
    return imgs, cats, anns


def build_image_categories(
    anns: List[dict],
    min_area: float = 1.0,
    iscrowd_ok: bool = False,
) -> Dict[int, set]:
    img2cats: Dict[int, set] = defaultdict(set)
    for a in anns:
        if (not iscrowd_ok) and int(a.get("iscrowd", 0)) == 1:
            continue
        if float(a.get("area", 1.0)) < min_area:
            continue
        img2cats[int(a["image_id"])].add(int(a["category_id"]))
    return img2cats


def auto_minima(
    count_per_class: Counter,
    pair_count: Counter,
    q_class: float,
    q_pair: float,
    floor_class: int,
    floor_pair: int,
) -> Tuple[int, int]:
    """
    Pick dataset-aware minima using quantiles, then clamp with floors.
    """
    cls_vals = np.array(list(count_per_class.values()), dtype=np.int64)
    pr_vals = np.array(list(pair_count.values()), dtype=np.int64) if len(pair_count) else np.array([0])

    q_min_class = int(np.floor(np.quantile(cls_vals, q_class))) if len(cls_vals) else 0
    q_min_pair = int(np.floor(np.quantile(pr_vals, q_pair))) if len(pr_vals) else 0

    return max(q_min_class, floor_class), max(q_min_pair, floor_pair)


def compute_tables(
    img2cats: Dict[int, set],
    cats_id2name: Dict[int, str],
    min_class: int,
    min_pair: int,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Returns (df_oriented, df_unordered, count_per_class).
    """
    N = len(img2cats)
    # counts
    count_per_class = Counter()
    pair_count = Counter()  # (a<b)
    for _, cs in img2cats.items():
        for c in cs:
            count_per_class[c] += 1
        cs_sorted = sorted(cs)
        for i in range(len(cs_sorted)):
            for j in range(i + 1, len(cs_sorted)):
                pair_count[(cs_sorted[i], cs_sorted[j])] += 1

    # Filter eligible classes (appear in >= min_class images)
    eligible: set = {c for c, n in count_per_class.items() if n >= min_class}
    V = max(1, len(eligible))

    # Build unordered rows first (one per a<b)
    rows_un: List[dict] = []
    for (a, b), n_ab in pair_count.items():
        if (a not in eligible) or (b not in eligible):
            continue
        if n_ab < min_pair:
            continue
        n_a = count_per_class[a]
        n_b = count_per_class[b]

        # smoothed probabilities for PMI
        pa = (n_a + alpha) / (N + alpha * V)
        pb = (n_b + alpha) / (N + alpha * V)
        pab = (n_ab + alpha) / (N + alpha * (V * V))
        pmi = math.log(pab / (pa * pb) + 1e-12)

        # oriented NC (both directions but in one row here)
        nc_ab = n_ab / max(1, n_a)
        nc_ba = n_ab / max(1, n_b)

        rows_un.append(
            dict(
                cat_a_id=a,
                cat_a_name=cats_id2name.get(a, str(a)),
                cat_b_id=b,
                cat_b_name=cats_id2name.get(b, str(b)),
                N_img=N,
                count_a=n_a,
                count_b=n_b,
                count_ab=n_ab,
                nc_ab=nc_ab,
                nc_ba=nc_ba,
                pmi=pmi,
            )
        )
    df_un = pd.DataFrame(rows_un).sort_values("pmi", ascending=False).reset_index(drop=True)

    # Oriented rows (two per unordered row)
    rows_or: List[dict] = []
    for _, r in df_un.iterrows():
        a = int(r["cat_a_id"])
        b = int(r["cat_b_id"])
        # row: a->b
        rows_or.append(
            dict(
                anchor_id=a,
                anchor=cats_id2name.get(a, str(a)),
                partner_id=b,
                partner=cats_id2name.get(b, str(b)),
                n_img=int(r["N_img"]),
                count_anchor=int(r["count_a"]),
                count_partner=int(r["count_b"]),
                count_both=int(r["count_ab"]),
                nc=float(r["nc_ab"]),
                pmi=float(r["pmi"]),
            )
        )
        # row: b->a
        rows_or.append(
            dict(
                anchor_id=b,
                anchor=cats_id2name.get(b, str(b)),
                partner_id=a,
                partner=cats_id2name.get(a, str(a)),
                n_img=int(r["N_img"]),
                count_anchor=int(r["count_b"]),
                count_partner=int(r["count_a"]),
                count_both=int(r["count_ab"]),
                nc=float(r["nc_ba"]),
                pmi=float(r["pmi"]),
            )
        )
    df_or = pd.DataFrame(rows_or).sort_values(["anchor", "nc"], ascending=[True, False]).reset_index(drop=True)
    return df_or, df_un, dict(count_per_class)


def summarize_nc_thresholds(df_oriented: pd.DataFrame, q_hi: float, q_lo: float) -> Dict[str, dict]:
    """
    For each anchor, compute NC percentiles (hi, lo) and basic stats.
    """
    out: Dict[str, dict] = {}
    for anchor, grp in df_oriented.groupby("anchor"):
        vals = grp["nc"].values
        if len(vals) == 0:
            continue
        out[anchor] = dict(
            n_partners=int(len(vals)),
            nc_mean=float(np.mean(vals)),
            nc_median=float(np.median(vals)),
            tau_hi=float(np.quantile(vals, q_hi)),
            tau_lo=float(np.quantile(vals, q_lo)),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Path to COCO/LVIS instances JSON.")
    ap.add_argument("--out_dir", required=True)

    # Filtering
    ap.add_argument("--min_area", type=float, default=1.0)
    ap.add_argument("--min_class", type=int, default=None, help="Absolute minimum images per class. If None and --auto_min, use quantile.")
    ap.add_argument("--min_pair", type=int, default=None, help="Absolute minimum images per pair. If None and --auto_min, use quantile.")
    ap.add_argument("--auto_min", action="store_true", help="Derive minima from dataset distribution (quantiles) and clamp with floors.")
    ap.add_argument("--q_class", type=float, default=0.10, help="Quantile for class minima when --auto_min (e.g., 0.10 = 10th pct).")
    ap.add_argument("--q_pair", type=float, default=0.10, help="Quantile for pair minima when --auto_min.")
    ap.add_argument("--floor_class", type=int, default=10, help="Hard lower bound for class minima when --auto_min.")
    ap.add_argument("--floor_pair", type=int, default=5, help="Hard lower bound for pair minima when --auto_min.")

    # PMI
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha for PMI.")
    # Save compact unordered CSV too
    ap.add_argument("--save_unordered", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    imgs, cats_id2name, anns = load_annotations(args.ann)
    img2cats = build_image_categories(anns, min_area=args.min_area)

    # First pass counts for auto minima
    N = len(img2cats)
    count_per_class = Counter()
    pair_count = Counter()
    for _, cs in img2cats.items():
        for c in cs:
            count_per_class[c] += 1
        cs_sorted = sorted(cs)
        for i in range(len(cs_sorted)):
            for j in range(i + 1, len(cs_sorted)):
                pair_count[(cs_sorted[i], cs_sorted[j])] += 1

    if args.auto_min:
        auto_min_class, auto_min_pair = auto_minima(
            count_per_class, pair_count,
            q_class=args.q_class, q_pair=args.q_pair,
            floor_class=args.floor_class, floor_pair=args.floor_pair
        )
        min_class = auto_min_class if args.min_class is None else max(args.min_class, auto_min_class)
        min_pair = auto_min_pair if args.min_pair is None else max(args.min_pair, auto_min_pair)
    else:
        # pure absolute minima (or sensible defaults)
        min_class = args.min_class if args.min_class is not None else 10
        min_pair = args.min_pair if args.min_pair is not None else 5

    df_or, df_un, count_per_class_dict = compute_tables(
        img2cats, cats_id2name, min_class=min_class, min_pair=min_pair, alpha=args.alpha
    )

    out_or = os.path.join(args.out_dir, "cooccur_oriented.csv")
    df_or.to_csv(out_or, index=False)
    print(f"[OK] Oriented co-occurrence -> {out_or} (rows={len(df_or)})")

    if args.save_unordered:
        out_un = os.path.join(args.out_dir, "cooccur_unordered.csv")
        df_un.to_csv(out_un, index=False)
        print(f"[OK] Unordered (compact) -> {out_un} (rows={len(df_un)})")

    # Per-anchor NC thresholds & summary
    nc_summ = summarize_nc_thresholds(df_or, q_hi=0.90, q_lo=0.10)
    with open(os.path.join(args.out_dir, "nc_thresholds.json"), "w") as f:
        json.dump(
            dict(
                N_images=N,
                min_class=min_class,
                min_pair=min_pair,
                alpha=args.alpha,
                anchors=nc_summ,
            ),
            f,
            indent=2,
        )
    print(f"[OK] NC thresholds/stats -> {os.path.join(args.out_dir, 'nc_thresholds.json')}")


if __name__ == "__main__":
    main()

