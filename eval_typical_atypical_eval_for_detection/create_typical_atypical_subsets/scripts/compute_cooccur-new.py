#!/usr/bin/env python3
"""
Compute oriented co-occurrence tables (NC, PMI, TS) for COCO/LVIS-like instances JSON.

Outputs:
  - <out_dir>/cooccur_oriented.csv
      columns:
        anchor_id,anchor,partner_id,partner,
        n_img,count_anchor,count_partner,count_both,
        nc,pmi,ts

  - <out_dir>/cooccur_unordered.csv  (if --save_unordered)
      columns with nc_ab, nc_ba and pmi per unordered pair (a<b)

  - <out_dir>/nc_thresholds.json     (per-anchor percentiles for NC)

Optional plots:
  --plot_topk K with --plot_metric {ts|pmi|nc}  -> <out_dir>/topk_<metric>.png
  --plot_heatmap PATH with --metric_for_heatmap {ts|pmi|nc} and class caps

Notes
-----
- NC(a->b) = P(b | a) = count_both / count_anchor
- PMI(a,b) = log( P(a,b) / (P(a)P(b)) ), with Laplace smoothing:
    pa  = (n_a  + alpha) / (N + alpha * V)
    pb  = (n_b  + alpha) / (N + alpha * V)
    pab = (n_ab + alpha) / (N + alpha * (V*V))
- TS (directed typicality, “TS→”) combines evidence (z-scored log-odds ratio),
  conditional typicality (row-wise normalized NC), and a support gate:

    lor   = log( (n11 * n00) / (n10 * n01 + eps) + eps )
    zlor  = (lor - mean(lor_row)) / (std(lor_row)+1e-6)      # evidence vs. independence
    nc'   = NC(a->b) / max(NC(a->*))
    w     = sigmoid( (n11 - T) / slope )
    TS→   = max(0, zlor) * w * nc'

Python 3.8-compatible.
"""
import argparse, json, math, os, random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- I/O & counting ------------------------- #

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


# ------------------------- metrics: PMI, NC, TS ------------------------- #

def _pmi_smoothed(n_a: int, n_b: int, n_ab: int, N: int, V: int, alpha: float) -> float:
    pa  = (n_a  + alpha) / (N + alpha * V)
    pb  = (n_b  + alpha) / (N + alpha * V)
    pab = (n_ab + alpha) / (N + alpha * (V * V))
    return math.log(pab / (pa * pb) + 1e-12)


def _log_odds_ratio(n11, n10, n01, n00) -> float:
    # lor = log( (n11 * n00) / (n10 * n01) ), stabilized
    num = (n11 * n00) + 1e-12
    den = (n10 * n01) + 1e-12
    return float(np.log(num / den))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_tables(
    img2cats: Dict[int, set],
    cats_id2name: Dict[int, str],
    min_class: int,
    min_pair: int,
    alpha: float,
    ts_support_T: int,
    ts_support_slope: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Returns (df_oriented, df_unordered, count_per_class).
    """
    N = len(img2cats)

    # counts
    count_per_class = Counter()
    pair_count = Counter()  # unordered (a<b)
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

    # Pre-compute n10/n01/n00 components for lor
    # n11 = co-occur
    # n10 = count(a) - n11
    # n01 = count(b) - n11
    # n00 = N - n11 - n10 - n01
    rows_un: List[dict] = []
    for (a, b), n_ab in pair_count.items():
        if (a not in eligible) or (b not in eligible):
            continue
        if n_ab < min_pair:
            continue
        n_a = count_per_class[a]
        n_b = count_per_class[b]

        pmi = _pmi_smoothed(n_a, n_b, n_ab, N, V, alpha)
        nc_ab = n_ab / max(1, n_a)
        nc_ba = n_ab / max(1, n_b)

        n11 = n_ab
        n10 = max(0, n_a - n_ab)
        n01 = max(0, n_b - n_ab)
        n00 = max(0, N - n11 - n10 - n01)
        lor = _log_odds_ratio(n11, n10, n01, n00)

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
                lor=lor,
            )
        )
    df_un = pd.DataFrame(rows_un).sort_values("pmi", ascending=False).reset_index(drop=True)

    # Oriented rows (two per unordered row) with TS fields
    rows_or: List[dict] = []
    for _, r in df_un.iterrows():
        a = int(r["cat_a_id"])
        b = int(r["cat_b_id"])
        n_img = int(r["N_img"])
        n_a   = int(r["count_a"])
        n_b   = int(r["count_b"])
        n_ab  = int(r["count_ab"])
        nc_ab = float(r["nc_ab"])
        nc_ba = float(r["nc_ba"])
        pmi   = float(r["pmi"])
        lor   = float(r["lor"])

        # support gate
        w_ab = _sigmoid((n_ab - ts_support_T) / max(1e-6, ts_support_slope))

        rows_or.append(
            dict(
                anchor_id=a, anchor=cats_id2name.get(a, str(a)),
                partner_id=b, partner=cats_id2name.get(b, str(b)),
                n_img=n_img,
                count_anchor=n_a, count_partner=n_b, count_both=n_ab,
                nc=nc_ab, pmi=pmi, lor=lor, w_support=w_ab,
            )
        )
        # reverse
        w_ba = _sigmoid((n_ab - ts_support_T) / max(1e-6, ts_support_slope))
        rows_or.append(
            dict(
                anchor_id=b, anchor=cats_id2name.get(b, str(b)),
                partner_id=a, partner=cats_id2name.get(a, str(a)),
                n_img=n_img,
                count_anchor=n_b, count_partner=n_a, count_both=n_ab,
                nc=nc_ba, pmi=pmi, lor=lor, w_support=w_ba,
            )
        )

    df_or = pd.DataFrame(rows_or)

    if df_or.empty:
        return df_or, df_un, dict(count_per_class)

    # Row-wise z-score of LOR by anchor -> zlor
    def zscore_by_anchor(g: pd.DataFrame) -> pd.DataFrame:
        m = float(np.mean(g["lor"].values))
        s = float(np.std(g["lor"].values)) + 1e-6
        g = g.copy()
        g["zlor"] = (g["lor"] - m) / s
        return g

    df_or = df_or.groupby("anchor", group_keys=False).apply(zscore_by_anchor)

    # Row-wise normalized NC: nc' = nc / max_row(nc)
    def nc_prime(g: pd.DataFrame) -> pd.DataFrame:
        mx = float(np.max(g["nc"].values)) + 1e-12
        g = g.copy()
        g["nc_pct"] = g["nc"] / mx
        return g

    df_or = df_or.groupby("anchor", group_keys=False).apply(nc_prime)

    # Directed TS→ = max(0, zlor) * w_support * nc'
    df_or["ts"] = np.maximum(0.0, df_or["zlor"].astype(float)) * df_or["w_support"].astype(float) * df_or["nc_pct"].astype(float)

    # Final sort for readability
    df_or = df_or.sort_values(["anchor", "ts"], ascending=[True, False]).reset_index(drop=True)
    return df_or, df_un, dict(count_per_class)


# ------------------------- summaries & plots ------------------------- #

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


def _plot_topk(df: pd.DataFrame, metric: str, top_k: int, out_dir: str) -> Optional[str]:
    if df.empty or metric not in df.columns or top_k <= 0:
        return None
    sub = df.sort_values(metric, ascending=False).head(top_k)
    labels = [f"{r['anchor']}→{r['partner']} (AB:{int(r['count_both'])})" for _, r in sub.iterrows()]
    vals = [float(r[metric]) for _, r in sub.iterrows()]
    xs = list(range(len(labels)))
    plt.figure(figsize=(max(12, 0.6 * len(labels)), 7))
    plt.bar(xs, vals)
    plt.xticks(xs, labels, rotation=70, ha="right")
    plt.ylabel(metric.upper())
    plt.title(f"Top-{top_k} by {metric.upper()}")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"topk_{metric}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _build_heatmap_matrix(df: pd.DataFrame, metric: str, max_cats: Optional[int], seed: Optional[int]) -> Tuple[List[str], np.ndarray]:
    if df.empty:
        return [], np.zeros((0, 0), dtype=float)
    # frequency proxy to choose category set
    freq: Dict[str, int] = {}
    for _, r in df.iterrows():
        a = r["anchor"]; b = r["partner"]
        freq[a] = max(freq.get(a, 0), int(r["count_anchor"]))
        freq[b] = max(freq.get(b, 0), int(r["count_partner"]))
    names_pool = list(freq.keys())
    if max_cats is not None and max_cats > 0 and len(names_pool) > max_cats:
        rng = random.Random(seed)
        names_pool = rng.sample(names_pool, k=max_cats)
    names = sorted(names_pool, key=lambda n: (-freq[n], n))
    idx = {n: i for i, n in enumerate(names)}
    C = len(names)
    M = np.full((C, C), np.nan, dtype=float)
    for _, r in df.iterrows():
        a = r["anchor"]; b = r["partner"]
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        val = float(r.get(metric, np.nan))
        if np.isnan(M[i, j]) or val > M[i, j]:
            M[i, j] = val; M[j, i] = val
    np.fill_diagonal(M, 0.0)
    return names, M


def _save_heatmap(names: List[str], M: np.ndarray, out_path: str, center_zero: bool = True):
    if M.size == 0:
        return
    finite_vals = M[np.isfinite(M)]
    if finite_vals.size == 0:
        vlim = 1.0
    else:
        vmax = float(np.nanmax(np.abs(finite_vals)))
        vlim = vmax if vmax > 0 else 1.0
    plt.figure(figsize=(max(6.0, min(18.0, 0.25 * len(names))),) * 2, dpi=200)
    cmap = plt.get_cmap("bwr").copy()
    cmap.set_bad("0.85")
    im = plt.imshow(M, vmin=-vlim if center_zero else np.nanmin(M), vmax=vlim if center_zero else np.nanmax(M), aspect="equal", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="score")
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.title("Category × Category")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# ------------------------- main ------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Path to instances JSON.")
    ap.add_argument("--out_dir", required=True)

    # Filtering
    ap.add_argument("--min_area", type=float, default=1.0)
    ap.add_argument("--min_class", type=int, default=None, help="Absolute minimum images per class. If None and --auto_min, use quantile.")
    ap.add_argument("--min_pair", type=int, default=None, help="Absolute minimum images per pair. If None and --auto_min, use quantile.")
    ap.add_argument("--auto_min", action="store_true", help="Derive minima from dataset distribution (quantiles) and clamp with floors.")
    ap.add_argument("--q_class", type=float, default=0.10, help="Quantile for class minima when --auto_min.")
    ap.add_argument("--q_pair", type=float, default=0.10, help="Quantile for pair minima when --auto_min.")
    ap.add_argument("--floor_class", type=int, default=10, help="Hard lower bound for class minima when --auto_min.")
    ap.add_argument("--floor_pair", type=int, default=5, help="Hard lower bound for pair minima when --auto_min.")

    # PMI smoothing
    ap.add_argument("--alpha", type=float, default=1.0)

    # TS support gate
    ap.add_argument("--ts_support_T", type=int, default=15)
    ap.add_argument("--ts_support_slope", type=float, default=10.0)

    # Save unordered too?
    ap.add_argument("--save_unordered", action="store_true")

    # Top-K plot
    ap.add_argument("--plot_topk", type=int, default=0)
    ap.add_argument("--plot_metric", type=str, default="ts", choices=["ts", "pmi", "nc"])

    # Heatmap
    ap.add_argument("--plot_heatmap", default=None, help="If set, save a heatmap PNG to this path.")
    ap.add_argument("--metric_for_heatmap", default="ts", choices=["ts", "pmi", "nc"])
    ap.add_argument("--heatmap_max_cats", type=int, default=None)
    ap.add_argument("--heatmap_pad", default="nan", choices=["nan", "zero"])  # reserved, currently unused (matrix uses NaN)
    ap.add_argument("--heatmap_sample_seed", type=int, default=None)

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
        min_class = args.min_class if args.min_class is not None else 10
        min_pair = args.min_pair if args.min_pair is not None else 5

    df_or, df_un, _ = compute_tables(
        img2cats, cats_id2name,
        min_class=min_class, min_pair=min_pair, alpha=args.alpha,
        ts_support_T=args.ts_support_T, ts_support_slope=args.ts_support_slope
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

    # Plots
    if args.plot_topk and len(df_or) > 0:
        path = _plot_topk(df_or, args.plot_metric, args.plot_topk, args.out_dir)
        if path:
            print(f"[OK] Top-{args.plot_topk} plot ({args.plot_metric}) -> {path}")

    if args.plot_heatmap:
        names, M = _build_heatmap_matrix(df_or, args.metric_for_heatmap, args.heatmap_max_cats, args.heatmap_sample_seed)
        if len(names) == 0:
            print("[WARN] Heatmap requested but no categories eligible; skipping.")
        else:
            _save_heatmap(names, M, args.plot_heatmap, center_zero=True)
            print(f"[OK] Heatmap saved -> {args.plot_heatmap} (C={len(names)})")


if __name__ == "__main__":
    main()

