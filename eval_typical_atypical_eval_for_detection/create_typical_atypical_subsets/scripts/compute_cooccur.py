#!/usr/bin/env python3
"""
Compute co-occurrence tables (NC, PMI, ZLOR) + directed Typicality Score (TS→) for COCO/LVIS.

Outputs
-------
<out_dir>/cooccur_oriented.csv
    columns:
      anchor_id,anchor,partner_id,partner,
      n_img,count_anchor,count_partner,count_both,
      nc, nc_prime, nc_row_pct,
      pmi, zlor, ts_dir

<out_dir>/cooccur_unordered.csv    (optional; one row per unordered pair a<b)
<out_dir>/nc_thresholds.json       (per-anchor NC percentiles & stats)
Optional: Top-K bar plot for a chosen metric (pmi|nc|ts) and PMI heatmap.

Notes
-----
- NC(a->b) = P(b | a) = count_both / count_anchor
- Laplace-smoothed conditional:
    nc'(a->b) = (n11 + alpha) / (n_a + 2*alpha)
- PMI smoothing:
    pa  = (n_a  + alpha) / (N + alpha*V)
    pb  = (n_b  + alpha) / (N + alpha*V)
    pab = (n11 + alpha) / (N + alpha*(V*V))
    PMI = log( pab / (pa*pb) )
- ZLOR uses Haldane-Anscombe correction and standard error from 2x2 table.
- TS→(a,b) = max(0, evidence(a,b)) * support_gate(n11; T, slope) * direction_term
    evidence  := ZLOR or PMI
    direction := nc'  or row-percentile of NC in the anchor's row

Python 3.8-compatible.
"""
import argparse, json, math, os, random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


# -------------------------
# Data loading / preprocessing
# -------------------------

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


# -------------------------
# Utilities
# -------------------------

def auto_minima(
    count_per_class: Counter,
    pair_count: Counter,
    q_class: float,
    q_pair: float,
    floor_class: int,
    floor_pair: int,
) -> Tuple[int, int]:
    """Pick dataset-aware minima using quantiles, then clamp with floors."""
    cls_vals = np.array(list(count_per_class.values()), dtype=np.int64)
    pr_vals = np.array(list(pair_count.values()), dtype=np.int64) if len(pair_count) else np.array([0])

    q_min_class = int(np.floor(np.quantile(cls_vals, q_class))) if len(cls_vals) else 0
    q_min_pair  = int(np.floor(np.quantile(pr_vals,  q_pair ))) if len(pr_vals)  else 0

    return max(q_min_class, floor_class), max(q_min_pair, floor_pair)


def log_odds_ratio_z(n11, n10, n01, n00) -> float:
    """
    Z-score of the log-odds ratio with Haldane-Anscombe correction (+0.5 to all cells).
    """
    a = n11 + 0.5
    b = n10 + 0.5
    c = n01 + 0.5
    d = n00 + 0.5
    lor = math.log((a * d) / (b * c))
    se  = math.sqrt(1.0/a + 1.0/b + 1.0/c + 1.0/d)
    return lor / se


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def row_percentiles(values: np.ndarray) -> np.ndarray:
    """
    Convert a 1D array to percentile ranks in [0,1].
    Ties receive their average rank. Stable and SciPy-free.
    """
    if values.size <= 1:
        return np.ones_like(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")   # stable
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)

    # Handle ties: average rank for equal values
    v = values[order]
    i = 0
    while i < v.size:
        j = i + 1
        while j < v.size and v[j] == v[i]:
            j += 1
        # average rank for [i, j)
        avg = (i + (j - 1)) / 2.0
        ranks[i:j] = avg
        i = j
    denom = max(1.0, (values.size - 1))
    return ranks / denom


# -------------------------
# Core computation
# -------------------------

def compute_tables(
    img2cats: Dict[int, set],
    cats_id2name: Dict[int, str],
    min_class: int,
    min_pair: int,
    alpha: float,
    ts_evidence: str,
    ts_use_row_percentile: bool,
    support_T: float,
    support_slope: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Returns (df_oriented, df_unordered, count_per_class).
    df_oriented contains NC, PMI, ZLOR, nc_prime, nc_row_pct, and TS→.
    """
    N = len(img2cats)

    # Counts
    count_per_class = Counter()
    pair_count = Counter()  # keys are unordered (a<b)
    for _, cs in img2cats.items():
        for c in cs:
            count_per_class[c] += 1
        cs_sorted = sorted(cs)
        for i in range(len(cs_sorted)):
            for j in range(i + 1, len(cs_sorted)):
                pair_count[(cs_sorted[i], cs_sorted[j])] += 1

    # Filter eligible classes
    eligible: set = {c for c, n in count_per_class.items() if n >= min_class}
    V = max(1, len(eligible))

    # Build unordered rows a<b
    rows_un: List[dict] = []
    for (a, b), n_ab in pair_count.items():
        if (a not in eligible) or (b not in eligible):
            continue
        if n_ab < min_pair:
            continue
        n_a = count_per_class[a]
        n_b = count_per_class[b]

        # PMI (smoothed)
        pa  = (n_a + alpha) / (N + alpha * V)
        pb  = (n_b + alpha) / (N + alpha * V)
        pab = (n_ab + alpha) / (N + alpha * (V * V))
        pmi = math.log(pab / (pa * pb) + 1e-12)

        # NC both directions
        nc_ab = n_ab / max(1, n_a)  # P(b|a)
        nc_ba = n_ab / max(1, n_b)  # P(a|b)

        # 2x2 table for ZLOR
        n11 = n_ab
        n10 = n_a - n_ab
        n01 = n_b - n_ab
        n00 = N - n11 - n10 - n01
        zlor = log_odds_ratio_z(n11, n10, n01, n00)

        rows_un.append(
            dict(
                cat_a_id=a, cat_a_name=cats_id2name.get(a, str(a)),
                cat_b_id=b, cat_b_name=cats_id2name.get(b, str(b)),
                N_img=N,
                count_a=n_a, count_b=n_b, count_ab=n_ab,
                nc_ab=nc_ab, nc_ba=nc_ba,
                pmi=pmi, zlor=zlor,
            )
        )
    df_un = pd.DataFrame(rows_un).reset_index(drop=True)

    # Oriented rows (two per unordered row) with nc', TS later
    rows_or: List[dict] = []
    for _, r in df_un.iterrows():
        a = int(r["cat_a_id"]); b = int(r["cat_b_id"])
        n_a = int(r["count_a"]); n_b = int(r["count_b"]); n_ab = int(r["count_ab"])
        # a->b
        rows_or.append(dict(
            anchor_id=a, anchor=cats_id2name.get(a, str(a)),
            partner_id=b, partner=cats_id2name.get(b, str(b)),
            n_img=int(r["N_img"]),
            count_anchor=n_a, count_partner=n_b, count_both=n_ab,
            nc=float(r["nc_ab"]), pmi=float(r["pmi"]), zlor=float(r["zlor"])
        ))
        # b->a
        rows_or.append(dict(
            anchor_id=b, anchor=cats_id2name.get(b, str(b)),
            partner_id=a, partner=cats_id2name.get(a, str(a)),
            n_img=int(r["N_img"]),
            count_anchor=n_b, count_partner=n_a, count_both=n_ab,
            nc=float(r["nc_ba"]), pmi=float(r["pmi"]), zlor=float(r["zlor"])
        ))
    df_or = pd.DataFrame(rows_or)

    if df_or.empty:
        return df_or, df_un, dict(count_per_class)

    # Add nc' (smoothed conditional)
    nc_prime = (df_or["count_both"].astype(float) + float(alpha)) / (
        df_or["count_anchor"].astype(float) + 2.0 * float(alpha)
    )
    df_or["nc_prime"] = nc_prime.values

    # Add row-percentile of raw NC per anchor (0..1)
    df_or["nc_row_pct"] = 0.0
    for anchor, grp_idx in df_or.groupby("anchor").groups.items():
        idx = list(grp_idx)
        vals = df_or.loc[idx, "nc"].to_numpy(dtype=np.float64)
        pct = row_percentiles(vals)
        df_or.loc[idx, "nc_row_pct"] = pct

    # Support gate: sigmoid((n11 - T) / slope)
    gate = df_or["count_both"].astype(float).apply(
        lambda n11: sigmoid((n11 - float(support_T)) / max(1e-6, float(support_slope)))
    )

    # Evidence (positive part)
    if ts_evidence.lower() == "zlor":
        ev = df_or["zlor"].astype(float).clip(lower=0.0)
    elif ts_evidence.lower() == "pmi":
        ev = df_or["pmi"].astype(float).clip(lower=0.0)
    else:
        raise ValueError("--ts_evidence must be 'zlor' or 'pmi'")

    # Direction term
    if ts_use_row_percentile:
        direction = df_or["nc_row_pct"].astype(float)
    else:
        direction = df_or["nc_prime"].astype(float)

    df_or["ts_dir"] = (ev * gate * direction).astype(float)

    # Pretty ordering: by anchor then by ts_dir desc (fallback to nc)
    df_or = df_or.sort_values(["anchor", "ts_dir", "nc"], ascending=[True, False, False]).reset_index(drop=True)
    return df_or, df_un, dict(count_per_class)


# -------------------------
# Summaries & plotting
# -------------------------

def summarize_nc_thresholds(df_oriented: pd.DataFrame, q_hi: float, q_lo: float) -> Dict[str, dict]:
    """For each anchor, compute NC percentiles (hi, lo) and basic stats."""
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


def _plot_topk(df_or: pd.DataFrame, top_k: int, metric: str, out_dir: str) -> Optional[str]:
    if df_or.empty or top_k <= 0:
        return None
    metric = metric.lower()
    if metric not in {"pmi", "nc", "ts"}:
        metric = "ts"

    if metric == "pmi":
        series = df_or["pmi"]
        title = "Top-K by PMI"
        fname = f"topk_pmi_k{top_k}.png"
    elif metric == "nc":
        series = df_or["nc"]
        title = "Top-K by NC (P(partner|anchor))"
        fname = f"topk_nc_k{top_k}.png"
    else:
        series = df_or["ts_dir"]
        title = "Top-K by TS→"
        fname = f"topk_ts_k{top_k}.png"

    top = df_or.nlargest(top_k, series.name)
    labels = [f"{r.anchor} → {r.partner}\n(n11={int(r.count_both)})" for r in top.itertuples()]
    heights = top[series.name].astype(float).tolist()
    xs = list(range(len(labels)))

    plt.figure(figsize=(max(10, 0.6 * len(labels)), 6))
    plt.bar(xs, heights)
    plt.xticks(xs, labels, rotation=70, ha="right")
    plt.ylabel(series.name.upper())
    plt.title(title)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def _plot_topk_bottomk(df_or: pd.DataFrame, top_k: int, metric: str, out_dir: str) -> Optional[str]:
    """
    Plot Top-K and Bottom-K pairs for a chosen metric ("pmi", "nc", or "ts"),
    saving a single bar chart with two groups.
    """
    if df_or.empty or top_k <= 0:
        return None

    metric = (metric or "").lower()
    # Map user metric -> column name and labels
    if metric == "pmi":
        col = "pmi"
        ylab = "PMI"
        title = f"Top-{top_k} and Bottom-{top_k} by PMI"
        fname = f"topbottom_pmi_k{top_k}.png"
    elif metric == "nc":
        col = "nc"
        ylab = "NC  (P(partner | anchor))"
        title = f"Top-{top_k} and Bottom-{top_k} by NC"
        fname = f"topbottom_nc_k{top_k}.png"
    else:
        # default to directed TS
        col = "ts_dir"
        ylab = "TS→ (directed typicality score)"
        title = f"Top-{top_k} and Bottom-{top_k} by TS→"
        fname = f"topbottom_ts_k{top_k}.png"

    if col not in df_or.columns:
        return None

    # Work on a clean copy (drop NaNs to avoid pandas warnings)
    dd = df_or[[col, "anchor", "partner", "count_both"]].dropna(subset=[col]).copy()
    if dd.empty:
        return None

    top = dd.nlargest(top_k, col)
    bot = dd.nsmallest(top_k, col)

    # Build labels and heights: first Top-K, then Bottom-K
    def _labels(df):
        return [f"{r.anchor} → {r.partner}\n(n11={int(r.count_both)})" for r in df.itertuples()]

    labels_top = _labels(top)
    labels_bot = _labels(bot)
    heights_top = top[col].astype(float).tolist()
    heights_bot = bot[col].astype(float).tolist()

    n_top = len(labels_top)
    n_bot = len(labels_bot)
    xs_top = list(range(n_top))
    xs_bot = list(range(n_top, n_top + n_bot))

    plt.figure(figsize=(max(12, 0.6 * (n_top + n_bot)), 7))
    plt.bar(xs_top, heights_top, label=f"Top-{n_top}")
    plt.bar(xs_bot, heights_bot, label=f"Bottom-{n_bot}")
    plt.xticks(xs_top + xs_bot, labels_top + labels_bot, rotation=70, ha="right")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def build_nc_matrix_oriented(df_un: pd.DataFrame,
                             cats_id2name: Dict[int, str],
                             max_cats: Optional[int],
                             sample_seed: Optional[int],
                             pad_value: float) -> Tuple[List[str], np.ndarray]:
    """
    Oriented C×C NC matrix from the unordered df (one row per a<b).
    M[i,j] = NC(a->b) = P(b|a)
    M[j,i] = NC(b->a) = P(a|b)

    - Rows are anchors, columns are partners.
    - Diagonal is set to 0.0 for readability (P(a|a) is not defined here).
    """
    if df_un.empty:
        return [], np.zeros((0, 0), dtype=float)

    # Frequency proxy to pick categories (by max per-class count)
    freq: Dict[str, int] = {}
    for _, r in df_un.iterrows():
        a = cats_id2name.get(int(r["cat_a_id"]), str(int(r["cat_a_id"])))
        b = cats_id2name.get(int(r["cat_b_id"]), str(int(r["cat_b_id"])))
        freq[a] = max(freq.get(a, 0), int(r["count_a"]))
        freq[b] = max(freq.get(b, 0), int(r["count_b"]))

    pool = list(freq.keys())
    if max_cats is not None and max_cats > 0 and len(pool) > max_cats:
        rng = random.Random(sample_seed)
        pool = rng.sample(pool, k=max_cats)

    names = sorted(pool, key=lambda n: (-freq[n], n))
    name2idx = {n: i for i, n in enumerate(names)}
    C = len(names)
    M = np.full((C, C), pad_value, dtype=float)

    for _, r in df_un.iterrows():
        a_name = cats_id2name.get(int(r["cat_a_id"]), str(int(r["cat_a_id"])))
        b_name = cats_id2name.get(int(r["cat_b_id"]), str(int(r["cat_b_id"])))
        if a_name not in name2idx or b_name not in name2idx:
            continue
        i, j = name2idx[a_name], name2idx[b_name]
        nc_ab = float(r["nc_ab"])  # P(b|a)
        nc_ba = float(r["nc_ba"])  # P(a|b)

        # Keep the larger value if multiple rows hit the same cell
        if np.isnan(M[i, j]) or nc_ab > M[i, j]:
            M[i, j] = nc_ab
        if np.isnan(M[j, i]) or nc_ba > M[j, i]:
            M[j, i] = nc_ba

    np.fill_diagonal(M, 0.0)
    return names, M

def build_pmi_matrix(df_un: pd.DataFrame, cats_id2name: Dict[int, str],
                     max_cats: Optional[int], sample_seed: Optional[int],
                     pad_value: float) -> Tuple[List[str], np.ndarray]:
    """Symmetric C×C PMI matrix from unordered df (a<b)."""
    if df_un.empty:
        return [], np.zeros((0, 0), dtype=float)

    # Frequency proxy to pick categories (by max per-class count)
    freq: Dict[str, int] = {}
    for _, r in df_un.iterrows():
        a = cats_id2name.get(int(r["cat_a_id"]), str(int(r["cat_a_id"])))
        b = cats_id2name.get(int(r["cat_b_id"]), str(int(r["cat_b_id"])))
        freq[a] = max(freq.get(a, 0), int(r["count_a"]))
        freq[b] = max(freq.get(b, 0), int(r["count_b"]))

    pool = list(freq.keys())
    if max_cats is not None and max_cats > 0 and len(pool) > max_cats:
        rng = random.Random(sample_seed)
        pool = rng.sample(pool, k=max_cats)

    names = sorted(pool, key=lambda n: (-freq[n], n))
    name2idx = {n: i for i, n in enumerate(names)}
    C = len(names)
    M = np.full((C, C), pad_value, dtype=float)

    for _, r in df_un.iterrows():
        a_name = cats_id2name.get(int(r["cat_a_id"]), str(int(r["cat_a_id"])))
        b_name = cats_id2name.get(int(r["cat_b_id"]), str(int(r["cat_b_id"])))
        if a_name not in name2idx or b_name not in name2idx:
            continue
        i = name2idx[a_name]; j = name2idx[b_name]
        v = float(r["pmi"])
        if np.isnan(M[i, j]) or v > M[i, j]:
            M[i, j] = v
        if np.isnan(M[j, i]) or v > M[j, i]:
            M[j, i] = v
    np.fill_diagonal(M, 0.0)
    return names, M


def save_heatmap(names: List[str], M: np.ndarray, out_path: str, cmap_name="bwr") -> None:
    if M.size == 0:
        raise SystemExit("Heatmap: empty matrix.")
    finite_vals = M[np.isfinite(M)]
    vmax = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size else 1.0
    vlim = vmax if vmax > 0 else 1.0
    fig_w = max(6.0, min(18.0, 0.25 * len(names)))
    fig_h = fig_w
    plt.figure(figsize=(fig_w, fig_h), dpi=200)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("0.85")
    im = plt.imshow(M, vmin=-vlim, vmax=vlim, aspect="equal", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="PMI")
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.title("Category × Category PMI")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def save_heatmap_nc_oriented(names: List[str],
                             M: np.ndarray,
                             out_path: str,
                             cmap_name: str = "viridis") -> None:
    """
    Save an oriented NC heatmap.
    Rows = anchor (a), Cols = partner (b), Value = P(b|a) in [0,1].
    """
    if M.size == 0:
        raise SystemExit("NC heatmap: empty matrix.")
    finite_vals = M[np.isfinite(M)]
    vmax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    vmin = 0.0

    fig_w = max(6.0, min(18.0, 0.25 * len(names)))
    fig_h = fig_w
    plt.figure(figsize=(fig_w, fig_h), dpi=200)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("0.85")
    im = plt.imshow(M, vmin=vmin, vmax=vmax, aspect="equal", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="NC (P(partner|anchor))")
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.title("Oriented NC: rows=anchor, cols=partner (P(partner | anchor))")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()



# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Path to COCO/LVIS instances JSON.")
    ap.add_argument("--out_dir", required=True)

    # Filtering
    ap.add_argument("--min_area", type=float, default=1.0)
    ap.add_argument("--min_class", type=int, default=None, help="Absolute minimum images per class. If None and --auto_min, use quantile.")
    ap.add_argument("--min_pair", type=int, default=None, help="Absolute minimum images per pair. If None and --auto_min, use quantile.")
    ap.add_argument("--auto_min", action="store_true", help="Derive minima from dataset distribution (quantiles) and clamp with floors.")
    ap.add_argument("--q_class", type=float, default=0.10, help="Quantile for class minima when --auto_min.")
    ap.add_argument("--q_pair", type=float, default=0.10, help="Quantile for pair minima when --auto_min.")
    ap.add_argument("--floor_class", type=int, default=10, help="Hard lower bound for class minima when --auto_min.")
    ap.add_argument("--floor_pair", type=int, default=5,  help="Hard lower bound for pair minima when --auto_min.")

    # Smoothing / TS config
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha (PMI, nc').")
    ap.add_argument("--ts_evidence", choices=["zlor","pmi"], default="zlor", help="Evidence term for TS→.")
    ap.add_argument("--ts_use_row_percentile", action="store_true",
                    help="If set, direction term uses NC row-percentile instead of smoothed nc'.")
    ap.add_argument("--support_T", type=float, default=20.0, help="Support-gate midpoint T (in n11).")
    ap.add_argument("--support_slope", type=float, default=8.0, help="Support-gate slope (larger = gentler).")

    # Save compact unordered CSV too
    ap.add_argument("--save_unordered", action="store_true")

    # Optional plots
    ap.add_argument("--plot_topk", type=int, default=0)
    ap.add_argument("--plot_metric", choices=["pmi","nc","ts"], default="ts")
    ap.add_argument("--plot_heatmap", default=None, help="Heatmap PNG path (symmetric metric - PMI, or asymmetric - NC).")
    ap.add_argument("--heatmap_max_cats", type=int, default=None)
    ap.add_argument("--heatmap_pad", choices=["nan","zero"], default="nan")
    ap.add_argument("--heatmap_sample_seed", type=int, default=None)
    ap.add_argument("--heatmap_metric", choices=["pmi","nc"], default="pmi", help="Metric for heatmap: PMI (symmetric) or NC (oriented).")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    imgs, cats_id2name, anns = load_annotations(args.ann)
    img2cats = build_image_categories(anns, min_area=args.min_area)

    # First pass for auto minima
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
        min_pair  = auto_min_pair  if args.min_pair  is None else max(args.min_pair,  auto_min_pair)
    else:
        min_class = args.min_class if args.min_class is not None else 10
        min_pair  = args.min_pair  if args.min_pair  is not None else 5

    df_or, df_un, count_per_class_dict = compute_tables(
        img2cats, cats_id2name,
        min_class=min_class, min_pair=min_pair, alpha=args.alpha,
        ts_evidence=args.ts_evidence,
        ts_use_row_percentile=args.ts_use_row_percentile,
        support_T=args.support_T, support_slope=args.support_slope
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
                N_images=len(img2cats),
                min_class=min_class,
                min_pair=min_pair,
                alpha=args.alpha,
                ts_evidence=args.ts_evidence,
                ts_direction=("row_percentile" if args.ts_use_row_percentile else "nc_prime"),
                support_T=args.support_T,
                support_slope=args.support_slope,
                anchors=nc_summ,
            ),
            f,
            indent=2,
        )
    print(f"[OK] NC thresholds/stats -> {os.path.join(args.out_dir, 'nc_thresholds.json')}")

    # Optional plots
    if args.plot_topk > 0 and len(df_or) > 0:
        p = _plot_topk(df_or, args.plot_topk, args.plot_metric, args.out_dir)
        if p:
            print(f"[OK] Top-{args.plot_topk} {args.plot_metric.upper()} plot -> {p}")
        p = _plot_topk_bottomk(df_or, args.plot_topk, args.plot_metric, args.out_dir)
        if p:
            print(f"[OK] Top-and-Bottom-{args.plot_topk} {args.plot_metric.upper()} plot -> {p}")

    """	
    if args.plot_heatmap:
        pad_value = np.nan if args.heatmap_pad == "nan" else 0.0
        names, M = build_pmi_matrix(df_un, cats_id2name, args.heatmap_max_cats,
                                    args.heatmap_sample_seed, pad_value)
        if len(names) == 0:
            print("[WARN] Heatmap requested but no eligible categories; skipping.")
        else:
            save_heatmap(names, M, args.plot_heatmap, cmap_name="bwr")
            print(f"[OK] PMI heatmap -> {args.plot_heatmap} (C={len(names)})")
    """

    if args.plot_heatmap:
        pad_value = np.nan if args.heatmap_pad == "nan" else 0.0
        if args.heatmap_metric == "pmi":
            names, M = build_pmi_matrix(df_un, cats_id2name, args.heatmap_max_cats,
                                        args.heatmap_sample_seed, pad_value)
            if len(names) == 0:
                print("[WARN] Heatmap requested but no eligible categories; skipping.")
            else:
                save_heatmap(names, M, args.plot_heatmap, cmap_name="bwr")
                print(f"[OK] PMI heatmap -> {args.plot_heatmap} (C={len(names)})")
        else:
            names, M = build_nc_matrix_oriented(df_un, cats_id2name, args.heatmap_max_cats,
                                                args.heatmap_sample_seed, pad_value)
            if len(names) == 0:
                print("[WARN] Heatmap requested but no eligible categories; skipping.")
            else:
                save_heatmap_nc_oriented(names, M, args.plot_heatmap, cmap_name="bwr")
                print(f"[OK] NC heatmap (oriented) -> {args.plot_heatmap} (C={len(names)})")

if __name__ == "__main__":
    main()
 
