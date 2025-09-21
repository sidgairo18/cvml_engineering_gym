#!/usr/bin/env python3
# Python 3.8+ compatible

import argparse, json, math, os, random
from collections import defaultdict, Counter
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# IO
# -----------------------------
def load_annotations(ann_path: str, dataset: str):
    with open(ann_path, "r") as f:
        ann = json.load(f)
    imgs = {im["id"]: im for im in ann["images"]}
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    anns = ann["annotations"]
    return imgs, cats, anns


def build_image_categories(anns, min_area: float = 1.0, iscrowd_ok: bool = False):
    """Map: image_id -> set(category_id) after basic filtering."""
    img2cats = defaultdict(set)
    for a in anns:
        if not iscrowd_ok and a.get("iscrowd", 0) == 1:
            continue
        if a.get("area", 1.0) < min_area:
            continue
        img2cats[a["image_id"]].add(a["category_id"])
    return img2cats


# -----------------------------
# Stats: counts, PMI, NC, LOR, z, TS
# -----------------------------
def _lor_and_se(n11, n10, n01, n00):
    # Haldane–Anscombe smoothing (0.5) to avoid zeros
    a = n11 + 0.5
    b = n10 + 0.5
    c = n01 + 0.5
    d = n00 + 0.5
    lor = math.log((a * d) / (b * c))
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    return lor, se


def _support_weight(n11, T=10.0, s=5.0):
    # Logistic ramp from ~0 to 1 around T
    return 1.0 / (1.0 + math.exp(-(n11 - T) / max(1e-6, s)))


def _nc_smooth(n11, n_a, alpha=1.0):
    # Laplace/Beta smoothing for conditional probability
    return (n11 + alpha) / (max(0.0, n_a) + 2.0 * alpha)


def compute_cooccur(
    img2cats: Dict[int, set],
    min_images: int = 50,
    min_pair: int = 10,
    alpha: float = 1.0,      # for PMI smoothing
    ts_T: float = 10.0,      # TS support midpoint
    ts_s: float = 5.0,       # TS support slope
    nc_alpha: float = 1.0,   # NC smoothing
):
    """
    Returns:
      df: rows for unordered pairs (a<b) with:
          N_img, count_a, count_b, count_ab,
          PMI,
          NC_a_to_b, NC_b_to_a, NCs (smoothed),
          LOR, SE, z_LOR,
          TS_a_to_b, TS_b_to_a (see notes),
          plus names mapped later.
      cat_count, pair_count (raw Counters)
    """
    N_img = len(img2cats)

    # Per-image category presence
    cat_count = Counter()
    pair_count = Counter()        # symmetric (a<b): co-occur-in-image count
    # Also track marginals needed for LOR table
    # n11: co-occur; n10: a present b absent; n01: a absent b present; n00: neither
    # We can compute n11 from pair_count and derive n10/n01 from cat_count later.

    for _, cats in img2cats.items():
        cats_list = sorted(list(cats))
        # update marginals
        for c in cats_list:
            cat_count[c] += 1
        # update pairs
        for i in range(len(cats_list)):
            for j in range(i + 1, len(cats_list)):
                pair = (cats_list[i], cats_list[j])
                pair_count[pair] += 1

    # Eligible categories by frequency
    eligible_cats = {c for c, n in cat_count.items() if n >= min_images}

    rows = []
    V = max(1, len(eligible_cats))
    for (a, b), n11 in pair_count.items():
        if a not in eligible_cats or b not in eligible_cats:
            continue
        if n11 < min_pair:
            continue

        n_a = cat_count[a]
        n_b = cat_count[b]
        # PMI with add-alpha smoothing (image-level)
        pa = (n_a + alpha) / (N_img + alpha * V)
        pb = (n_b + alpha) / (N_img + alpha * V)
        pab = (n11 + alpha) / (N_img + alpha * (V * V))
        pmi = math.log(pab / (pa * pb) + 1e-12)

        # Directed NC (unsmoothed and smoothed)
        nc_a_b = n11 / float(max(1, n_a))
        nc_b_a = n11 / float(max(1, n_b))
        nc_a_b_s = _nc_smooth(n11, n_a, nc_alpha)
        nc_b_a_s = _nc_smooth(n11, n_b, nc_alpha)

        # 2x2 table over images for LOR / z:
        # n11 known; derive n10 = n_a - n11; n01 = n_b - n11; n00 = N_img - (n11+n10+n01)
        n10 = max(0, n_a - n11)
        n01 = max(0, n_b - n11)
        n00 = max(0, N_img - (n11 + n10 + n01))

        lor, se = _lor_and_se(n11, n10, n01, n00)
        z = lor / max(1e-9, se)

        # TS in *both* directions: positive association only
        z_pos = max(0.0, z)
        w = _support_weight(n11, T=ts_T, s=ts_s)

        ts_a_b = z_pos * w * nc_a_b_s
        ts_b_a = z_pos * w * nc_b_a_s

        rows.append(
            dict(
                cat_a=a,
                cat_b=b,
                N_img=N_img,
                count_a=n_a,
                count_b=n_b,
                count_ab=n11,
                pmi=pmi,
                nc_a_to_b=nc_a_b,
                nc_b_to_a=nc_b_a,
                nc_a_to_b_smooth=nc_a_b_s,
                nc_b_to_a_smooth=nc_b_a_s,
                lor=lor,
                se_lor=se,
                z_lor=z,
                support_w=w,
                ts_a_to_b=ts_a_b,
                ts_b_to_a=ts_b_a,
            )
        )

    df = pd.DataFrame(rows).sort_values("pmi", ascending=False)
    return df, cat_count, pair_count


# -----------------------------
# Plotting
# -----------------------------
def _plot_top_bottom(df: pd.DataFrame, top_k: int, out_dir: str, metric: str = "pmi"):
    """
    metric: one of {'pmi','nc_mean','ts_max'} for symmetric ranking
      - pmi          : symmetric PMI
      - nc_mean      : 0.5*(nc_a_to_b + nc_b_to_a)
      - ts_max       : max(ts_a_to_b, ts_b_to_a)
    """
    if df.empty:
        return None

    def sym_value(row):
        if metric == "pmi":
            return float(row["pmi"])
        elif metric == "nc_mean":
            return 0.5 * (float(row["nc_a_to_b"]) + float(row["nc_b_to_a"]))
        elif metric == "ts_max":
            return max(float(row["ts_a_to_b"]), float(row["ts_b_to_a"]))
        else:
            return float(row["pmi"])

    df2 = df.copy()
    df2["sym_metric"] = df2.apply(sym_value, axis=1)

    top_df = df2.sort_values("sym_metric", ascending=False).head(top_k)
    bot_df = df2.sort_values("sym_metric", ascending=True).head(top_k)

    labels, heights = [], []
    n_top = len(top_df)
    for _, row in top_df.iterrows():
        labels.append(
            f"{row['cat_a_name']} | {row['cat_b_name']}\n"
            f"A:{int(row['count_a'])} B:{int(row['count_b'])} AB:{int(row['count_ab'])}"
        )
        heights.append(float(row["sym_metric"]))
    for _, row in bot_df.iterrows():
        labels.append(
            f"{row['cat_a_name']} | {row['cat_b_name']}\n"
            f"A:{int(row['count_a'])} B:{int(row['count_b'])} AB:{int(row['count_ab'])}"
        )
        heights.append(float(row["sym_metric"]))

    xs = list(range(len(labels)))
    plt.figure(figsize=(max(12, 0.6 * len(labels)), 7))
    plt.bar(xs[:n_top], heights[:n_top], label="Top-K (highest)")
    plt.bar(xs[n_top:], heights[n_top:], label="Bottom-K (lowest)")
    plt.xticks(xs, labels, rotation=75, ha="right")
    plt.ylabel(metric.upper())
    plt.title(f"Top {top_k} and Bottom {top_k} by {metric.upper()}")
    plt.tight_layout()
    plt.legend()
    plot_path = os.path.join(out_dir, f"cooccur_top_bottom_{metric}_k{top_k}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def _select_names_for_heatmap(
    df_out: pd.DataFrame,
    max_cats: Optional[int],
    sample_seed: Optional[int],
):
    # Frequency proxy from df_out (max of counts where a class appears)
    freq: Dict[str, int] = {}
    for _, r in df_out.iterrows():
        a = r["cat_a_name"]
        b = r["cat_b_name"]
        freq[a] = max(freq.get(a, 0), int(r["count_a"]))
        freq[b] = max(freq.get(b, 0), int(r["count_b"]))

    names_pool = list(freq.keys())
    if max_cats is not None and max_cats > 0 and len(names_pool) > max_cats:
        rng = random.Random(sample_seed)
        names_pool = rng.sample(names_pool, k=max_cats)

    names = sorted(names_pool, key=lambda n: (-freq[n], n))
    return names


def build_metric_matrix(
    df_out: pd.DataFrame,
    names: List[str],
    which: str = "pmi",           # 'pmi' | 'nc_mean' | 'ts_max'
    pad_value: float = np.nan,
) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    C = len(names)
    M = np.full((C, C), pad_value, dtype=float)

    for _, r in df_out.iterrows():
        a = r["cat_a_name"]
        b = r["cat_b_name"]
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]

        if which == "pmi":
            val = float(r["pmi"])
        elif which == "nc_mean":
            val = 0.5 * (float(r["nc_a_to_b"]) + float(r["nc_b_to_a"]))
        elif which == "ts_max":
            val = max(float(r["ts_a_to_b"]), float(r["ts_b_to_a"]))
        else:
            val = float(r["pmi"])

        # symmetric fill (keep larger magnitude for visibility)
        if np.isnan(M[i, j]) or abs(val) > abs(M[i, j]):
            M[i, j] = val
        if np.isnan(M[j, i]) or abs(val) > abs(M[j, i]):
            M[j, i] = val

    # Diagonal for readability
    np.fill_diagonal(M, 0.0)
    return M


def save_heatmap(names: List[str], M: np.ndarray, out_path: str, diverging: bool = True, title: str = ""):
    if M.size == 0:
        raise SystemExit("Heatmap: empty matrix (no eligible categories).")

    finite_vals = M[np.isfinite(M)]
    if finite_vals.size == 0:
        vlim = 1.0
    else:
        vmax = float(np.nanmax(np.abs(finite_vals))) if diverging else float(np.nanmax(finite_vals))
        vlim = vmax if vmax > 0 else 1.0

    fig_w = max(6.0, min(18.0, 0.25 * len(names)))
    fig_h = fig_w
    plt.figure(figsize=(fig_w, fig_h), dpi=200)

    cmap = plt.get_cmap("bwr").copy() if diverging else plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")  # light gray for NaNs

    if diverging:
        im = plt.imshow(M, vmin=-vlim, vmax=vlim, aspect="equal", cmap=cmap)
    else:
        im = plt.imshow(M, vmin=0.0, vmax=vlim, aspect="equal", cmap=cmap)

    plt.colorbar(im, fraction=0.046, pad=0.04, label=title or "value")
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.title(title or "Category × Category")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute co-occurrence stats: PMI, NC, LOR/z, TS.")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--dataset", required=True, choices=["coco", "lvis"])
    ap.add_argument("--min_images", type=int, default=100)
    ap.add_argument("--min_pair", type=int, default=20)
    ap.add_argument("--out_dir", required=True)

    # Smoothing / TS knobs
    ap.add_argument("--pmi_alpha", type=float, default=1.0)
    ap.add_argument("--nc_alpha", type=float, default=1.0)
    ap.add_argument("--ts_T", type=float, default=10.0)
    ap.add_argument("--ts_s", type=float, default=5.0)

    # Plots
    ap.add_argument("--plot_topk", type=int, default=0)
    ap.add_argument("--topk_metric", default="pmi", choices=["pmi", "nc_mean", "ts_max"])
    ap.add_argument("--plot_heatmap", default=None, help="If set, save a heatmap PNG to this path.")
    ap.add_argument("--heatmap_max_cats", type=int, default=None)
    ap.add_argument("--heatmap_metric", default="pmi", choices=["pmi", "nc_mean", "ts_max"])
    ap.add_argument("--heatmap_pad", default="nan", choices=["nan", "zero"])
    ap.add_argument("--heatmap_sample_seed", type=int, default=None)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    imgs, cats, anns = load_annotations(args.ann, args.dataset)
    img2cats = build_image_categories(anns)

    df, cat_count, pair_count = compute_cooccur(
        img2cats,
        min_images=args.min_images,
        min_pair=args.min_pair,
        alpha=args.pmi_alpha,
        ts_T=args.ts_T,
        ts_s=args.ts_s,
        nc_alpha=args.nc_alpha,
    )

    # Decorate with names
    df_out = df.copy()
    df_out["cat_a_name"] = df_out["cat_a"].map(cats)
    df_out["cat_b_name"] = df_out["cat_b"].map(cats)

    # Save CSV
    out_csv = os.path.join(args.out_dir, "cooccur.csv")
    df_out.to_csv(out_csv, index=False)

    with open(os.path.join(args.out_dir, "category_map.json"), "w") as f:
        json.dump({str(k): v for k, v in cats.items()}, f, indent=2)

    # Quick previews
    top_typ = df_out.sort_values("pmi", ascending=False).head(300)
    low_pmi = df_out.sort_values("pmi", ascending=True).head(300)
    top_typ.to_csv(os.path.join(args.out_dir, "preview_top_pmi.csv"), index=False)
    low_pmi.to_csv(os.path.join(args.out_dir, "preview_low_pmi.csv"), index=False)

    print(f"[OK] Co-occurrence table -> {out_csv} (rows={len(df_out)})")

    # Top/bottom plot
    if args.plot_topk and len(df_out) > 0:
        plot_path = _plot_top_bottom(df_out, args.plot_topk, args.out_dir, metric=args.topk_metric)
        if plot_path:
            print(f"[OK] Top/Bottom plot -> {plot_path}")

    # Heatmap
    if args.plot_heatmap:
        pad_value = np.nan if args.heatmap_pad == "nan" else 0.0
        names = _select_names_for_heatmap(df_out, args.heatmap_max_cats, args.heatmap_sample_seed)
        if len(names) == 0:
            print("[WARN] Heatmap requested but no categories eligible; skipping.")
        else:
            # Diverging color only for PMI/TS (centered around 0 for PMI; TS is non-negative but diverging still ok)
            diverging = args.heatmap_metric in ("pmi", "ts_max")
            title = {
                "pmi": "PMI",
                "nc_mean": "NC (mean of directions)",
                "ts_max": "TS (max of directions)"
            }[args.heatmap_metric]
            M = build_metric_matrix(df_out, names, which=args.heatmap_metric, pad_value=pad_value)
            save_heatmap(names, M, args.plot_heatmap, diverging=diverging, title=title)
            print(f"[OK] Heatmap saved to {args.plot_heatmap} (C={len(names)})")


if __name__ == "__main__":
    main()

