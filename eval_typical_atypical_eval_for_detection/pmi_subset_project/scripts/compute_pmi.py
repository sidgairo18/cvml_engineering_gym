#!/usr/bin/env python3
import argparse, json, math, os, random
from collections import defaultdict, Counter
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_annotations(ann_path, dataset):
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    imgs = {im['id']: im for im in ann['images']}
    cats = {c['id']: c['name'] for c in ann['categories']}
    anns = ann['annotations']
    return imgs, cats, anns

def build_image_categories(anns, min_area=1.0, iscrowd_ok=False):
    img2cats = defaultdict(set)
    for a in anns:
        if not iscrowd_ok and a.get('iscrowd', 0) == 1:
            continue
        if a.get('area', 1.0) < min_area:
            continue
        img2cats[a['image_id']].add(a['category_id'])
    return img2cats

def compute_pmi(img2cats, min_images=50, min_pair=10, alpha=1.0):
    N_img = len(img2cats)
    cat_count = Counter()
    pair_count = Counter()
    for _, cats in img2cats.items():
        for c in cats:
            cat_count[c] += 1
        cats_list = sorted(list(cats))
        for i in range(len(cats_list)):
            for j in range(i+1, len(cats_list)):
                pair = (cats_list[i], cats_list[j])
                pair_count[pair] += 1

    eligible_cats = {c for c, n in cat_count.items() if n >= min_images}
    rows = []
    V = max(1, len(eligible_cats))
    for (a,b), cab in pair_count.items():
        if a not in eligible_cats or b not in eligible_cats:
            continue
        if cab < min_pair:
            continue
        pa  = (cat_count[a] + alpha) / (N_img + alpha*V)
        pb  = (cat_count[b] + alpha) / (N_img + alpha*V)
        pab = (cab + alpha) / (N_img + alpha*(V*V))
        pmi = math.log(pab / (pa*pb) + 1e-12)
        rows.append({'cat_a': a, 'cat_b': b, 'N_img': N_img,
                     'count_a': cat_count[a], 'count_b': cat_count[b],
                     'count_ab': cab, 'pmi': pmi})
    df = pd.DataFrame(rows).sort_values('pmi', ascending=False)
    return df, cat_count, pair_count

def _plot_top_bottom(df_out, top_k, out_dir):
    top_df = df_out.sort_values('pmi', ascending=False).head(top_k)
    bot_df = df_out.sort_values('pmi', ascending=True).head(top_k)
    labels, heights = [], []
    n_top = len(top_df)
    for _, row in top_df.iterrows():
        labels.append(f"{row['cat_a_name']} | {row['cat_b_name']}\nA:{int(row['count_a'])} B:{int(row['count_b'])} AB:{int(row['count_ab'])}")
        heights.append(float(row['pmi']))
    for _, row in bot_df.iterrows():
        labels.append(f"{row['cat_a_name']} | {row['cat_b_name']}\nA:{int(row['count_a'])} B:{int(row['count_b'])} AB:{int(row['count_ab'])}")
        heights.append(float(row['pmi']))
    xs = list(range(len(labels)))
    plt.figure(figsize=(max(12, 0.6*len(labels)), 7))
    plt.bar(xs[:n_top], heights[:n_top], label='Top-K (highest PMI)')
    plt.bar(xs[n_top:], heights[n_top:], label='Bottom-K (lowest PMI)')
    plt.xticks(xs, labels, rotation=75, ha='right')
    plt.ylabel('PMI (Pointwise Mutual Information)')
    plt.title(f'Top {top_k} and Bottom {top_k} PMI pairs (with counts A, B, AB)')
    plt.tight_layout()
    plt.legend()
    plot_path = os.path.join(out_dir, f'pmi_top_bottom_plot_k{top_k}.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path

def build_pmi_matrix(
    df_out: pd.DataFrame,
    cats_id2name: Dict[int, str],
    max_cats: Optional[int] = None,
    sample_seed: Optional[int] = None,   # NEW
    pad_value: Optional[float] = np.nan,
) -> Tuple[List[str], np.ndarray]:
    """
    Build a symmetric category×category PMI matrix from df_out (rows are pairs with a<b).
    Returns names (matrix order) and PMI matrix M (C×C).
    """
    if df_out.empty:
        return [], np.zeros((0, 0), dtype=float)

    # Choose category set by frequency proxy from df_out
    '''
    freq: Dict[str, int] = {}
    for _, r in df_out.iterrows():
        a = r["cat_a_name"]; b = r["cat_b_name"]
        freq[a] = max(freq.get(a, 0), int(r["count_a"]))
        freq[b] = max(freq.get(b, 0), int(r["count_b"]))

    names_all = sorted(freq.keys(), key=lambda k: (-freq[k], k))
    names = names_all[:max_cats] if (max_cats is not None and max_cats > 0) else names_all
    '''
    # Build the candidate pool from categories present in df_out and their frequencies
    freq: Dict[str, int] = {}
    for _, r in df_out.iterrows():
        a = r["cat_a_name"]; b = r["cat_b_name"]
        freq[a] = max(freq.get(a, 0), int(r["count_a"]))
        freq[b] = max(freq.get(b, 0), int(r["count_b"]))

    names_pool = list(freq.keys())

    # If max_cats is set, randomly pick that many first (without replacement)
    if max_cats is not None and max_cats > 0 and len(names_pool) > max_cats:
        rng = random.Random(sample_seed)  # deterministic if seed provided
        names_pool = rng.sample(names_pool, k=max_cats)

    # Now sort the sampled names by frequency (desc) then name for stability
    names = sorted(names_pool, key=lambda n: (-freq[n], n))

    idx = {n: i for i, n in enumerate(names)}
    C = len(names)
    fill = pad_value if pad_value is not None else np.nan
    M = np.full((C, C), fill, dtype=float)

    # Fill symmetric PMI values
    for _, r in df_out.iterrows():
        a = r["cat_a_name"]; b = r["cat_b_name"]
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        pmi = float(r["pmi"])
        # keep the larger if multiple entries would map to same cell
        if np.isnan(M[i, j]) or pmi > M[i, j]:
            M[i, j] = pmi
        if np.isnan(M[j, i]) or pmi > M[j, i]:
            M[j, i] = pmi

    # Diagonal as 0.0 for readability
    np.fill_diagonal(M, 0.0)
    return names, M


def save_pmi_heatmap(names: List[str], M: np.ndarray, out_path: str) -> None:
    """
    Save the PMI matrix as a heatmap image with symmetric color scaling around 0.
    """
    import matplotlib.pyplot as plt
    import os

    if M.size == 0:
        raise SystemExit("PMI heatmap: empty matrix (no eligible categories).")

    finite_vals = M[np.isfinite(M)]
    if finite_vals.size == 0:
        vlim = 1.0
    else:
        vmax = float(np.nanmax(np.abs(finite_vals)))
        vlim = vmax if vmax > 0 else 1.0

    fig_w = max(6.0, min(18.0, 0.25 * len(names)))
    fig_h = fig_w
    plt.figure(figsize=(fig_w, fig_h), dpi=200)
    
    cmap = plt.get_cmap("bwr").copy()
    cmap.set_bad("0.85")  # light gray for NaNs
    #im = plt.imshow(M, vmin=-vlim, vmax=vlim, aspect="equal")
    im = plt.imshow(M, vmin=-vlim, vmax=vlim, aspect="equal", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="PMI")

    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.title("Category × Category PMI")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--dataset', required=True, choices=['coco','lvis'])
    ap.add_argument('--min_images', type=int, default=100)
    ap.add_argument('--min_pair', type=int, default=20)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--plot_topk', type=int, default=0)
    ap.add_argument('--plot_heatmap', default=None, help="If set, save a PMI heatmap PNG to this path.")
    ap.add_argument('--heatmap_max_cats', type=int, default=None, help="Optional cap on number of categories shown (most frequent).")
    ap.add_argument('--heatmap_pad', default='nan', choices=['nan','zero'], help="Fill for missing pairs: 'nan' or 'zero'.")
    ap.add_argument('--heatmap_sample_seed', type=int, default=None,
                help="Random seed for selecting categories into the heatmap (before sorting).")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs, cats, anns = load_annotations(args.ann, args.dataset)
    img2cats = build_image_categories(anns)
    df, cat_count, pair_count = compute_pmi(img2cats, args.min_images, args.min_pair)

    df_out = df.copy()
    df_out['cat_a_name'] = df_out['cat_a'].map(cats)
    df_out['cat_b_name'] = df_out['cat_b'].map(cats)
    pmi_csv = os.path.join(args.out_dir, 'pmi.csv')
    df_out.to_csv(pmi_csv, index=False)

    with open(os.path.join(args.out_dir, 'category_map.json'), 'w') as f:
        json.dump({str(k): v for k,v in cats.items()}, f, indent=2)

    top_typ = df_out.head(300)
    low_aty = df_out.sort_values('pmi', ascending=True).head(300)
    top_typ.to_csv(os.path.join(args.out_dir,'preview_top_typical.csv'), index=False)
    low_aty.to_csv(os.path.join(args.out_dir,'preview_low_atypical.csv'), index=False)

    if args.plot_topk and len(df_out)>0:
        plot_path = _plot_top_bottom(df_out, args.plot_topk, args.out_dir)
        print(f"[OK] Plot saved to {plot_path}")
    print(f"[OK] PMI saved to {pmi_csv}. Rows={len(df_out)}")

    # Optional: full PMI heatmap across categories
    if args.plot_heatmap:
        pad_value = np.nan if args.heatmap_pad == 'nan' else 0.0
        names, M = build_pmi_matrix(df_out, cats, max_cats=args.heatmap_max_cats, 
                                    sample_seed=args.heatmap_sample_seed, pad_value=pad_value)
        if len(names) == 0:
            print("[WARN] Heatmap requested but no categories eligible; skipping.")
        else:
            save_pmi_heatmap(names, M, args.plot_heatmap)
            print(f"[OK] Heatmap saved to {args.plot_heatmap} (C={len(names)})")


if __name__ == '__main__':
    main()
