#!/usr/bin/env python3
import argparse, json, math, os
from collections import defaultdict, Counter
from tqdm import tqdm
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--dataset', required=True, choices=['coco','lvis'])
    ap.add_argument('--min_images', type=int, default=100)
    ap.add_argument('--min_pair', type=int, default=20)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--plot_topk', type=int, default=0)
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

if __name__ == '__main__':
    main()
