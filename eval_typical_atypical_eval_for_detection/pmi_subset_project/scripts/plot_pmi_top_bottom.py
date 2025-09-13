#!/usr/bin/env python3
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def plot_top_bottom(pmi_csv, top_k, out_path):
    df = pd.read_csv(pmi_csv)
    if df.empty:
        raise SystemExit("Empty PMI CSV.")
    top_df = df.sort_values('pmi', ascending=False).head(top_k)
    bot_df = df.sort_values('pmi', ascending=True).head(top_k)

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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--pmi_csv', required=True)
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    plot_top_bottom(args.pmi_csv, args.top_k, args.out)
