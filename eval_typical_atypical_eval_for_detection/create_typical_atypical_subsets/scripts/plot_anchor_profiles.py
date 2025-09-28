#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_anchor(df: pd.DataFrame, anchor: str, metric: str, k: int, out_dir: str):
    sub = df[df["anchor"]==anchor].copy()
    if sub.empty:
        return
    sub = sub.sort_values(metric, ascending=False).head(k)
    labels = [f"{p} (AB:{int(n)})" for p, n in zip(sub["partner"], sub["count_both"])]
    vals = sub[metric].astype(float).values
    xs = list(range(len(vals)))
    plt.figure(figsize=(max(8, 0.5*len(vals)), 5))
    plt.bar(xs, vals)
    plt.xticks(xs, labels, rotation=70, ha="right")
    plt.ylabel(metric.upper())
    plt.title(f"{anchor} — Top-{k} by {metric.upper()}")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{anchor}_top{k}_{metric}.png")
    plt.savefig(out, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co_csv", required=True, help="cooccur_oriented.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--metric", default="ts", choices=["ts","pmi","nc"])
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--anchors", nargs="*", default=None, help="If provided, limit to these anchors")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.co_csv)
    anchors = args.anchors if args.anchors else sorted(df["anchor"].unique().tolist())
    for a in anchors:
        plot_anchor(df, a, args.metric, args.top_k, args.out_dir)
    print(f"[OK] Plots written to {args.out_dir}")

if __name__ == "__main__":
    main()

