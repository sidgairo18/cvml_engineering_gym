#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot top/bottom partners for a given anchor from cooccur_oriented.csv")
    ap.add_argument("--co_csv", required=True)
    ap.add_argument("--metric", choices=["nc","ts","pmi"], default="pmi")
    ap.add_argument("--anchor", required=True, help="Anchor class name, e.g., 'keyboard'")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", required=True, help="PNG file path")
    args = ap.parse_args()

    df = pd.read_csv(args.co_csv)
    df = df[(df["anchor_name"] == args.anchor)].copy()
    if df.empty:
        raise SystemExit(f"No rows for anchor '{args.anchor}' in {args.co_csv}")

    # Typical: largest metric; Atypical: smallest metric
    top_df = df.sort_values(args.metric, ascending=False).head(args.topk)
    bot_df = df.sort_values(args.metric, ascending=True).head(args.topk)

    # Build a single horizontal plot stacking typical (top) and atypical (bottom)
    names_top = list(top_df["ctx_name"])
    vals_top  = list(top_df[args.metric].astype(float))
    names_bot = list(bot_df["ctx_name"])
    vals_bot  = list(bot_df[args.metric].astype(float))

    n_top, n_bot = len(names_top), len(names_bot)
    H = max(5, 0.35*(n_top + n_bot) + 1.0)

    fig = plt.figure(figsize=(11, H), dpi=200)
    y_top = np.arange(n_top)
    y_bot = np.arange(n_bot)

    # Typical (top)
    ax1 = plt.subplot(2,1,1)
    ax1.barh(y_top, vals_top)
    ax1.set_yticks(y_top)
    ax1.set_yticklabels(names_top)
    ax1.invert_yaxis()
    ax1.set_title(f"{args.anchor} — TOP {n_top} (typical) by {args.metric.upper()}")
    ax1.set_xlabel(args.metric.upper())

    # Atypical (bottom)
    ax2 = plt.subplot(2,1,2)
    ax2.barh(y_bot, vals_bot)
    ax2.set_yticks(y_bot)
    ax2.set_yticklabels(names_bot)
    ax2.invert_yaxis()
    ax2.set_title(f"{args.anchor} — BOTTOM {n_bot} (atypical) by {args.metric.upper()}")
    ax2.set_xlabel(args.metric.UPPER() if hasattr(str, "UPPER") else args.metric.upper())

    plt.tight_layout()
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    plt.close()
    print(f"[OK] Saved {args.out}")

if __name__ == "__main__":
    main()

