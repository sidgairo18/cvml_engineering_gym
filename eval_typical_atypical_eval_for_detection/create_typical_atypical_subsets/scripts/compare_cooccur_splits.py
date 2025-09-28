#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _safe_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    rx = pd.Series(x).rank(method="average").values
    ry = pd.Series(y).rank(method="average").values
    xm, ym = rx - rx.mean(), ry - ry.mean()
    denom = (np.linalg.norm(xm) * np.linalg.norm(ym)) + 1e-12
    return float((xm @ ym) / denom)

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    xm, ym = x - x.mean(), y - y.mean()
    denom = (np.linalg.norm(xm) * np.linalg.norm(ym)) + 1e-12
    return float((xm @ ym) / denom)

def _scatter_plot(anchor, x, y, metric, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(5,5))
    plt.scatter(x, y, s=10)
    lim = max(np.max(np.abs(x)), np.max(np.abs(y)))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    plt.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1)
    plt.xlabel(f"Train {metric.upper()}")
    plt.ylabel(f"Eval {metric.upper()}")
    plt.title(anchor)
    plt.tight_layout()
    out = os.path.join(out_dir, f"{anchor}_{metric}.png")
    plt.savefig(out, dpi=160)
    plt.close()
    return out

def compare(df_tr: pd.DataFrame, df_ev: pd.DataFrame, metric: str, out_dir: str, scatter_k: int = 0):
    # keep only necessary columns
    cols = ["anchor","partner",metric]
    A = df_tr[cols].copy(); A = A.rename(columns={metric: "m_tr"})
    B = df_ev[cols].copy(); B = B.rename(columns={metric: "m_ev"})

    # anchors present in both
    anchors = sorted(set(A["anchor"]).intersection(set(B["anchor"])))
    rows = []
    scatters = []
    for a in anchors:
        aA = A[A["anchor"]==a].set_index("partner")
        aB = B[B["anchor"]==a].set_index("partner")
        common = sorted(set(aA.index).intersection(set(aB.index)))
        if len(common) == 0:
            continue
        vA = aA.loc[common, "m_tr"].astype(float).values
        vB = aB.loc[common, "m_ev"].astype(float).values
        spear = _safe_rank_corr(vA, vB)
        pear  = _safe_pearson(vA, vB)
        rows.append(dict(anchor=a, n_common=len(common), spearman=spear, pearson=pear))
        # optional scatter for top-K union (ranked by |train metric|)
        if scatter_k and len(common) > 1:
            # pick top-K by train metric magnitude
            order = np.argsort(-np.abs(vA))
            sel = order[:min(scatter_k, len(order))]
            _ = _scatter_plot(a, vA[sel], vB[sel], metric, os.path.join(out_dir, "scatter"))

    df_sum = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    df_sum.to_csv(os.path.join(out_dir, "split_compare_summary.csv"), index=False)

    # global macro averages
    macro = dict(
        anchors=len(df_sum),
        spearman_macro=float(np.nanmean(df_sum["spearman"])) if len(df_sum) else np.nan,
        pearson_macro=float(np.nanmean(df_sum["pearson"])) if len(df_sum) else np.nan,
    )
    pd.DataFrame([macro]).to_csv(os.path.join(out_dir, "split_compare_global.csv"), index=False)
    print(f"[OK] Wrote per-anchor and global summaries to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="cooccur_oriented.csv from train")
    ap.add_argument("--eval_csv", required=True, help="cooccur_oriented.csv from eval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--metric", default="ts", choices=["ts","pmi","nc"])
    ap.add_argument("--scatter_k", type=int, default=0, help="If >0, save per-anchor scatter of top-K partners by |train metric|")
    args = ap.parse_args()

    if args.metric == "ts":
        args.metric="ts_dir"

    os.makedirs(args.out_dir, exist_ok=True)
    df_tr = pd.read_csv(args.train_csv)
    df_ev = pd.read_csv(args.eval_csv)
    compare(df_tr, df_ev, args.metric, args.out_dir, scatter_k=args.scatter_k)

if __name__ == "__main__":
    main()

