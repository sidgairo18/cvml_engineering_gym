#!/usr/bin/env python3
import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr

EPS = 1e-12

def row_to_prob(vals: np.ndarray, metric: str) -> np.ndarray:
    """Convert a row of metric values into a probability distribution."""
    v = vals.astype(float)
    if metric in ("nc","ts"):
        v = np.clip(v, 0.0, None)
        s = v.sum()
        if s <= 0:
            return np.ones_like(v) / len(v)
        return v / s
    # PMI: min-max to [0,1] per row, then normalize
    vmin, vmax = v.min(), v.max()
    if vmax - vmin <= 1e-9:
        return np.ones_like(v) / len(v)
    v = (v - vmin) / (vmax - vmin)
    s = v.sum()
    if s <= 0:
        return np.ones_like(v) / len(v)
    return v / s

def dict_from_df(df: pd.DataFrame, metric: str):
    # anchor -> {ctx_name: metric_value}
    out = {}
    for _, r in df.iterrows():
        a = r["anchor_name"]; b = r["ctx_name"]
        val = float(r[metric])
        out.setdefault(a, {})[b] = val
    return out

def topk_overlap(ctx2v_A: dict, ctx2v_B: dict, k: int):
    if k <= 0:
        return np.nan, np.nan, np.nan
    topA = sorted(ctx2v_A.items(), key=lambda kv: kv[1], reverse=True)[:k]
    topB = sorted(ctx2v_B.items(), key=lambda kv: kv[1], reverse=True)[:k]
    setA = {c for c,_ in topA}
    setB = {c for c,_ in topB}
    inter = len(setA & setB)
    union = len(setA | setB)
    jacc = inter / union if union>0 else np.nan
    prec = inter / max(1,len(setA))
    rec  = inter / max(1,len(setB))
    return jacc, prec, rec

def align_rows(ctx2v_A: dict, ctx2v_B: dict):
    ctxs = sorted(set(ctx2v_A.keys()) | set(ctx2v_B.keys()))
    A = np.array([ctx2v_A.get(c, 0.0) for c in ctxs], dtype=float)
    B = np.array([ctx2v_B.get(c, 0.0) for c in ctxs], dtype=float)
    return ctxs, A, B

def plot_anchor_diff(anchor, ctxs, Avals, Bvals, metric, out_path, topk=15):
    # Show top-K by max(A,B)
    order = np.argsort(-(np.maximum(Avals, Bvals)))
    sel = order[:topk]
    ctxs_s = [ctxs[i] for i in sel]
    A_s = Avals[sel]
    B_s = Bvals[sel]

    h = max(4, 0.3 * len(ctxs_s))
    plt.figure(figsize=(10, h))
    y = np.arange(len(ctxs_s))
    plt.barh(y+0.2, A_s, height=0.4, label="train")
    plt.barh(y-0.2, B_s, height=0.4, label="eval")
    plt.yticks(y, ctxs_s)
    plt.gca().invert_yaxis()
    plt.xlabel(metric.upper())
    plt.title(f"{anchor} — train vs eval ({metric.upper()})")
    plt.legend()
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Compare co-occurrence profiles between train and eval splits.")
    ap.add_argument("--co_csv_train", required=True, help="cooccur_oriented.csv from train split")
    ap.add_argument("--co_csv_eval",  required=True, help="cooccur_oriented.csv from eval/val split")
    ap.add_argument("--metric", choices=["nc","ts","pmi"], default="nc")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk_overlap", type=int, default=5)
    ap.add_argument("--plot_anchors", type=int, default=10, help="Save plots for N anchors with largest JS divergence")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df_tr = pd.read_csv(args.co_csv_train)
    df_ev = pd.read_csv(args.co_csv_eval)

    # Keep minimal columns and drop NaNs
    cols = ["anchor_name","ctx_name", args.metric, "n_anchor", "n_ctx", "n_both"]
    df_tr = df_tr[cols].dropna()
    df_ev = df_ev[cols].dropna()

    A = dict_from_df(df_tr, args.metric)
    B = dict_from_df(df_ev, args.metric)

    anchors = sorted(set(A.keys()) & set(B.keys()))
    rows = []
    js_scores = []

    for a in anchors:
        ctxA = A[a]; ctxB = B[a]
        # quick counts
        n_ctx_tr = len(ctxA); n_ctx_ev = len(ctxB)

        # correlation on aligned raw metric vectors (zeros if missing)
        ctxs, Avec, Bvec = align_rows(ctxA, ctxB)
        r = np.nan
        try:
            if np.std(Avec) > 0 and np.std(Bvec) > 0:
                r = pearsonr(Avec, Bvec)[0]
        except Exception:
            r = np.nan

        # JS divergence on normalized row distributions
        p = row_to_prob(Avec, args.metric)
        q = row_to_prob(Bvec, args.metric)
        js = float(jensenshannon(p, q, base=np.e)**2)  # JS divergence (not distance)

        # top-k overlap
        jacc, prec, rec = topk_overlap(ctxA, ctxB, args.topk_overlap)

        rows.append(dict(
            anchor=a,
            n_ctx_train=n_ctx_tr,
            n_ctx_eval=n_ctx_ev,
            corr_pearson=r,
            js_div=js,
            topk=args.topk_overlap,
            topk_jaccard=jacc,
            topk_precision=prec,
            topk_recall=rec
        ))
        js_scores.append((a, js, ctxs, Avec, Bvec))

    out_csv = os.path.join(args.out_dir, f"train_vs_eval_{args.metric}_summary.csv")
    pd.DataFrame(rows).sort_values(["js_div","corr_pearson"], ascending=[False, True]).to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} (anchors: {len(rows)})")

    # Plots for anchors with largest JS divergence
    js_scores.sort(key=lambda x: -x[1])
    for i, (a, js, ctxs, Avec, Bvec) in enumerate(js_scores[:max(0,args.plot_anchors)]):
        plot_anchor_diff(
            a, ctxs, Avec, Bvec, args.metric,
            out_path=os.path.join(args.out_dir, f"anchor_{i+1:03d}_{a}_train-vs-eval_{args.metric}.png"),
            topk=15
        )
    if args.plot_anchors > 0:
        print(f"[OK] Plotted {min(args.plot_anchors, len(js_scores))} anchors with largest drift.")

if __name__ == "__main__":
    main()

