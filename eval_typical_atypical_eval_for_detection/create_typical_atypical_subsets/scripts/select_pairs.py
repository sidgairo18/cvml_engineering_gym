#!/usr/bin/env python3
import argparse, json, os
import pandas as pd
import numpy as np
from collections import defaultdict

def pick_top_bottom(df: pd.DataFrame, metric: str, n_typ: int, n_aty: int, max_per_anchor: int = 2):
    # Typical: high metric; Atypical: low metric
    df = df.copy()
    df["score"] = df[metric].astype(float)

    # cap per anchor to diversify
    per_anchor_typ = defaultdict(int)
    per_anchor_aty = defaultdict(int)

    typical = []
    atypical = []

    # Typical descending
    for _, r in df.sort_values("score", ascending=False).iterrows():
        a = r["anchor"]; b = r["partner"]
        if per_anchor_typ[a] >= max_per_anchor:
            continue
        typical.append(dict(a=a, b=b, type="typical", metric=float(r["score"]), source="cooccur"))
        per_anchor_typ[a] += 1
        if len(typical) >= n_typ:
            break

    # Atypical ascending
    for _, r in df.sort_values("score", ascending=True).iterrows():
        a = r["anchor"]; b = r["partner"]
        if per_anchor_aty[a] >= max_per_anchor:
            continue
        atypical.append(dict(a=a, b=b, type="atypical", metric=float(r["score"]), source="cooccur"))
        per_anchor_aty[a] += 1
        if len(atypical) >= n_aty:
            break

    return typical, atypical

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co_csv", required=True, help="cooccur_oriented.csv")
    ap.add_argument("--dataset", required=True, choices=["coco","lvis"])
    ap.add_argument("--metric", default="ts", choices=["ts","pmi","nc"])
    ap.add_argument("--top_typical", type=int, default=20)
    ap.add_argument("--top_atypical", type=int, default=20)
    ap.add_argument("--max_per_anchor", type=int, default=2)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.co_csv)
    # keep only anchors that have valid metric
    df = df[np.isfinite(df[args.metric].values)]
    typical, atypical = pick_top_bottom(df, args.metric, args.top_typical, args.top_atypical, args.max_per_anchor)

    pairs = typical + atypical
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(dict(dataset=args.dataset, pairs=pairs), f, indent=2)
    print(f"[OK] Saved pairs -> {args.out_json}  (typical={len(typical)}, atypical={len(atypical)})")

if __name__ == "__main__":
    main()

