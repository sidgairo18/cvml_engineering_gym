#!/usr/bin/env python3
import argparse, os, json
import pandas as pd
import numpy as np

def build_per_anchor(df: pd.DataFrame, metric: str, k_typ: int, k_aty: int):
    out_matched = []
    flat = []
    match_id = 1
    for anchor, grp in df.groupby("anchor"):
        g = grp.copy()
        g = g[np.isfinite(g[metric].values)]
        if g.empty:
            continue

        g_typ = g.sort_values(metric, ascending=False)
        g_aty = g.sort_values(metric, ascending=True)

        # typical picks
        typ_list = []
        used = set()
        for _, r in g_typ.iterrows():
            b = r["partner"]
            if b in used:
                continue
            typ_list.append(dict(a=anchor, b=b, type="typical", metric=float(r[metric])))
            used.add(b)
            if len(typ_list) >= k_typ:
                break

        # atypical picks disjoint from typical partners
        aty_list = []
        for _, r in g_aty.iterrows():
            b = r["partner"]
            if b in used:
                continue
            aty_list.append(dict(a=anchor, b=b, type="atypical", metric=float(r[metric])))
            used.add(b)
            if len(aty_list) >= k_aty:
                break

        if len(typ_list)==0 and len(aty_list)==0:
            continue

        out_matched.append(dict(match_id=match_id, anchor=anchor, typical=typ_list, atypical=aty_list))
        for p in typ_list:
            flat.append({**p, "match_id": match_id})
        for p in aty_list:
            flat.append({**p, "match_id": match_id})
        match_id += 1
    return out_matched, flat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co_csv", required=True, help="cooccur_oriented.csv")
    ap.add_argument("--dataset", required=True, choices=["coco","lvis"])
    ap.add_argument("--metric", default="ts", choices=["ts","pmi","nc"])
    ap.add_argument("--k_typ", type=int, default=1, help="Typical partners per anchor")
    ap.add_argument("--k_aty", type=int, default=1, help="Atypical partners per anchor")
    ap.add_argument("--out_matched_json", required=True)
    ap.add_argument("--out_pairs_flat_json", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.co_csv)
    matched, flat = build_per_anchor(df, args.metric, args.k_typ, args.k_aty)

    os.makedirs(os.path.dirname(args.out_matched_json), exist_ok=True)
    with open(args.out_matched_json,'w') as f:
        json.dump({'dataset': args.dataset, 'matched': matched}, f, indent=2)
    with open(args.out_pairs_flat_json,'w') as f:
        json.dump({'dataset': args.dataset, 'pairs': flat}, f, indent=2)
    print(f"[OK] Wrote {args.out_matched_json} (groups={len(matched)}) and {args.out_pairs_flat_json} (pairs={len(flat)})")

if __name__ == '__main__':
    main()

