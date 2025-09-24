#!/usr/bin/env python3
"""
Select top Typical and Atypical pairs from ORIENTED co-occurrence.

Input:
  --co_csv   : cooccur_oriented.csv from compute_cooccur.py

Outputs:
  --out_json : {"dataset": <name>, "pairs": [ {a,b,type,source,score_nc,score_pmi,...} ] }

Behavior:
- Typical per-anchor: highest NC (top_k_typ).
- Atypical per-anchor: lowest NC among pairs that DO appear (count_both >= min_pair_aty).
- If an anchor has fewer than requested partners, we take as many as available (graceful degradation).
- Optional PMI tie-breaker.

You can later feed 'pairs' into your subset builders (original/real-paste/diffusion).
"""
import argparse, json, os
import pandas as pd


def pick_topk_per_anchor(df, k, ascending=False, min_pair=1, tiebreak_pmi=True):
    """
    Return list of dicts per anchor with up to k partners, sorted by NC (and PMI if requested).
    ascending=False -> Typical (high NC), ascending=True -> Atypical (low NC).
    """
    out = []
    order = ["nc"]
    if tiebreak_pmi:
        order.append("pmi")
    for anchor, grp in df.groupby("anchor"):
        g = grp[grp["count_both"] >= min_pair].copy()
        if g.empty:
            continue
        g = g.sort_values(order, ascending=[ascending] + [ascending]* (len(order)-1))
        sel = g.head(k)
        for _, r in sel.iterrows():
            out.append(
                dict(
                    a=str(anchor),
                    b=str(r["partner"]),
                    type="typical" if not ascending else "atypical",
                    source="cooccur_nc",
                    score_nc=float(r["nc"]),
                    score_pmi=float(r["pmi"]),
                    counts=dict(
                        n_img=int(r["n_img"]),
                        count_anchor=int(r["count_anchor"]),
                        count_partner=int(r["count_partner"]),
                        count_both=int(r["count_both"]),
                    ),
                )
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co_csv", required=True, help="cooccur_oriented.csv")
    ap.add_argument("--dataset", required=True, choices=["coco", "lvis"])
    ap.add_argument("--top_k_typ", type=int, default=1, help="Typical partners per anchor")
    ap.add_argument("--top_k_aty", type=int, default=1, help="Atypical partners per anchor")
    ap.add_argument("--min_pair_aty", type=int, default=1, help="Require at least this many images with both for atypical")
    ap.add_argument("--tiebreak_pmi", action="store_true")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.co_csv)

    typical = pick_topk_per_anchor(df, k=args.top_k_typ, ascending=False, min_pair=1, tiebreak_pmi=args.tiebreak_pmi)
    atypical = pick_topk_per_anchor(df, k=args.top_k_aty, ascending=True, min_pair=args.min_pair_aty, tiebreak_pmi=args.tiebreak_pmi)

    # merge and dedup identical triples (a,b,type)
    seen = set()
    pairs = []
    for p in (typical + atypical):
        key = (p["a"], p["b"], p["type"])
        if key in seen:
            continue
        seen.add(key)
        # Add default scene/relation if you want (used by your synthesis pipelines)
        p.setdefault("scene", "street" if p["type"] == "typical" else "indoor room")
        p.setdefault("relation", "next to" if p["type"] == "typical" else "on")
        pairs.append(p)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"dataset": args.dataset, "pairs": pairs}, f, indent=2)
    print(f"[OK] Wrote {args.out_json}  (pairs={len(pairs)})")


if __name__ == "__main__":
    main()

