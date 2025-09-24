#!/usr/bin/env python3
"""
Build matched pair-sets from ORIENTED co-occurrence so each anchor gets
both Typical and Atypical partners (balanced).

Inputs:
  --co_csv : cooccur_oriented.csv

Outputs:
  --out_matched_json : {"dataset": <name>, "matched": [
                          {"match_id": i, "anchor": A,
                           "typical":   [{a=A,b=Bi,type="typical",score_nc,...}, ...],
                           "atypical":  [{a=A,b=Cj,type="atypical",score_nc,...}, ...]}
                       ]}
  --out_pairs_flat_json : {"dataset": <name>, "pairs": [ ...flat list... ]}

Strategy:
- Choose K anchors with the largest support (count_anchor).
- For each anchor:
  * pick up to P_typ partners with highest NC,
  * pick up to P_aty partners with lowest NC (but existing; count_both >= min_pair_aty),
    also avoid overlap with the chosen typical partners.
- If an anchor cannot supply both sides, skip it (keeps sets balanced).
"""
import argparse, json, os
from collections import defaultdict
import pandas as pd


def choose_k_anchors(df, k):
    # pick anchors with largest count_anchor (proxy for stability)
    agg = df.groupby("anchor")["count_anchor"].max().sort_values(ascending=False)
    return list(agg.head(k).index)


def rank_partners(df, anchor, min_pair_aty, tiebreak_pmi=True):
    g = df[df["anchor"] == anchor].copy()
    if g.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Typical (high NC first)
    typical = g.sort_values(["nc", "pmi"] if tiebreak_pmi else ["nc"], ascending=[False, False]).reset_index(drop=True)
    # Atypical (low NC), require that pair exists at least min_pair_aty times
    aty = g[g["count_both"] >= min_pair_aty].sort_values(["nc", "pmi"] if tiebreak_pmi else ["nc"], ascending=[True, True]).reset_index(drop=True)
    return typical, aty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co_csv", required=True, help="cooccur_oriented.csv")
    ap.add_argument("--dataset", required=True, choices=["coco", "lvis"])
    ap.add_argument("--k", type=int, default=20, help="Number of anchors")
    ap.add_argument("--pairs_per_anchor_typ", type=int, default=1)
    ap.add_argument("--pairs_per_anchor_aty", type=int, default=1)
    ap.add_argument("--min_pair_aty", type=int, default=1)
    ap.add_argument("--tiebreak_pmi", action="store_true")
    ap.add_argument("--out_matched_json", required=True)
    ap.add_argument("--out_pairs_flat_json", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.co_csv)

    anchors = choose_k_anchors(df, args.k)
    matched = []
    pairs_flat = []
    match_id = 1

    for a in anchors:
        typical, atypical = rank_partners(df, a, args.min_pair_aty, tiebreak_pmi=args.tiebreak_pmi)
        if typical.empty or atypical.empty:
            continue

        # Pick top P_typ typical
        picked_typ = []
        for _, r in typical.iterrows():
            picked_typ.append(
                dict(
                    a=str(a),
                    b=str(r["partner"]),
                    type="typical",
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
            if len(picked_typ) >= args.pairs_per_anchor_typ:
                break

        # Pick top P_aty atypical, disjoint from typical partners
        typ_partners = {p["b"] for p in picked_typ}
        picked_aty = []
        for _, r in atypical.iterrows():
            partner = str(r["partner"])
            if partner in typ_partners:
                continue
            picked_aty.append(
                dict(
                    a=str(a),
                    b=partner,
                    type="atypical",
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
            if len(picked_aty) >= args.pairs_per_anchor_aty:
                break

        if not picked_typ or not picked_aty:
            continue

        matched.append(
            dict(
                match_id=match_id,
                anchor=str(a),
                typical=picked_typ,
                atypical=picked_aty,
            )
        )
        # flat view (add simple defaults for downstream synthesis)
        for p in picked_typ:
            pairs_flat.append({**p, "match_id": match_id, "relation": "next to", "scene": "street"})
        for p in picked_aty:
            pairs_flat.append({**p, "match_id": match_id, "relation": "on", "scene": "indoor room"})
        match_id += 1

    os.makedirs(os.path.dirname(args.out_matched_json), exist_ok=True)
    with open(args.out_matched_json, "w") as f:
        json.dump({"dataset": args.dataset, "matched": matched}, f, indent=2)
    with open(args.out_pairs_flat_json, "w") as f:
        json.dump({"dataset": args.dataset, "pairs": pairs_flat}, f, indent=2)

    print(f"[OK] Wrote {args.out_matched_json} (sets={len(matched)}) and {args.out_pairs_flat_json} (pairs={len(pairs_flat)})")


if __name__ == "__main__":
    main()

