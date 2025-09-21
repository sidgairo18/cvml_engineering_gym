#!/usr/bin/env python3
# Python 3.8-compatible
import argparse, os, json
import pandas as pd
from collections import defaultdict

# ---------- helpers ----------
def _load_llm_pairs(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def _plausibility_lookup(llm_pairs):
    table = {}
    for p in llm_pairs:
        key = (p["a"], p["b"], p.get("type","typical"))
        table[key] = {
            "plausibility": float(p.get("plausibility", 1.0)),
            "scene": p.get("scene"),
            "relation": p.get("relation"),
        }
    return table

def _ensure_cols(df: pd.DataFrame, need: list):
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns in cooccur.csv: {miss}")

def _value_fn_from_metric(metric: str, direction: str):
    """
    metric in: pmi|nc|ts|z_lor
    direction in: auto|a_to_b|b_to_a
    Returns value_fn(row), ascending_for_atypical
    """
    metric = metric.lower()
    direction = direction.lower()

    if metric == "pmi":
        return (lambda r: float(r["pmi"])), True  # atypical=low
    if metric == "z_lor":
        return (lambda r: float(r["z_lor"])), True
    if metric == "nc":
        if direction == "b_to_a":
            return (lambda r: float(r["nc_b_to_a"])), True
        # default a_to_b
        return (lambda r: float(r["nc_a_to_b"])), True
    if metric == "ts":
        if direction == "b_to_a":
            return (lambda r: float(r["ts_b_to_a"])), True
        return (lambda r: float(r["ts_a_to_b"])), True

    # fallback
    return (lambda r: float(r["pmi"])), True

def _choose_k_anchors(df: pd.DataFrame, k: int):
    # frequency proxy → anchors with many images
    counts = defaultdict(int)
    for _, row in df.iterrows():
        counts[row["cat_a_name"]] = max(counts[row["cat_a_name"]], int(row["count_a"]))
        counts[row["cat_b_name"]] = max(counts[row["cat_b_name"]], int(row["count_b"]))
    anchors = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)[:k]
    return anchors

def _rank_partners_directed(df: pd.DataFrame, anchor: str, value_fn):
    """
    Build two dataframes containing rows where anchor is 'a' (a->b) and 'b' (b->a),
    but both converted to (anchor -> partner) shape for easy ranking.
    """
    rows = []
    for _, r in df.iterrows():
        a = r["cat_a_name"]; b = r["cat_b_name"]
        if a == anchor:
            rows.append(dict(anchor=anchor, partner=b, pair=(a,b), val=float(value_fn(r)),
                             count_a=int(r["count_a"]), count_b=int(r["count_b"]),
                             count_ab=int(r["count_ab"])))
        elif b == anchor:
            # flip: reuse the same val_fn (it should already pick correct direction for ts/nc if asked as b_to_a)
            rows.append(dict(anchor=anchor, partner=a, pair=(a,b), val=float(value_fn(r)),
                             count_a=int(r["count_a"]), count_b=int(r["count_b"]),
                             count_ab=int(r["count_ab"])))
    if not rows:
        return pd.DataFrame([])
    sub = pd.DataFrame(rows)
    # we return sorted both ways later in the caller
    return sub

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Select matched typical/atypical pairs per anchor using cooccur.csv")
    ap.add_argument("--cooccur_csv", required=True)
    ap.add_argument("--dataset", required=True, choices=["coco","lvis"])
    ap.add_argument("--k", type=int, default=20, help="Number of anchors")
    ap.add_argument("--pairs_per_anchor_typ", type=int, default=1)
    ap.add_argument("--pairs_per_anchor_aty", type=int, default=1)

    # metric & direction
    ap.add_argument("--metric", default="ts", choices=["ts","nc","pmi","z_lor"],
                    help="Ranking metric for typical high / atypical low")
    ap.add_argument("--direction", default="a_to_b",
                    choices=["a_to_b","b_to_a","auto"],
                    help="Direction for directed metrics (ts/nc). "
                         "'auto' treats pmi/z_lor as symmetric and ts/nc as a_to_b.")

    # filters
    ap.add_argument("--min_count_ab", type=int, default=0)
    ap.add_argument("--min_count_a", type=int, default=0)
    ap.add_argument("--min_count_b", type=int, default=0)

    # optional plausibility from LLM list
    ap.add_argument("--llm_pairs_json", default=None)
    ap.add_argument("--min_plausibility", type=float, default=0.0)

    # outputs
    ap.add_argument("--out_matched_json", required=True)
    ap.add_argument("--out_pairs_flat_json", required=True)

    args = ap.parse_args()

    df = pd.read_csv(args.cooccur_csv)
    _ensure_cols(df, [
        "cat_a_name","cat_b_name","count_ab","count_a","count_b",
        "pmi","nc_a_to_b","nc_b_to_a","ts_a_to_b","ts_b_to_a","z_lor"
    ])

    # Filters
    df = df[(df["count_ab"] >= args.min_count_ab) &
            (df["count_a"]  >= args.min_count_a) &
            (df["count_b"]  >= args.min_count_b)].copy()
    if df.empty:
        raise SystemExit("No pairs left after filtering.")

    # Direction resolution
    direction = args.direction
    if direction == "auto":
        direction = "a_to_b" if args.metric in ("ts","nc") else "auto"

    # Build value function
    if args.metric in ("ts","nc"):
        value_fn_typ, asc_for_atyp = _value_fn_from_metric(args.metric, direction)
        value_fn_aty, _ = value_fn_typ, asc_for_atyp
        direct = True
    else:
        # symmetric
        value_fn_typ, asc_for_atyp = _value_fn_from_metric(args.metric, "auto")
        value_fn_aty = value_fn_typ
        direct = False

    # Anchor selection by frequency
    anchors = _choose_k_anchors(df, args.k)

    # LLM plausibility table
    llm_pairs = _load_llm_pairs(args.llm_pairs_json)
    llm_table = _plausibility_lookup(llm_pairs)

    def is_plaus(a,b,typetag):
        key = (a,b,typetag)
        key2 = (b,a,typetag)
        if key in llm_table:
            return llm_table[key]["plausibility"] >= args.min_plausibility
        if key2 in llm_table:
            return llm_table[key2]["plausibility"] >= args.min_plausibility
        return True

    matched = []
    flat_pairs = []
    match_id = 1

    for anchor in anchors:
        sub = _rank_partners_directed(df, anchor, value_fn_typ)
        if sub.empty:
            continue

        # typical = high val
        typical_sorted = sub.sort_values("val", ascending=False)
        # atypical = low val
        atypical_sorted = sub.sort_values("val", ascending=asc_for_atyp)

        picked_typ = []
        for _, row in typical_sorted.iterrows():
            a,b = row["pair"]
            partner = row["partner"]
            if not is_plaus(anchor, partner, "typical"): 
                continue
            picked_typ.append(dict(a=anchor, b=partner, metric=float(row["val"]), type="typical"))
            if len(picked_typ) >= args.pairs_per_anchor_typ:
                break

        picked_aty = []
        typ_partners = {p["b"] for p in picked_typ}
        for _, row in atypical_sorted.iterrows():
            a,b = row["pair"]
            partner = row["partner"]
            if partner in typ_partners:
                continue
            if not is_plaus(anchor, partner, "atypical"):
                continue
            picked_aty.append(dict(a=anchor, b=partner, metric=float(row["val"]), type="atypical"))
            if len(picked_aty) >= args.pairs_per_anchor_aty:
                break

        if not picked_typ or not picked_aty:
            continue

        matched.append({
            "match_id": match_id,
            "anchor": anchor,
            "typical": picked_typ,
            "atypical": picked_aty,
            "metric": args.metric,
            "direction": direction
        })

        # flat list (add default scene/relation)
        for p in picked_typ:
            flat_pairs.append({**p, "match_id": match_id, "relation": "next to", "scene": "street"})
        for p in picked_aty:
            flat_pairs.append({**p, "match_id": match_id, "relation": "on", "scene": "indoor room"})

        match_id += 1

    os.makedirs(os.path.dirname(args.out_matched_json), exist_ok=True)
    with open(args.out_matched_json, "w") as f:
        json.dump({"dataset": args.dataset, "matched": matched}, f, indent=2)
    with open(args.out_pairs_flat_json, "w") as f:
        json.dump({"dataset": args.dataset, "pairs": flat_pairs}, f, indent=2)

    print(f"[OK] Wrote {args.out_matched_json} and {args.out_pairs_flat_json} "
          f"(anchors={len(matched)}; metric={args.metric}; direction={direction})")

if __name__ == "__main__":
    main()

