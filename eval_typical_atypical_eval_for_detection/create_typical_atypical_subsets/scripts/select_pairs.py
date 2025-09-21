#!/usr/bin/env python3
# Python 3.8-compatible
import argparse, json, os
import pandas as pd

# -------- helpers --------
def _load_llm_pairs(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def _resolve_metric_cols(metric: str):
    """
    Returns (value_fn, ascending_for_atypical, is_directed)
    value_fn(row) -> float to rank by.
    """
    metric = metric.lower()
    if metric == "pmi":
        return (lambda r: float(r["pmi"])), True, False
    if metric == "nc_mean":
        return (lambda r: 0.5 * (float(r["nc_a_to_b"]) + float(r["nc_b_to_a"]))), True, False
    if metric == "nc_a_to_b":
        return (lambda r: float(r["nc_a_to_b"])), True, True
    if metric == "nc_b_to_a":
        return (lambda r: float(r["nc_b_to_a"])), True, True
    if metric == "ts_max":
        return (lambda r: max(float(r["ts_a_to_b"]), float(r["ts_b_to_a"]))), True, False
    if metric == "ts_a_to_b":
        return (lambda r: float(r["ts_a_to_b"])), True, True
    if metric == "ts_b_to_a":
        return (lambda r: float(r["ts_b_to_a"])), True, True
    if metric in ("z_lor", "lor_z", "zl"):
        return (lambda r: float(r["z_lor"])), True, False
    # default
    return (lambda r: float(r["pmi"])), True, False

def _ensure_cols(df: pd.DataFrame, need: list):
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in cooccur.csv: {missing}")

# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="Select typical/atypical pairs from cooccur.csv")
    ap.add_argument("--cooccur_csv", required=True, help="Output CSV from compute_cooccur.py")
    ap.add_argument("--dataset", required=True, choices=["coco","lvis"])
    ap.add_argument("--top_typical", type=int, default=20)
    ap.add_argument("--top_atypical", type=int, default=20)
    ap.add_argument("--metric", default="ts_a_to_b",
                    choices=["pmi","nc_mean","nc_a_to_b","nc_b_to_a","ts_max","ts_a_to_b","ts_b_to_a","z_lor"],
                    help="Ranking metric; typical = highest, atypical = lowest")
    ap.add_argument("--min_count_ab", type=int, default=0, help="Filter pairs with count_ab < this")
    ap.add_argument("--min_count_a", type=int, default=0)
    ap.add_argument("--min_count_b", type=int, default=0)
    ap.add_argument("--llm_pairs_json", default=None, help="Optional JSON to merge/override")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.cooccur_csv)
    _ensure_cols(df, [
        "cat_a_name","cat_b_name","count_ab","count_a","count_b",
        "pmi","nc_a_to_b","nc_b_to_a","ts_a_to_b","ts_b_to_a","z_lor"
    ])

    # Basic filtering
    df = df[(df["count_ab"] >= args.min_count_ab) &
            (df["count_a"]  >= args.min_count_a) &
            (df["count_b"]  >= args.min_count_b)].copy()

    if df.empty:
        raise SystemExit("No pairs left after filtering.")

    value_fn, asc_for_atyp, is_directed = _resolve_metric_cols(args.metric)

    # Typical: sort by metric DESC
    df["rank_val"] = df.apply(value_fn, axis=1)
    typical_df = df.sort_values("rank_val", ascending=False).head(args.top_typical)
    typical = [
        dict(a=row["cat_a_name"], b=row["cat_b_name"], type="typical",
             source=f"metric:{args.metric}", metric=float(row["rank_val"]))
        for _, row in typical_df.iterrows()
    ]

    # Atypical: sort by metric ASC
    atyp_df = df.sort_values("rank_val", ascending=asc_for_atyp).head(args.top_atypical)
    atypical = [
        dict(a=row["cat_a_name"], b=row["cat_b_name"], type="atypical",
             source=f"metric:{args.metric}", metric=float(row["rank_val"]))
        for _, row in atyp_df.iterrows()
    ]

    # Merge with optional LLM suggestions (override / add)
    llm_pairs = _load_llm_pairs(args.llm_pairs_json)
    all_pairs = {(p["a"], p["b"], p["type"]): p for p in (typical + atypical)}
    for p in llm_pairs:
        key = (p["a"], p["b"], p.get("type","typical"))
        all_pairs[key] = {**p, **dict(source="llm")}

    final = list(all_pairs.values())

    # Default scene/relation heuristics (you can change later)
    for p in final:
        p.setdefault("scene", "street" if p["type"]=="typical" else "indoor room")
        p.setdefault("relation", "next to" if p["type"]=="typical" else "on")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(dict(dataset=args.dataset, pairs=final), f, indent=2)

    print(f"[OK] Selected pairs -> {args.out_json}  (N={len(final)}; metric={args.metric})")

if __name__ == "__main__":
    main()

