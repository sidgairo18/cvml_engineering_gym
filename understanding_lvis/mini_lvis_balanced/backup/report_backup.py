# mini_lvis_balanced/report.py
import os
import json
import math
import argparse
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from lvis import LVIS


# -----------------------------
# Bucketing / helpers
# -----------------------------
def _size_bucket_from_bbox(bbox, areaRng):
    w, h = bbox[2], bbox[3]
    if w < 1 or h < 1:
        return None
    a = w * h
    if a < areaRng[0]: return "S"
    if a < areaRng[1]: return "M"
    return "L"

def _cat_meta(lvis):
    cats = lvis.load_cats(lvis.get_cat_ids())
    by_id = {c["id"]: c for c in cats}
    tiers = {"c": set(), "f": set(), "r": set()}
    for c in cats:
        tiers[c["frequency"]].add(c["id"])
    return by_id, tiers

def _tier_of(cat_meta, cid):
    return cat_meta[cid]["frequency"]


# -----------------------------
# Full-train stats & targets
# -----------------------------
def compute_tier_stats(lvis):
    cat_meta, tiers = _cat_meta(lvis)
    # classes per tier
    classes = {t: len(tiers[t]) for t in "cfr"}
    # annotations per tier
    ann_counts = {"c":0, "f":0, "r":0}
    for ann in lvis.anns.values():
        ann_counts[_tier_of(cat_meta, ann["category_id"])] += 1
    # image sets per tier (≥1 instance)
    img_sets = {"c": set(), "f": set(), "r": set()}
    for t in "cfr":
        for cid in tiers[t]:
            img_sets[t].update(lvis.cat_img_map[cid])
    images = {t: len(img_sets[t]) for t in "cfr"}
    totals = {
        "classes": sum(classes.values()),
        "annotations": sum(ann_counts.values()),
        "images": len(lvis.imgs),
    }
    return classes, ann_counts, images, totals

def compute_target_tier_mix(classes, ann_counts, alpha=0.6):
    cls_sum = sum(classes.values())
    ann_sum = sum(ann_counts.values())
    cls_share = np.array([classes[t]/cls_sum for t in "cfr"])
    ann_share = np.array([ann_counts[t]/ann_sum for t in "cfr"])
    mix = alpha * ann_share + (1 - alpha) * cls_share
    mix = mix / mix.sum()
    return {"c": mix[0], "f": mix[1], "r": mix[2]}


# -----------------------------
# Build class×size tables
# -----------------------------
def build_class_size_table(lvis, areaRng, tier=None):
    """Return dict[(cid, size)] -> count. If tier is None, include all tiers."""
    cat_meta, _ = _cat_meta(lvis)
    tab = Counter()
    for ann in lvis.anns.values():
        cid = ann["category_id"]
        if (tier is not None) and (_tier_of(cat_meta, cid) != tier):
            continue
        sb = _size_bucket_from_bbox(ann["bbox"], areaRng)
        if sb is None:
            continue
        tab[(cid, sb)] += 1
    return tab

def build_subset_class_size_table(lvis_train, subset_json_path, areaRng, tier=None):
    """Same schema but computed from a LVIS-style subset json."""
    with open(subset_json_path, "r") as f:
        subset = json.load(f)

    # Build quick cat->tier from train meta (subset lacks 'frequency')
    cat_meta, _ = _cat_meta(lvis_train)

    tab = Counter()
    for ann in subset["annotations"]:
        cid = ann["category_id"]
        if (tier is not None) and (_tier_of(cat_meta, cid) != tier):
            continue
        sb = _size_bucket_from_bbox(ann["bbox"], areaRng)
        if sb is None:
            continue
        tab[(cid, sb)] += 1
    return tab

def counts_to_aligned_vectors(full_tab, mini_tab):
    """Align (cid,size) keys, return (full_vec, mini_vec, keys)."""
    keys = sorted(set(full_tab.keys()) | set(mini_tab.keys()))
    f = np.array([full_tab.get(k, 0) for k in keys], dtype=np.float64)
    m = np.array([mini_tab.get(k, 0) for k in keys], dtype=np.float64)
    return f, m, keys


# -----------------------------
# Coverage / per-tier aggregates
# -----------------------------
def subset_coverage_by_tier(lvis_train, subset_json_path):
    with open(subset_json_path, "r") as f:
        subset = json.load(f)
    cat_meta, tiers = _cat_meta(lvis_train)

    # Which categories appear in subset?
    present = set(a["category_id"] for a in subset["annotations"])

    coverage = {}
    for t in "cfr":
        total_t = len(tiers[t])
        have_t = sum(1 for cid in tiers[t] if cid in present)
        coverage[t] = (have_t, total_t, have_t / max(1, total_t))
    return coverage

def subset_tier_image_annotation_counts(lvis_train, subset_json_path):
    with open(subset_json_path, "r") as f:
        subset = json.load(f)
    cat_meta, _ = _cat_meta(lvis_train)

    # Images per tier: an image counts toward a tier if it has ≥1 ann of that tier.
    img_tiers = {"c": set(), "f": set(), "r": set()}
    for ann in subset["annotations"]:
        t = _tier_of(cat_meta, ann["category_id"])
        img_tiers[t].add(ann["image_id"])

    # Annots per tier:
    ann_tiers = {"c":0, "f":0, "r":0}
    for ann in subset["annotations"]:
        t = _tier_of(cat_meta, ann["category_id"])
        ann_tiers[t] += 1

    return (
        {t: len(img_tiers[t]) for t in "cfr"},
        ann_tiers,
        len(subset["images"]),
        len(subset["annotations"]),
    )


# -----------------------------
# Correlations & plotting
# -----------------------------
def safe_pearson(x, y):
    if np.all(x == 0) and np.all(y == 0):
        return 1.0
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0,1])

def plot_report(
    out_png,
    tier_metrics,
    tier_scatter_data,
    target_mix,
    actual_mix
):
    """
    tier_metrics: dict tier -> dict of numbers printed
    tier_scatter_data: dict tier -> (full_vec, mini_vec)
    target_mix, actual_mix: dict c/f/r -> share
    """
    plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 3, height_ratios=[1.2, 1.0])

    # --- Panel A: Tier mix bar chart ---
    axA = plt.subplot(gs[0, 0])
    tiers = ["c","f","r"]; names=["Common","Frequent","Rare"]
    tgt = [target_mix[t] for t in tiers]
    act = [actual_mix[t]  for t in tiers]
    x = np.arange(3)
    axA.bar(x-0.2, tgt, width=0.4, label="Target")
    axA.bar(x+0.2, act, width=0.4, label="Mini (actual)")
    axA.set_xticks(x); axA.set_xticklabels(names)
    axA.set_ylim(0, 1.0)
    axA.set_title("Tier Mix (Image Share)")
    axA.legend()

    # --- Panel B/C/D: Scatter per tier (log-log) ---
    for col, t in enumerate(tiers):
        ax = plt.subplot(gs[0, col if col>0 else 1])  # B at [0,1], C at [0,2]; D at [1,0]? Let's place neatly:
    # To keep layout simple, do 1st row: mix + two scatters; 2nd row: one scatter + legend text
    plt.close()  # The simple GridSpec above got a bit messy; fallback to a simpler layout below.

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Top-left: tier mix
    axA = axes[0,0]
    axA.bar(x-0.2, tgt, width=0.4, label="Target")
    axA.bar(x+0.2, act, width=0.4, label="Mini (actual)")
    axA.set_xticks(x); axA.set_xticklabels(names)
    axA.set_ylim(0, 1.0)
    axA.set_title("Tier Mix (Image Share)")
    axA.legend()

    # Scatters: we’ll show Frequent & Common & Rare in the remaining three panels
    order = ["c","f","r"]
    for ax, t in zip(axes.flatten()[1:], order):
        f, m = tier_scatter_data[t]
        # remove zeros for log plotting, but keep correlation computed elsewhere
        f_plot = np.where(f <= 0, np.nan, f)
        m_plot = np.where(m <= 0, np.nan, m)
        ax.scatter(f_plot, m_plot, s=8, alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Full-train counts")
        ax.set_ylabel("Mini counts")
        ax.set_title(f"{t.upper()}  (pearson: {tier_metrics[t]['pearson']:.3f})")
        # Reference y=x
        lim = [np.nanmin(f_plot), np.nanmax(f_plot)]
        lim = [max(1, lim[0]), max(1, lim[1])]
        mn, mx = lim[0], lim[1]
        ax.plot([mn, mx], [mn, mx], ls="--", lw=1, color="gray")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# Main entry
# -----------------------------
def main():
    ap = argparse.ArgumentParser("mini-LVIS: reporting & diagnostics")
    ap.add_argument("--train_json", required=True, help="LVIS v1 train json")
    ap.add_argument("--subset_json", required=True, help="Your mini-LVIS subset json (output of sampler)")
    ap.add_argument("--alpha", type=float, default=0.6, help="Blend for target tier mix (ann-share vs class-share)")
    ap.add_argument("--area_small", type=float, default=32.0)
    ap.add_argument("--area_medium", type=float, default=96.0)
    ap.add_argument("--save_prefix", type=str, default="lvis_mini_report")
    ap.add_argument("--export_csv", action="store_true", help="Also export per-tier class×size tables to CSV")
    args = ap.parse_args()

    areaRng = [args.area_small**2, args.area_medium**2, 1e5**2]

    # Load
    lvis_train = LVIS(args.train_json)

    # Targets from full train
    classes, ann_counts, _, totals = compute_tier_stats(lvis_train)
    target_mix = compute_target_tier_mix(classes, ann_counts, alpha=args.alpha)

    # Actual subset tier shares (by images)
    img_tier_counts, ann_tier_counts, total_img_subset, total_ann_subset = subset_tier_image_annotation_counts(
        lvis_train, args.subset_json
    )
    actual_mix = {t: img_tier_counts[t] / max(1, total_img_subset) for t in "cfr"}

    # Per-tier class×size (full vs mini)
    tier_metrics = {}
    tier_scatter_data = {}
    for t in "cfr":
        full_tab = build_class_size_table(lvis_train, areaRng, tier=t)
        mini_tab = build_subset_class_size_table(lvis_train, args.subset_json, areaRng, tier=t)
        f, m, keys = counts_to_aligned_vectors(full_tab, mini_tab)

        # normalize to proportions to compare shapes
        f_norm = f / max(1, f.sum())
        m_norm = m / max(1, m.sum())

        pear = safe_pearson(f_norm, m_norm)
        mse = float(np.mean((f_norm - m_norm) ** 2))
        k_nonzero_full = int(np.sum(f > 0))
        k_nonzero_mini = int(np.sum(m > 0))

        tier_metrics[t] = {
            "pearson": pear,
            "mse_norm": mse,
            "k_nonzero_full": k_nonzero_full,
            "k_nonzero_mini": k_nonzero_mini,
            "sum_full": int(f.sum()),
            "sum_mini": int(m.sum()),
        }
        tier_scatter_data[t] = (f, m)

        if args.export_csv:
            import csv
            csv_path = f"{args.save_prefix}_tier_{t}_class_size.csv"
            with open(csv_path, "w", newline="") as cf:
                w = csv.writer(cf)
                w.writerow(["category_id", "size_bucket", "full_count", "mini_count"])
                for (cid, sb), fv in full_tab.items():
                    w.writerow([cid, sb, fv, mini_tab.get((cid, sb), 0)])
            print(f"[OK] wrote {csv_path}")

    # Coverage
    coverage = subset_coverage_by_tier(lvis_train, args.subset_json)

    # Print concise text summary
    print("\n=== mini-LVIS REPORT ===")
    print(f"subset images: {total_img_subset}, subset annotations: {total_ann_subset}")
    print("Tier mix (target vs actual, by images):")
    for t, name in zip("cfr", ["Common","Frequent","Rare"]):
        tgt = target_mix[t]; act = actual_mix[t]
        print(f"  {name:<8}  target={tgt:6.3f}   actual={act:6.3f}")
    print("\nPer-tier class×size alignment (normalized):")
    for t, name in zip("cfr", ["Common","Frequent","Rare"]):
        m = tier_metrics[t]
        print(f"  {name:<8}  pearson={m['pearson']:.3f}  mse={m['mse_norm']:.6f}  "
              f"k_full={m['k_nonzero_full']}  k_mini={m['k_nonzero_mini']}  "
              f"sum_full={m['sum_full']}  sum_mini={m['sum_mini']}")
    print("\nCoverage (categories present ≥1 ann in subset):")
    for t, name in zip("cfr", ["Common","Frequent","Rare"]):
        have, total, frac = coverage[t]
        print(f"  {name:<8}  {have}/{total}  ({frac*100:.1f}%)")

    # Plot PNG
    out_png = f"{args.save_prefix}_overview.png"
    plot_report(out_png, tier_metrics, tier_scatter_data, target_mix, actual_mix)
    print(f"\n[OK] wrote {out_png}")

    # Markdown summary for experiment logs
    md_path = f"{args.save_prefix}_summary.md"
    with open(md_path, "w") as mf:
        mf.write("# mini-LVIS Report\n\n")
        mf.write(f"- Subset images: **{total_img_subset}**\n")
        mf.write(f"- Subset annotations: **{total_ann_subset}**\n\n")
        mf.write("## Tier mix (image share)\n\n")
        mf.write("| Tier | Target | Actual |\n|---|---:|---:|\n")
        for t, name in zip("cfr", ["Common","Frequent","Rare"]):
            mf.write(f"| {name} | {target_mix[t]:.3f} | {actual_mix[t]:.3f} |\n")
        mf.write("\n## Per-tier class×size alignment\n\n")
        mf.write("| Tier | Pearson | MSE (norm) | Nonzero (full) | Nonzero (mini) | Sum full | Sum mini |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for t, name in zip("cfr", ["Common","Frequent","Rare"]):
            m = tier_metrics[t]
            mf.write(f"| {name} | {m['pearson']:.3f} | {m['mse_norm']:.6f} | {m['k_nonzero_full']} | {m['k_nonzero_mini']} | {m['sum_full']} | {m['sum_mini']} |\n")
        mf.write("\n## Coverage (categories with ≥1 ann)\n\n")
        mf.write("| Tier | Covered / Total | % |\n|---|---:|---:|\n")
        for t, name in zip("cfr", ["Common","Frequent","Rare"]):
            have, total, frac = coverage[t]
            mf.write(f"| {name} | {have}/{total} | {frac*100:.1f}% |\n")
        mf.write(f"\n**Figure:** `{os.path.basename(out_png)}`\n")
    print(f"[OK] wrote {md_path}")


if __name__ == "__main__":
    main()

