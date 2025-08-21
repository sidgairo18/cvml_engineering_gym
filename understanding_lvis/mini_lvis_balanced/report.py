# mini_lvis_balanced/report.py
import argparse
import numpy as np
from collections import Counter
from lvis import LVIS
import matplotlib.pyplot as plt
import csv

# ---------- helpers ----------
def _cat_meta(lvis):
    cats = lvis.load_cats(lvis.get_cat_ids())
    by_id = {c["id"]: c for c in cats}
    tiers = {"c": set(), "f": set(), "r": set()}
    for c in cats:
        tiers[c["frequency"]].add(c["id"])
    return by_id, tiers

def _tier_of(cat_meta, cid):
    return cat_meta[cid]["frequency"]

def _compute_tier_stats(lvis):
    cat_meta, tiers = _cat_meta(lvis)
    classes = {t: len(tiers[t]) for t in "cfr"}
    ann_counts = {"c": 0, "f": 0, "r": 0}
    for ann in lvis.anns.values():
        ann_counts[_tier_of(cat_meta, ann["category_id"])] += 1
    img_sets = {"c": set(), "f": set(), "r": set()}
    for t in "cfr":
        for cid in tiers[t]:
            img_sets[t].update(lvis.cat_img_map.get(cid, []))
    images = {t: len(img_sets[t]) for t in "cfr"}
    totals = {
        "classes": sum(classes.values()),
        "annotations": sum(ann_counts.values()),
        "images": len(lvis.imgs),
    }
    return classes, ann_counts, images, totals, cat_meta, tiers, img_sets

def _category_ann_count(lvis):
    c = Counter()
    for ann in lvis.anns.values():
        c[ann["category_id"]] += 1
    return c

def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0 or x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

def _plot_class_annotation_bars(lvis_full, lvis_mini, f_tiers, f_meta,
                                save_prefix, seed=1337, n_per_tier=20,
                                ylog=False, eps=1e-8):
    """
    Draw two figures comparing per-class annotation ratios (full vs mini) for
    60 random categories: 20 from each tier (c,f,r).
    Two normalizations:
      (A) class_ann / total_images   (per dataset)
      (B) class_ann / total_annots   (per dataset)

    Extras:
      * ylog=True -> log-scale y-axis (with epsilon)
      * shaded backgrounds + colored tick labels for c/f/r blocks
    Outputs:
      f"{save_prefix}_norm_images.png"
      f"{save_prefix}_norm_annots.png"
    """
    rng = np.random.default_rng(seed)

    # sample 20 per tier and keep them grouped in order: [20 c][20 f][20 r]
    picks_by_tier = {}
    for t in "cfr":
        ids = sorted(list(f_tiers[t]))
        rng.shuffle(ids)
        picks_by_tier[t] = ids[:min(n_per_tier, len(ids))]

    sampled = picks_by_tier["c"] + picks_by_tier["f"] + picks_by_tier["r"]
    counts_c = len(picks_by_tier["c"])
    counts_f = len(picks_by_tier["f"])
    counts_r = len(picks_by_tier["r"])

    # per-category ann counts
    from collections import Counter
    def per_cat_ann(lvis):
        c = Counter()
        for a in lvis.anns.values():
            c[a["category_id"]] += 1
        return c
    full_per_cat = per_cat_ann(lvis_full)
    mini_per_cat = per_cat_ann(lvis_mini)

    # totals
    full_total_imgs = max(1, len(lvis_full.imgs))
    mini_total_imgs = max(1, len(lvis_mini.imgs))
    full_total_anns = max(1, sum(full_per_cat.values()))
    mini_total_anns = max(1, sum(mini_per_cat.values()))

    # labels & tiers
    names = [f_meta[cid]["name"] for cid in sampled]
    tiers_per_class = [f_meta[cid]["frequency"] for cid in sampled]  # 'c','f','r'
    label_txt = [n if len(n) <= 18 else n[:16] + "…" for n in names]

    # helper: compute series
    #import numpy as np
    def build_series(norm="images"):
        full_vals, mini_vals = [], []
        for cid in sampled:
            fa = full_per_cat.get(cid, 0)
            ma = mini_per_cat.get(cid, 0)
            if norm == "images":
                full_vals.append((fa / full_total_imgs) + (eps if ylog else 0.0))
                mini_vals.append((ma / mini_total_imgs) + (eps if ylog else 0.0))
            else:
                full_vals.append((fa / full_total_anns) + (eps if ylog else 0.0))
                mini_vals.append((ma / mini_total_anns) + (eps if ylog else 0.0))
        return np.array(full_vals, dtype=float), np.array(mini_vals, dtype=float)

    # styling helpers
    tier_colors = {"c": "#2b8a3e", "f": "#f08c00", "r": "#e03131"}  # green / orange / red (ticks & spans)
    def shade_tier_blocks(ax):
        # ranges: [0, c-1], [c, c+f-1], [c+f, c+f+r-1]
        spans = [
            (0, max(0, counts_c - 1), "c"),
            (counts_c, max(counts_c, counts_c + counts_f - 1), "f"),
            (counts_c + counts_f, max(counts_c + counts_f, counts_c + counts_f + counts_r - 1), "r")
        ]
        for lo, hi, t in spans:
            if hi < lo:  # empty tier slice
                continue
            ax.axvspan(lo - 0.5, hi + 0.5, color=tier_colors[t], alpha=0.06)

    def color_tick_labels(ax, start_index=0):
        for i, tick in enumerate(ax.get_xticklabels()):
            t = tiers_per_class[start_index + i]
            tick.set_color(tier_colors[t])

    # draw routine (split into 2 rows of 30 items for readability)
    def draw_and_save(full_vals, mini_vals, title, out_png):
        import matplotlib.pyplot as plt
        K = len(sampled)
        halves = [slice(0, min(30, K)), slice(min(30, K), K)]
        fig, axs = plt.subplots(2, 1, figsize=(20, 9), constrained_layout=False)
        for row_idx, (ax, sl) in enumerate(zip(axs, halves)):
            idx = np.arange(sl.stop - sl.start)
            fv = full_vals[sl]
            mv = mini_vals[sl]
            labs = label_txt[sl]

            # tier shading for this window
            shade_tier_blocks(ax)

            width = 0.42
            ax.bar(idx - width/2, fv, width=width, label="Full (train)")
            ax.bar(idx + width/2, mv, width=width, label="Mini")
            ax.set_xticks(idx)
            ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=9)
            ax.set_ylabel("Annotation Ratio")

            # color xtick labels by tier
            start_index = sl.start
            color_tick_labels(ax, start_index=start_index)

            if ylog:
                ax.set_yscale("log")
            ax.grid(axis="y", linestyle=":", alpha=0.35)

        axs[0].legend(loc="upper right")
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

    # (A) normalized by total image count
    f_img, m_img = build_series(norm="images")
    draw_and_save(
        f_img, m_img,
        "Per-class Annotation Ratios — normalized by total image count (random 60: 20 c, 20 f, 20 r)",
        f"{save_prefix}_norm_images.png"
    )

    # (B) normalized by total annotation count
    f_ann, m_ann = build_series(norm="annots")
    draw_and_save(
        f_ann, m_ann,
        "Per-class Annotation Ratios — normalized by total annotation count (random 60: 20 c, 20 f, 20 r)",
        f"{save_prefix}_norm_annots.png"
    )


# ---------- main reporting ----------
def main():
    ap = argparse.ArgumentParser("Mini-LVIS report (tier ratios, coverage, overall, per-class dump)")
    ap.add_argument("--full_train_json", required=True, help="Path to lvis_v1_train.json (full)")
    ap.add_argument("--mini_json", required=True, help="Path to mini LVIS JSON (sampled subset)")
    ap.add_argument("--save_fig", default="report_tier_ratios.png", help="Where to save the bar plot")
    ap.add_argument("--out_csv", type=str, help="Optional CSV to export tier+overall summary")
    ap.add_argument("--out_per_class_csv", type=str, help="Optional CSV to export per-class stats")
    ap.add_argument("--no_fig", action="store_true", help="Skip figure")
    ap.add_argument("--class_bar_fig", type=str,
                    help="If set, save per-class bar charts here (prefix). Two images will be written: *_norm_images.png and *_norm_annots.png")
    ap.add_argument("--class_bar_seed", type=int, default=1337,
                    help="Random seed for sampling 20 classes per tier (c/f/r)")
    ap.add_argument("--class_bar_log", action="store_true",
                    help="Use log-scale on Y axis for per-class bar charts")
    ap.add_argument("--class_bar_eps", type=float, default=1e-8,
                    help="Small epsilon added before log to avoid zeros")
    args = ap.parse_args()

    # Load datasets
    print("Annotations Loading ...")
    lvis_full = LVIS(args.full_train_json)
    lvis_mini = LVIS(args.mini_json)
    print("Annotations Loaded!")

    # Tier stats
    f_classes, f_anns, f_imgs, f_totals, f_meta, f_tiers, _ = _compute_tier_stats(lvis_full)
    m_classes, m_anns, m_imgs, m_totals, m_meta, m_tiers, _ = _compute_tier_stats(lvis_mini)

    # Coverage per tier (category has ≥1 annotation in mini)
    cats_with_anns_mini = set(a["category_id"] for a in lvis_mini.anns.values())
    coverage = {
        t: len(f_tiers[t] & cats_with_anns_mini) / max(1, len(f_tiers[t]))
        for t in "cfr"
    }

    # Ratios per tier
    ann_ratio = {t: m_anns[t] / max(1, f_anns[t]) for t in "cfr"}
    img_ratio = {t: m_imgs[t] / max(1, f_imgs[t]) for t in "cfr"}

    # Per-tier Pearson
    full_per_cat = _category_ann_count(lvis_full)
    mini_per_cat = _category_ann_count(lvis_mini)
    pearson = {}
    for t in "cfr":
        ids = sorted(list(f_tiers[t]))
        x = [full_per_cat.get(cid, 0) for cid in ids]
        y = [mini_per_cat.get(cid, 0) for cid in ids]
        pearson[t] = _pearson(x, y)

    # Overall ratios
    overall_ann_ratio = m_totals["annotations"] / max(1, f_totals["annotations"])
    overall_img_ratio = m_totals["images"] / max(1, f_totals["images"])

    # Print report
    def _fmt(d): return ", ".join([f"{k}:{d[k]}" for k in "cfr"])
    print("\n=== LVIS Tier Summary ===")
    print(f"Full  - classes: [{_fmt(f_classes)}], anns: [{_fmt(f_anns)}], imgs: [{_fmt(f_imgs)}]")
    print(f"Mini  - classes: [{_fmt(m_classes)}], anns: [{_fmt(m_anns)}], imgs: [{_fmt(m_imgs)}]")
    print("\nCoverage per tier:", {k: f"{coverage[k]*100:.1f}%" for k in "cfr"})
    print("Annotation ratio:", {k: f"{ann_ratio[k]:.3f}" for k in "cfr"}, f"overall={overall_ann_ratio:.3f}")
    print("Image ratio     :", {k: f"{img_ratio[k]:.3f}" for k in "cfr"}, f"overall={overall_img_ratio:.3f}")
    print("Pearson:", {k: f"{pearson[k]:.3f}" if pearson[k]==pearson[k] else "nan" for k in "cfr"})

    # Tier/overall CSV
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["tier", "coverage_fraction", "ann_ratio", "img_ratio", "pearson"])
            for t in "cfr":
                writer.writerow([
                    {"c": "common", "f": "frequent", "r": "rare"}[t],
                    f"{coverage[t]:.4f}",
                    f"{ann_ratio[t]:.4f}",
                    f"{img_ratio[t]:.4f}",
                    f"{pearson[t]:.4f}" if pearson[t]==pearson[t] else "nan"
                ])
            writer.writerow(["overall", "", f"{overall_ann_ratio:.4f}", f"{overall_img_ratio:.4f}", ""])
        print(f"[OK] wrote CSV summary -> {args.out_csv}")

    # NEW: Per-class CSV (id, name, tier, counts, coverage, ratios)
    if args.out_per_class_csv:
        # Build quick lookups
        full_img_per_cat = {cid: len(lvis_full.cat_img_map.get(cid, [])) for cid in f_meta.keys()}
        mini_img_per_cat = {cid: len(lvis_mini.cat_img_map.get(cid, [])) for cid in f_meta.keys()}

        with open(args.out_per_class_csv, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "category_id", "category_name", "tier",
                "full_ann_count", "mini_ann_count",
                "full_img_count", "mini_img_count",
                "covered_in_mini",
                "ann_ratio", "img_ratio"
            ])
            for cid, cinfo in sorted(f_meta.items(), key=lambda kv: kv[0]):
                name = cinfo["name"]
                tier = cinfo["frequency"]  # 'c'/'f'/'r'
                fa = int(full_per_cat.get(cid, 0))
                ma = int(mini_per_cat.get(cid, 0))
                fi = int(full_img_per_cat.get(cid, 0))
                mi = int(mini_img_per_cat.get(cid, 0))
                covered = 1 if ma > 0 else 0
                ann_r = (ma / fa) if fa > 0 else 0.0
                img_r = (mi / fi) if fi > 0 else 0.0
                writer.writerow([cid, name, tier, fa, ma, fi, mi, covered, f"{ann_r:.6f}", f"{img_r:.6f}"])
        print(f"[OK] wrote per-class CSV -> {args.out_per_class_csv}")

    # Figure
    if not args.no_fig:
        tiers = ["c","f","r"]; names = ["Common","Frequent","Rare"]
        x = np.arange(3); w = 0.35
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].bar(x - w/2, [ann_ratio[t] for t in tiers], width=w, label="Annotations")
        axs[0].bar(x + w/2, [img_ratio[t]  for t in tiers], width=w, label="Images")
        axs[0].set_xticks(x); axs[0].set_xticklabels(names)
        axs[0].set_title("Mini / Full Ratios by Tier"); axs[0].legend()
        axs[0].axhline(1.0, linestyle="--", linewidth=1)
        axs[1].bar(x, [coverage[t]*100 for t in tiers], width=0.6)
        axs[1].set_xticks(x); axs[1].set_xticklabels(names)
        axs[1].set_title("Category Coverage (%) by Tier"); axs[1].set_ylim(0, 100)
        fig.suptitle("Mini-LVIS: Tier Ratios & Coverage", fontsize=12)
        fig.tight_layout()
        fig.savefig(args.save_fig, dpi=200); plt.close(fig)
        print(f"[OK] saved figure -> {args.save_fig}")

    # --- Per-class bars (60 random: 20 per tier) ---
    if args.class_bar_fig:
        _plot_class_annotation_bars(
            lvis_full=lvis_full,
            lvis_mini=lvis_mini,
            f_tiers=f_tiers,          # from _compute_tier_stats(lvis_full)
            f_meta=f_meta,            # category metadata from full set
            save_prefix=args.class_bar_fig,
            seed=args.class_bar_seed,
            n_per_tier=20,
            ylog=args.class_bar_log,          # NEW
            eps=args.class_bar_eps            # NEW
        )
        print(f"[OK] saved per-class figures -> {args.class_bar_fig}_norm_images.png and *_norm_annots.png")

if __name__ == "__main__":
    main()
    exit(0)

