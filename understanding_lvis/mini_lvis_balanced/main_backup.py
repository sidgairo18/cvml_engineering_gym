# mini_lvis_balanced/main.py
import argparse
from lvis import LVIS
from sampler_backup import sample_balanced_greedy, compute_tier_stats, compute_target_tier_mix
from tools import write_lvis_subset, kept_categories, prune_gt, prune_preds, plot_tier_bars

def parse_args():
    p = argparse.ArgumentParser("Balanced mini-LVIS sampler (preserves long tail structure)")
    p.add_argument("--train_json", required=True, help="Path to lvis_v1_train.json")
    p.add_argument("--save_file_name", default="lvis_mini_balanced_25k")
    p.add_argument("--total_images", type=int, default=25000)

    # Tier mix & penalties
    p.add_argument("--alpha", type=float, default=0.6, help="Blend weight for annotation vs class shares")
    p.add_argument("--tier_bonus", type=str, default="1.0,1.0,0.6", help="Weights for (c,f,r) class-size terms")
    p.add_argument("--lambda_tier", type=float, default=3.0)
    p.add_argument("--lambda_cap", type=float, default=2.0)
    p.add_argument("--rare_cap", type=int, default=2)

    # Size buckets
    p.add_argument("--area_small", type=float, default=32.0)
    p.add_argument("--area_medium", type=float, default=96.0)

    # Greedy control
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--pool_size", type=int, default=2048)
    p.add_argument("--debug", action="store_true")

    # Optional: val/pruning & plots
    p.add_argument("--val_json", type=str, help="Path to lvis_v1_val.json for plots/pruning")
    p.add_argument("--plot_tiers", action="store_true")
    p.add_argument("--out_val_json", type=str, help="Output path for pruned val")
    p.add_argument("--drop_empty_images", action="store_true")
    p.add_argument("--pred_json", type=str, help="Detections to prune")
    p.add_argument("--out_pred_json", type=str)
    return p.parse_args()

def main():
    args = parse_args()
    areaRng = [args.area_small**2, args.area_medium**2, 1e5**2]
    tier_bonus = tuple(float(x) for x in args.tier_bonus.split(","))

    lvis = LVIS(args.train_json)

    # (optional) sanity plots of train vs val tiers
    print("Computing tier stats ...")
    train_classes, train_anns, _, _ = compute_tier_stats(lvis)
    print("Computer tier stats done!")
    if args.val_json and args.plot_tiers:
        from lvis import LVIS as LVISval
        lvis_val = LVISval(args.val_json)
        val_classes, val_anns, _, _ = compute_tier_stats(lvis_val)
        train_stats = {"num_classes": train_classes, "num_annotations": train_anns,
                       "num_images": {"c":0,"f":0,"r":0}}
        val_stats = {"num_classes": val_classes, "num_annotations": val_anns,
                     "num_images": {"c":0,"f":0,"r":0}}
        plot_tier_bars(train_stats, val_stats)

    # Sample!
    print("Runing sampling ...")
    subset = sample_balanced_greedy(
        lvis=lvis,
        total_images=args.total_images,
        areaRng=areaRng,
        alpha=args.alpha,
        tier_bonus=tier_bonus,
        lambda_tier=args.lambda_tier,
        lambda_cap=args.lambda_cap,
        rare_cap=args.rare_cap,
        rng_seed=args.seed,
        pool_size=args.pool_size,
        debug=args.debug
    )

    out_json = f"{args.save_file_name}.json"
    write_lvis_subset(lvis, subset, out_json)
    print(f"[OK] wrote {out_json} with {len(subset)} images")

    # eval pruning (optional)
    kept = kept_categories(subset)
    if args.val_json and args.out_val_json:
        outv = prune_gt(args.val_json, args.out_val_json, kept, args.drop_empty_images)
        print(f"[OK] pruned val json -> {outv}")
    if args.pred_json and args.out_pred_json:
        outp = prune_preds(args.pred_json, args.out_pred_json, kept)
        print(f"[OK] pruned predictions -> {outp}")

if __name__ == "__main__":
    main()

