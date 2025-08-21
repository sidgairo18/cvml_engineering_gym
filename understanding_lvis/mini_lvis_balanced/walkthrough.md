📂 main.py — the CLI entrypoint

This is the user-facing driver: you call it with LVIS train JSON + config flags, and it runs the sampler + writes out results.

Functions:

parse_args()

Sets up command-line arguments (using argparse).

Includes:

--train_json: path to LVIS v1 train file.

--total_images: how many images to keep in mini-LVIS.

Hyperparameters for tier-mix (alpha), tier bonuses, penalties (lambda_tier, lambda_cap), rare cap.

Size bucket thresholds (area_small, area_medium).

Random seed, greedy pool size, debug flag.

Optional val/pruning arguments (paths to val JSON, predictions JSON).

Returns the parsed args.

main()

Loads LVIS train dataset.

Optionally: plots tier bar plots comparing train vs val (plot_tiers).

Calls sample_balanced_greedy() from sampler.py to actually select the subset.

Writes the subset to a new JSON using write_lvis_subset() from tools.py.

Optionally: prunes val and prediction JSONs down to the kept categories, for evaluation consistency.

In short: main.py ties everything together. It takes user input, runs the sampler, saves the mini-LVIS JSON, and optionally produces eval-ready files.

📂 sampler.py — the sampler logic

This file contains the core logic for building a balanced mini-LVIS. It handles tier stats, target mix, class-size matching, rare control, and the greedy search itself.

Tier + metadata helpers

_size_bucket_from_bbox(bbox, areaRng)

Buckets an object annotation into S / M / L size bins using area thresholds.

Used to preserve object-size distribution when sampling.

_cat_meta(lvis)

Loads all categories from LVIS.

Builds two structures:

by_id: cat_id → category metadata (including frequency tier).

tiers: dictionary mapping 'c', 'f', 'r' → sets of category ids belonging to Common/Frequent/Rare.

_tier_of(cat_meta, cid)

Helper to get the frequency tier ('c', 'f', 'r') for a given category id.

Stats and target tier mix

compute_tier_stats(lvis)

Computes raw counts for each tier:

Number of categories,

Number of annotations,

Number of images containing ≥1 ann from that tier.

Also returns totals.

This describes how skewed LVIS is.

compute_target_tier_mix(classes, ann_counts, alpha=0.6)

LVIS is highly imbalanced (rare = many classes but few annotations).

This function blends:

annotation share (proportion of annotations per tier),

class share (proportion of categories per tier).

The blend weight alpha (default 0.6) controls how much you lean towards annotations vs categories.

Returns target tier mix (c/f/r fractions) to use when allocating images.

Reference histograms and per-image counts

build_ref_hist_tier(lvis, areaRng, cat_meta, tier)

Builds the reference histogram of counts per (cat_id, size) for one tier.

Example: (dog, S): 5000, (dog, M): 12000.

build_per_image_bins_tier(lvis, areaRng, cat_meta, tier)

For each image, counts how many annotations belong to each (cat_id, size) within that tier.

This allows us to know “if I add this image, what does it contribute to the histogram?”.

densify(per_img_bins, ref_hist)

Converts sparse per-image Counters into a dense NumPy matrix X of shape [num_images, num_keys].

ref_counts is the dense vector of full dataset counts per key.

Returns: keys, key_index, img_ids, X, ref_counts.

Objective functions

score_classsize(v, ref_counts)

Given a cumulative vector of sampled counts v, compares ratios vs full ref_counts.

Uses mean squared error of ratios → measures how well class×size distribution is preserved.

tier_mix_penalty(n_c, n_f, n_r, target_share, total_imgs)

Compares current tier distribution (images selected from c/f/r) against target tier mix.

Penalizes deviations.

rare_cap_penalty(imgs_by_class_counts, rare_quota)

Soft penalty: if a rare class exceeds its per-class quota (cap), add penalty.

Prevents any single rare class from hogging the budget.

Rare quota estimation

estimate_rare_quotas(lvis, cat_meta, tier="r", rare_cap=2, target_rare_images=None)

Heuristic to assign per-class quotas in rare tier:

Compute annotation count per rare class.

Estimate avg annots per image for that class.

Convert to “image quota” (annots/avg).

Clip by rare_cap.

Optionally rescale quotas to fit the rare tier’s total image budget.

Prevents over-representation of rare classes while ensuring some coverage.

The sampler itself

sample_balanced_greedy(...)

The main function.

Steps:

Compute target tier mix (blend of class + annotation share).

Allocate tier budgets (B_c, B_f, B_r).

Build per-tier reference histograms and per-image matrices.

Estimate rare quotas.

Initialize empty subset.

Greedy loop:

Candidate pool = random subset of remaining images.

For each candidate, compute score =
class-size mismatch (weighted per tier)
+ λ_tier * tier_mix_penalty
+ λ_cap * rare_cap_penalty.

Pick the image with the lowest score, commit it.

Update current counts and tier tallies.

If a tier exceeds its budget by >5%, remove pure-tier candidates to avoid runaway imbalance.

Stop when total_images is reached.

Returns: dict {image_id: [annotation dicts]} for the sampled subset.

📂 tools.py — IO, pruning, plotting

These are utilities for writing subsets, pruning eval sets, and visual diagnostics.

IO helpers

write_lvis_subset(lvis, imgs_best_sample, out_json_path)

Writes a valid LVIS-style JSON for the sampled subset.

Keeps info, licenses, categories, and only the selected images + annotations.

Preserves original IDs.

kept_categories(imgs_best_sample)

Returns a list of category IDs that appear in the subset.

Useful for pruning val/test.

Eval pruning

prune_gt(in_json, out_json, kept_cat_ids, drop_empty_images=False)

Prunes an LVIS val/test GT JSON to only the categories in kept_cat_ids.

Drops annotations from other categories.

If drop_empty_images=True, also removes images that end up with no annotations.

Useful for consistent evaluation (don’t evaluate on unseen classes).

prune_preds(in_pred, out_pred, kept_cat_ids)

Same pruning but for predictions JSON: removes detections for dropped categories.

Keeps evaluation aligned.

Plots

plot_tier_bars(train_stats, val_stats, save="lvis_tiers_train_val.png")

Creates bar charts comparing train vs val across tiers:

Number of classes,

Number of annotations,

Number of images.

Visual diagnostic to confirm the skew and check that train/val are consistent.

🔗 How the files fit together

main.py

User calls this script.

Parses CLI args.

Loads LVIS train.

Calls sample_balanced_greedy from sampler.py.

Writes results with write_lvis_subset from tools.py.

Optionally prunes eval sets and plots tier stats.

sampler.py

The heart of the project.

Defines how we measure distribution match (class×size), tier mix, rare quotas.

Greedy sampling builds the mini-LVIS with balanced long-tail.

tools.py

Convenience helpers for writing subsets, pruning eval sets, and plotting tier distributions.

Keeps main.py clean and modular.

✅ This design ensures:

Faithfulness: head/mid distributions preserved.

Long tail present: rare classes included but capped.

Evaluation aligned: val/test pruned so metrics are meaningful.
