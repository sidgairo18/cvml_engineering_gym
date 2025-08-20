# mini_lvis_balanced

A tier-aware sampler for **LVIS v1** that preserves the **long-tailed structure** while producing a compact train subset.

## Why this design?

LVIS is extremely skewed. Your train stats show:
- Annotations dominated by *frequent* classes,
- Many *rare* classes with very few instances,
- Overlaps in image membership by tier.

If we match annotation proportions exactly, **rare disappears**. If we force ≥1 image per class, **tail dominates**.  
We therefore:
1) Compute a **target tier mix** = α·(annotation share) + (1−α)·(class share) — default α=0.6.  
   This keeps the tail present without overpowering head/mid.
2) Allocate per-tier image budgets (c/f/r) from that mix.
3) **Greedy select** images with objective:
   - **Match class×size (S/M/L)** *inside each tier*,
   - **Keep overall tier shares** near target (penalty λ_tier),
   - **Soft rare caps** (λ_cap) so no rare class soaks up too many images.

This yields a mini-LVIS that:
- Keeps **all three tiers** visible,
- Maintains **head/mid fidelity** for ablations,
- Avoids **tail domination**.

## Install
`pip install -r requirements.txt`

## Run
```
python -m mini_lvis_balanced.main
--train_json /path/to/lvis_v1_train.json
--save_file_name lvis_mini_balanced_25k
--total_images 25000
--alpha 0.6
--tier_bonus 1.0,1.0,0.6
--lambda_tier 3.0
--lambda_cap 2.0
--rare_cap 2
--seed 1337
--debug
```
(Optionally add `--val_json` and `--plot_tiers` for quick bars.)

## Outputs
- `lvis_mini_balanced_25k.json` — LVIS-style subset (original IDs)
- (optional) pruned val/preds for eval on kept categories

## Suggested defaults (based on your stats)

* α (blend): 0.6 (gives rare some budget without overwhelming frequent).
* tier_bonus (c,f,r): 1.0, 1.0, 0.6 (rare’s class×size matching matters, but less).
* λ_tier: 3.0 (keeps overall tier shares tight).
* λ_cap: 2.0, rare_cap: 2 (rare per-class soft cap ~2 images).
* pool_size: 2048 (speed/quality trade-off).
