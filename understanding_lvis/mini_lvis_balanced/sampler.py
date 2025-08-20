# mini_lvis_balanced/sampler.py
import numpy as np
from collections import Counter, defaultdict
from lvis import LVIS

# ---------- Size bucket ----------
def _size_bucket_from_bbox(bbox, areaRng):
    w, h = bbox[2], bbox[3]
    if w < 1 or h < 1:
        return None
    a = w * h
    if a < areaRng[0]: return "S"
    if a < areaRng[1]: return "M"
    return "L"

# ---------- Metadata helpers ----------
def _cat_meta(lvis):
    cats = lvis.load_cats(lvis.get_cat_ids())
    by_id = {c["id"]: c for c in cats}
    tiers = {"c": set(), "f": set(), "r": set()}
    for c in cats:
        tiers[c["frequency"]].add(c["id"])
    return by_id, tiers

def _tier_of(cat_meta, cid):
    return cat_meta[cid]["frequency"]

# ---------- Compute raw tier stats ----------
def compute_tier_stats(lvis):
    cat_meta, tiers = _cat_meta(lvis)
    # classes per tier
    classes = {t: len(tiers[t]) for t in "cfr"}
    # annotations per tier
    ann_counts = {"c":0, "f":0, "r":0}
    for ann in lvis.anns.values():
        ann_counts[_tier_of(cat_meta, ann["category_id"])] += 1
    # images per tier (≥1 instance)
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

# ---------- Target tier mix (blended) ----------
def compute_target_tier_mix(classes, ann_counts, alpha=0.6):
    """
    alpha in [0,1]: weight on annotation-share; (1-alpha) on class-share.
    Using your train stats default is reasonable: annotation share is extremely skewed;
    blending with class share preserves the tail’s presence.
    """
    cls_sum = sum(classes.values())
    ann_sum = sum(ann_counts.values())
    cls_share = np.array([classes[t]/cls_sum for t in "cfr"])
    ann_share = np.array([ann_counts[t]/ann_sum for t in "cfr"])
    mix = alpha * ann_share + (1 - alpha) * cls_share
    mix = mix / mix.sum()
    return {"c": mix[0], "f": mix[1], "r": mix[2]}

# ---------- Reference & per-image matrices (within a tier) ----------
def build_ref_hist_tier(lvis, areaRng, cat_meta, tier):
    ref = Counter()
    for ann in lvis.anns.values():
        cid = ann["category_id"]
        if _tier_of(cat_meta, cid) != tier:
            continue
        sb = _size_bucket_from_bbox(ann["bbox"], areaRng)
        if sb is None:
            continue
        ref[f"{cid}_{sb}"] += 1
    return ref  # keys: f"{cid}_{S|M|L}"

def build_per_image_bins_tier(lvis, areaRng, cat_meta, tier):
    per_img = {}
    for im_id, ann_ids in lvis.img_ann_map.items():
        c = Counter()
        for aid in ann_ids:
            ann = lvis.anns[aid]
            cid = ann["category_id"]
            if _tier_of(cat_meta, cid) != tier: 
                continue
            sb = _size_bucket_from_bbox(ann["bbox"], areaRng)
            if sb is None:
                continue
            c[f"{cid}_{sb}"] += 1
        per_img[im_id] = c
    return per_img

def densify(per_img_bins, ref_hist):
    keys = list(ref_hist.keys())
    key_index = {k:i for i,k in enumerate(keys)}
    img_ids = list(per_img_bins.keys())
    M, K = len(img_ids), len(keys)
    X = np.zeros((M, K), dtype=np.int32)
    for i, iid in enumerate(img_ids):
        row = per_img_bins[iid]
        for k, v in row.items():
            j = key_index.get(k, None)
            if j is not None:
                X[i, j] = v
    ref_counts = np.array([ref_hist[k] for k in keys], dtype=np.float64)
    return keys, key_index, img_ids, X, ref_counts

# ---------- Objective pieces ----------
def score_classsize(v, ref_counts):
    r = v / np.maximum(ref_counts, 1e-9)
    m = r.mean()
    return ((r - m) ** 2).mean()

def tier_mix_penalty(n_c, n_f, n_r, target_share, total_imgs):
    if total_imgs <= 0:
        return 0.0
    share = np.array([n_c, n_f, n_r], dtype=np.float64) / float(total_imgs)
    tgt = np.array([target_share["c"], target_share["f"], target_share["r"]], dtype=np.float64)
    return ((share - tgt) ** 2).mean()

def rare_cap_penalty(imgs_by_class_counts, rare_quota):
    over = 0
    for cid, cap in rare_quota.items():
        over += max(0, imgs_by_class_counts.get(cid, 0) - cap)
    return float(over)

# ---------- Estimate rare quotas ----------
def estimate_rare_quotas(lvis, cat_meta, tier="r", rare_cap=2, target_rare_images=None):
    """
    Heuristic: proportional to per-class annotation counts, converted to image quotas,
    then clipped by rare_cap, and scaled to sum ≈ target_rare_images (if provided).
    """
    # per-class annots
    Hk = {}
    for ann in lvis.anns.values():
        cid = ann["category_id"]
        if _tier_of(cat_meta, cid) == tier:
            Hk[cid] = Hk.get(cid, 0) + 1

    # per-class avg annots per image
    img_presence = {cid: len(lvis.cat_img_map[cid]) for cid in Hk.keys()}
    mu = {cid: max(1.0, Hk[cid] / max(1, img_presence[cid])) for cid in Hk.keys()}

    # raw quotas (annots -> images)
    quota = {cid: min(rare_cap, int(np.ceil(Hk[cid] / mu[cid]))) for cid in Hk.keys()}

    if target_rare_images is not None:
        total = sum(quota.values())
        if total > 0:
            scale = target_rare_images / total
            # scale and round while keeping at least 0 or 1 if you want some coverage
            quota = {cid: max(0, int(np.floor(q * scale))) for cid, q in quota.items()}

    return quota  # cid -> images (cap applied)

# ---------- Tier-stratified greedy with global regularizers ----------
def sample_balanced_greedy(
    lvis,
    total_images,
    areaRng,
    alpha=0.6,              # blend between annotation-share and class-share for tier mix
    tier_bonus=(1.0,1.0,0.6), # relative importance of class-size matching inside (c,f,r)
    lambda_tier=3.0,        # weight for tier-mix penalty
    lambda_cap=2.0,         # weight for rare-cap overage
    rare_cap=2,             # max images per rare class (soft via penalty)
    rng_seed=1337,
    pool_size=2048,
    debug=False
):
    """
    Returns: dict {image_id: [ann dicts]} forming the mini-LVIS subset.
    Strategy:
      1) Compute target tier mix (c/f/r) via blended shares.
      2) Allocate per-tier image budgets (B_c, B_f, B_r) ~ target mix * total_images.
      3) Within each tier, build ref and per-image matrices for class×size.
      4) Greedy pick images with score = sum_t (w_t * classsize_t) + λ_tier * tier_penalty + λ_cap * rare_overage
         while respecting tier budgets (soft, via penalties + hard “don’t exceed by much”).
    """
    rng = np.random.default_rng(rng_seed)
    cat_meta, tiers = _cat_meta(lvis)
    classes, ann_counts, _, _ = compute_tier_stats(lvis)

    # target tier mix & budgets
    target_mix = compute_target_tier_mix(classes, ann_counts, alpha=alpha)
    Bc = int(round(target_mix["c"] * total_images))
    Bf = int(round(target_mix["f"] * total_images))
    Br = total_images - Bc - Bf
    if debug:
        print(f"[mix] target mix (c,f,r) = {target_mix} -> budgets (Bc,Bf,Br)=({Bc},{Bf},{Br})")

    # build tier-wise matrices
    tier_data = {}
    for t in "cfr":
        ref = build_ref_hist_tier(lvis, areaRng, cat_meta, t)
        per_img = build_per_image_bins_tier(lvis, areaRng, cat_meta, t)
        keys, key_index, img_ids, X, ref_counts = densify(per_img, ref)
        tier_data[t] = dict(ref=ref, per_img=per_img, keys=keys, key_index=key_index,
                            img_ids=img_ids, X=X, ref_counts=ref_counts)

    # rare quotas (soft)
    rare_quota = estimate_rare_quotas(lvis, cat_meta, tier="r", rare_cap=rare_cap, target_rare_images=Br)

    # global tracking
    chosen = set()
    cur_counts = {"c": np.zeros_like(tier_data["c"]["ref_counts"], dtype=np.int64),
                  "f": np.zeros_like(tier_data["f"]["ref_counts"], dtype=np.int64),
                  "r": np.zeros_like(tier_data["r"]["ref_counts"], dtype=np.int64)}
    count_imgs_tier = {"c":0, "f":0, "r":0}

    # rare image-per-class counts (presence, not annots)
    rare_img_presence = defaultdict(int)  # cid -> #images selected containing cid

    # map image -> tier flags for quick “which tiers does this image touch?”
    # and per-tier vectors for fast increments
    img_info_cache = {}
    for t in "cfr":
        X = tier_data[t]["X"]
        ids = tier_data[t]["img_ids"]
        key_index = tier_data[t]["key_index"]
        for i, iid in enumerate(ids):
            rec = img_info_cache.get(iid, {"tiers": set(), "vecs": {}})
            if X[i].sum() > 0:
                rec["tiers"].add(t)
                rec["vecs"][t] = X[i]
            img_info_cache[iid] = rec

    # helper for rare presence update
    def rare_classes_in_image(iid):
        out = set()
        for aid in lvis.img_ann_map[iid]:
            ann = lvis.anns[aid]
            cid = ann["category_id"]
            if _tier_of(cat_meta, cid) == "r":
                out.add(cid)
        return out

    # prebuild candidate pool
    all_ids = np.array(list(img_info_cache.keys()), dtype=np.int64)
    rng.shuffle(all_ids)

    def score_if_add(iid):
        # class-size term per tier (weighted)
        s_cs = 0.0
        for t, w_t in zip("cfr", tier_bonus):
            vec = img_info_cache[iid]["vecs"].get(t, None)
            if vec is None:
                continue
            cur = cur_counts[t] + vec
            s_cs += w_t * score_classsize(cur, tier_data[t]["ref_counts"])

        # tier mix term
        add_c = 1 if "c" in img_info_cache[iid]["tiers"] else 0
        add_f = 1 if "f" in img_info_cache[iid]["tiers"] else 0
        add_r = 1 if "r" in img_info_cache[iid]["tiers"] else 0
        s_tier = tier_mix_penalty(
            count_imgs_tier["c"]+add_c,
            count_imgs_tier["f"]+add_f,
            count_imgs_tier["r"]+add_r,
            target_mix,
            len(chosen)+1
        )

        # rare cap term (soft)
        over = 0
        if add_r:
            rset = rare_classes_in_image(iid)
            # count image-level presence if this image added
            for cid in rset:
                over += max(0, (rare_img_presence.get(cid, 0) + 1) - rare_quota.get(cid, 0))
        s_cap = float(over)

        return s_cs + lambda_tier * s_tier + lambda_cap * s_cap

    # greedy loop
    while len(chosen) < total_images:
        # create a beam of remaining candidates
        remaining = list(set(all_ids) - chosen)
        if not remaining: break
        pool = remaining if len(remaining) <= pool_size else list(rng.choice(remaining, size=pool_size, replace=False))
        best_iid, best_s = None, float("inf")
        for iid in pool:
            s = score_if_add(iid)
            if s < best_s:
                best_s, best_iid = s, iid

        if best_iid is None:
            break

        # commit best
        chosen.add(best_iid)
        for t in "cfr":
            vec = img_info_cache[best_iid]["vecs"].get(t, None)
            if vec is not None:
                cur_counts[t] += vec
                count_imgs_tier[t] += 1
        # update rare presence
        for cid in rare_classes_in_image(best_iid):
            rare_img_presence[cid] += 1

        # optional: tiny hard guard to avoid huge overdraw in any tier
        # allow up to +5% slack beyond target budget for each tier
        for t, B in zip("cfr", [Bc, Bf, Br]):
            if count_imgs_tier[t] > int(1.05 * B):
                # remove candidates that only contribute to this tier
                all_ids = np.array([iid for iid in all_ids if not (img_info_cache[iid]["tiers"] == {t})], dtype=np.int64)

        if debug and len(chosen) % 2000 == 0:
            print(f"[greedy] chosen={len(chosen)}  mix=({count_imgs_tier['c']},{count_imgs_tier['f']},{count_imgs_tier['r']}) / ({Bc},{Bf},{Br})")

    # return subset as {im_id: [ann dicts]}
    return {int(i): [lvis.anns[a] for a in lvis.img_ann_map[int(i)]] for i in chosen}

