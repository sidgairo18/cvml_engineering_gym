import argparse, json, math, os, yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO as COCO_API
from lvis import LVIS as LVIS_API

def load_cfg(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def build_presence_matrix_coco(coco):
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    catid_to_idx = {c:i for i,c in enumerate(cat_ids)}
    N, C = len(img_ids), len(cat_ids)
    pres = np.zeros((N, C), dtype=np.uint8)
    for n, img_id in enumerate(tqdm(img_ids, desc="COCO images")):
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        cats = {a['category_id'] for a in anns}
        for cid in cats:
            pres[n, catid_to_idx[cid]] = 1
    cats = coco.loadCats(cat_ids)
    names = [c['name'] for c in cats]
    return pres, img_ids, cat_ids, names

def build_presence_matrix_lvis(lvis):
    img_ids = list(lvis.imgs.keys())
    cat_ids = list(lvis.cats.keys())
    catid_to_idx = {c:i for i,c in enumerate(cat_ids)}
    N, C = len(img_ids), len(cat_ids)
    pres = np.zeros((N, C), dtype=np.uint8)
    for n, img_id in enumerate(tqdm(img_ids, desc="LVIS images")):
        ann_ids = lvis.get_ann_ids(img_ids=[img_id])
        anns = lvis.load_anns(ann_ids)
        cats = {a['category_id'] for a in anns}
        for cid in cats:
            pres[n, catid_to_idx[cid]] = 1
    names = [lvis.cats[c]['name'] for c in cat_ids]
    return pres, img_ids, cat_ids, names

def compute_pmi(pres, smoothing=1.0, min_images_per_class=50, names=None):
    N, C = pres.shape
    counts = pres.sum(axis=0)                 # class presence count
    keep = counts >= min_images_per_class
    idxs = np.where(keep)[0]
    pres = pres[:, idxs]; names = [names[i] for i in idxs] if names else None
    counts = counts[idxs]; Ck = len(idxs)
    # joint counts
    joint = pres.T @ pres                     # Ck x Ck
    # smoothing
    joint = joint.astype(np.float64) + smoothing
    counts = counts.astype(np.float64) + smoothing
    Nf = float(N + smoothing * Ck)
    # probabilities
    pA = counts / Nf
    pAB = joint / Nf
    # PMI (diag is PMI with itself, ignore)
    pmi = np.log(pAB) - (np.log(pA)[:,None] + np.log(pA)[None,:])
    # build pair table (upper triangle without diag)
    rows = []
    for i in range(Ck):
        for j in range(i+1, Ck):
            rows.append({
                "i": int(i), "j": int(j),
                "A": names[i], "B": names[j],
                "count_A": int(counts[i]), "count_B": int(counts[j]),
                "count_AB": int(pAB[i,j]*Nf - smoothing),  # approx unsmoothed joint
                "PMI": float(pmi[i,j])
            })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["COCO", "LVIS"], required=True)
    ap.add_argument("--cfg", default="./configs/dataset_paths.yaml")
    ap.add_argument("--pairs_cfg", default="./configs/pairs_config.yaml")
    ap.add_argument("--outdir", default="augment_eval/outputs/pairs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dscfg = load_cfg(args.cfg)[args.ds]
    pcfg   = load_cfg(args.pairs_cfg)

    if args.ds == "COCO":
        coco = COCO_API(dscfg["ann"])
        pres, img_ids, cat_ids, names = build_presence_matrix_coco(coco)
    else:
        lvis = LVIS_API(dscfg["ann"])
        pres, img_ids, cat_ids, names = build_presence_matrix_lvis(lvis)

    df = compute_pmi(
        pres,
        smoothing=pcfg.get("smoothing", 1.0),
        min_images_per_class=pcfg.get("min_images_per_class", 50),
        names=names
    )
    df.sort_values("PMI", ascending=False, inplace=True)
    topk = df.head(pcfg["top_k_typical"]).copy()
    bottomk = df.tail(pcfg["bottom_k_atypical"]).copy()

    df.to_csv(os.path.join(args.outdir, f"{args.ds}_pairs_all.csv"), index=False)
    topk.to_csv(os.path.join(args.outdir, f"{args.ds}_pairs_typical_topk.csv"), index=False)
    bottomk.to_csv(os.path.join(args.outdir, f"{args.ds}_pairs_atypical_bottomk.csv"), index=False)

    # JSON export for downstream
    out = {
        "typical": topk.to_dict(orient="records"),
        "atypical_candidates": bottomk.to_dict(orient="records")
    }
    with open(os.path.join(args.outdir, f"{args.ds}_pairs_selected.json"), "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()

