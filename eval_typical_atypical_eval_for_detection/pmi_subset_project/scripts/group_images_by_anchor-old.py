#!/usr/bin/env python3
# Group images into per-anchor folders for visual inspection.
# New: --draw overlays bounding boxes & labels (anchor vs partner highlighted).
#
# Modes:
#   original  : COCO/LVIS original images using manifests from build_subset_original_pairs.py
#   real_paste: outputs from build_subset_real_paste.py (images/ + annotations/*.json)
#   diffusion : diffusion outputs using manifest.jsonl with DONE entries (and optional masks)
#
# Examples
# --------
# Original (no draw):
#   python -m scripts.group_images_by_anchor \
#     --mode original \
#     --images_root /path/to/coco/train2017 \
#     --manifests_dir outputs/coco_subsets_original \
#     --out_dir viz/by_anchor_original \
#     --symlink
#
# Original (+draw, needs eval JSONs to get boxes):
#   python -m scripts.group_images_by_anchor \
#     --mode original \
#     --images_root /path/to/coco/train2017 \
#     --manifests_dir outputs/coco_subsets_original \
#     --ann_json_typical outputs/coco_subsets_original/typical.fixed.json \
#     --ann_json_atypical outputs/coco_subsets_original/atypical.fixed.json \
#     --out_dir viz/by_anchor_original_drawn \
#     --draw --draw_all   # draw all objects; omit --draw_all to draw only anchor/partner
#
# Real-paste (+draw from RP annotations):
#   python -m scripts.group_images_by_anchor \
#     --mode real_paste \
#     --rp_dir outputs/coco_subsets_real_paste \
#     --out_dir viz/by_anchor_real_paste_drawn \
#     --draw
#
# Diffusion (+draw bboxes from masks if provided in manifest):
#   python -m scripts.group_images_by_anchor \
#     --mode diffusion \
#     --images_root outputs/coco_subsets_diffusion/images \
#     --diffusion_manifest outputs/coco_subsets_diffusion/manifest.jsonl \
#     --out_dir viz/by_anchor_diffusion_drawn \
#     --draw

import os, sys, json, glob, shutil
from pathlib import Path

import cv2
import numpy as np

GREEN  = (50, 220, 50)    # anchor
BLUE   = (60, 130, 250)    # partner
GRAY   = (170, 170, 170)   # others
YELLOW = (0, 220, 220)     # typical tag
MAGENTA= (220, 50, 180)    # atypical tag

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _link_or_copy(src, dst, symlink=False):
    _ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        return
    if symlink:
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def _pad12(n):
    return f"{int(n):012d}"

def _draw_box(img, bbox, color, label=None):
    x,y,w,h = bbox
    x,y,w,h = int(x), int(y), int(w), int(h)
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    if label:
        tt = max(1, int(0.6 + 0.002*max(img.shape[:2])))
        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, tt, cv2.LINE_AA)

def _bbox_from_mask(path):
    M = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if M is None:
        return None
    ys, xs = np.where(M > 0)
    if len(xs)==0 or len(ys)==0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]

def _load_coco_index(ann_path):
    with open(ann_path, 'r') as f:
        data = json.load(f)
    # Build image_id -> [anns], cat_id -> name
    img_to_anns = {}
    for ann in data['annotations']:
        img_to_anns.setdefault(ann['image_id'], []).append(ann)
    id_to_name = {c['id']: c['name'] for c in data['categories']}
    images = {im['id']: im for im in data['images']}
    return images, img_to_anns, id_to_name

def group_original(images_root, manifests_dir, out_dir, symlink=False, limit_per_pair=None,
                   draw=False, draw_all=False, ann_json_typ=None, ann_json_aty=None):
    # Parse manifests
    man_paths = []
    for name in ["manifest_typical.jsonl", "manifest_atypical.jsonl"]:
        p = os.path.join(manifests_dir, name)
        if os.path.exists(p):
            man_paths.append(p)
    if not man_paths:
        raise SystemExit(f"No manifests found in {manifests_dir}")

    # Optional COCO indexes for drawing
    idx_typ = idx_aty = None
    if draw:
        if ann_json_typ and os.path.exists(ann_json_typ):
            idx_typ = _load_coco_index(ann_json_typ)
        if ann_json_aty and os.path.exists(ann_json_aty):
            idx_aty = _load_coco_index(ann_json_aty)
        if idx_typ is None and idx_aty is None:
            print("[WARN] --draw requested but no valid --ann_json_typical/--ann_json_atypical provided; skipping draw.")
            draw = False

    # Build plan and render
    for mp in man_paths:
        typetag = 'typical' if mp.endswith('manifest_typical.jsonl') else 'atypical'
        coco_idx = idx_typ if typetag=='typical' else idx_aty
        (images_idx, img_to_anns, id_to_name) = coco_idx if (draw and coco_idx) else (None, None, None)

        with open(mp, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                m = json.loads(line)
                a = m["a"]; b = m["b"]
                match_id = m.get("match_id", 0)
                orig_id = int(m["orig_img_id"])
                fn = _pad12(orig_id) + ".jpg"
                src = os.path.join(images_root, fn)
                if not os.path.exists(src):
                    alt = os.path.join(images_root, "train2017", fn)
                    if os.path.exists(alt): src = alt
                    else: continue

                dst_dir = os.path.join(out_dir, a, typetag, f"match_{match_id}")
                dst = os.path.join(dst_dir, os.path.basename(src))

                if draw and coco_idx is not None:
                    # draw and write into dst path (copy)
                    I = cv2.imread(src, cv2.IMREAD_COLOR)
                    if I is None:
                        continue
                    img_id = orig_id  # after normalization ids == stem
                    anns_here = img_to_anns.get(img_id, [])
                    # Draw boxes
                    for ann in anns_here:
                        cid = ann['category_id']
                        name = id_to_name.get(cid, str(cid))
                        box = ann.get('bbox', None)
                        if box is None: 
                            continue
                        # color logic
                        if name == a:
                            col = GREEN
                        elif name == b:
                            col = BLUE
                        else:
                            if not draw_all:
                                continue
                            col = GRAY
                        _draw_box(I, box, col, label=name)
                    # add typetag stripe
                    tag_col = YELLOW if typetag=='typical' else MAGENTA
                    cv2.rectangle(I, (0,0), (I.shape[1], 4), tag_col, -1)
                    _ensure_dir(dst_dir)
                    cv2.imwrite(dst, I, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    # just group
                    _link_or_copy(src, dst, symlink=symlink)

                if limit_per_pair is not None and (i+1) >= limit_per_pair:
                    # limit per manifest line is a rough cap; for precise per-pair control,
                    # pre-filter manifests or add counters keyed by (a,typetag,match_id).
                    pass

def group_real_paste(rp_dir, out_dir, symlink=False, limit_per_pair=None, draw=False, draw_all=False):
    ann_dir = os.path.join(rp_dir, "annotations")
    img_dir = os.path.join(rp_dir, "images")
    js = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    if not js:
        raise SystemExit(f"No annotation jsons under {ann_dir}")

    for j in js:
        data = json.load(open(j))
        samples = data.get("samples", [])
        for s in samples:
            pair = s.get("pair", {})
            a = pair.get("a", "anchor")
            b = pair.get("b", "partner")
            typ = pair.get("type", "typical")
            mid = pair.get("match_id", 0)
            fn = s.get("file_name")
            if not fn:
                continue
            src = os.path.join(img_dir, fn)
            if not os.path.exists(src):
                continue
            dst_dir = os.path.join(out_dir, a, typ, f"match_{mid}")
            dst = os.path.join(dst_dir, os.path.basename(src))

            if draw:
                I = cv2.imread(src, cv2.IMREAD_COLOR)
                if I is None:
                    continue
                objs = s.get("objects", [])
                for o in objs:
                    name = o.get("category", "")
                    box  = o.get("bbox", None)
                    if box is None:
                        continue
                    if name == a:
                        col = GREEN
                    elif name == b:
                        col = BLUE
                    else:
                        if not draw_all:
                            continue
                        col = GRAY
                    _draw_box(I, box, col, label=name)
                tag_col = YELLOW if typ=='typical' else MAGENTA
                cv2.rectangle(I, (0,0), (I.shape[1], 4), tag_col, -1)
                _ensure_dir(dst_dir)
                cv2.imwrite(dst, I, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                _link_or_copy(src, dst, symlink=symlink)

def group_diffusion(images_root, diffusion_manifest, out_dir, symlink=False, limit_per_pair=None, draw=False, draw_all=False):
    with open(diffusion_manifest, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("status","").upper() != "DONE":
                continue
            pair = obj.get("pair", {})
            a = pair.get("a", "anchor")
            b = pair.get("b", "partner")
            typ = pair.get("type", "typical")
            mid = pair.get("match_id", 0)
            img_rel = obj.get("image_path") or obj.get("file_name")
            if not img_rel:
                continue
            src = img_rel if os.path.isabs(img_rel) else os.path.join(images_root, img_rel)
            if not os.path.exists(src):
                continue
            dst_dir = os.path.join(out_dir, a, typ, f"match_{mid}")
            dst = os.path.join(dst_dir, os.path.basename(src))

            if draw:
                I = cv2.imread(src, cv2.IMREAD_COLOR)
                if I is None:
                    continue
                # If masks present, draw bboxes from them
                did_any = False
                for m in obj.get("masks", []):
                    name = m.get("category", "")
                    mpath = m.get("mask_path")
                    if not mpath:
                        continue
                    mpath = mpath if os.path.isabs(mpath) else os.path.join(os.path.dirname(src), "..", "masks", mpath)
                    box = _bbox_from_mask(mpath)
                    if box is None:
                        continue
                    if name == a: col = GREEN
                    elif name == b: col = BLUE
                    else:
                        if not draw_all:
                            continue
                        col = GRAY
                    _draw_box(I, box, col, label=name)
                    did_any = True
                if not did_any:
                    # fallback: just tag the image
                    pass
                tag_col = YELLOW if typ=='typical' else MAGENTA
                cv2.rectangle(I, (0,0), (I.shape[1], 4), tag_col, -1)
                _ensure_dir(dst_dir)
                cv2.imwrite(dst, I, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                _link_or_copy(src, dst, symlink=symlink)

    print(f"[OK] Grouped images under {out_dir}")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Group images by anchor into subfolders; optionally draw overlays.")
    ap.add_argument("--mode", required=True, choices=["original","real_paste","diffusion"])
    ap.add_argument("--images_root", default=None, help="For original/diffusion: directory containing images")
    ap.add_argument("--manifests_dir", default=None, help="For original: directory with manifest_typical.jsonl / manifest_atypical.jsonl")
    ap.add_argument("--ann_json_typical", default=None, help="For original+draw: typical eval json with boxes")
    ap.add_argument("--ann_json_atypical", default=None, help="For original+draw: atypical eval json with boxes")
    ap.add_argument("--rp_dir", default=None, help="For real_paste: directory with images/ and annotations/")
    ap.add_argument("--diffusion_manifest", default=None, help="For diffusion: manifest.jsonl with DONE entries")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying (saves disk)")
    ap.add_argument("--limit_per_pair", type=int, default=None, help="Optional cap per (anchor,typicality,match_id)")
    ap.add_argument("--draw", action="store_true", help="Render boxes/labels into the grouped images")
    ap.add_argument("--draw_all", action="store_true", help="When drawing, also render non-anchor/partner boxes")
    args = ap.parse_args()

    if args.mode == "original":
        if not args.images_root or not args.manifests_dir:
            raise SystemExit("--images_root and --manifests_dir are required for mode=original")
        group_original(args.images_root, args.manifests_dir, args.out_dir, symlink=args.symlink,
                       limit_per_pair=args.limit_per_pair, draw=args.draw, draw_all=args.draw_all,
                       ann_json_typ=args.ann_json_typical, ann_json_aty=args.ann_json_atypical)
    elif args.mode == "real_paste":
        if not args.rp_dir:
            raise SystemExit("--rp_dir is required for mode=real_paste")
        group_real_paste(args.rp_dir, args.out_dir, symlink=args.symlink, limit_per_pair=args.limit_per_pair,
                         draw=args.draw, draw_all=args.draw_all)
    else:
        if not args.images_root or not args.diffusion_manifest:
            raise SystemExit("--images_root and --diffusion_manifest are required for mode=diffusion")
        group_diffusion(args.images_root, args.diffusion_manifest, args.out_dir, symlink=args.symlink,
                        limit_per_pair=args.limit_per_pair, draw=args.draw, draw_all=args.draw_all)

if __name__ == "__main__":
    main()
