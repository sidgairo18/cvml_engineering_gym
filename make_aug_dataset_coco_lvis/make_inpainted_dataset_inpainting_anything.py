#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create inpainted datasets for COCO or LVIS using:
  - SAM (optional) to re-segment an instance from a click
  - LaMa to inpaint the removed region

Backends here: 'sam_lama' (LaMa inpainting, optional SAM refinement).
Strategies:
  - random / sizebased / hardnegative : remove k instances per image
  - all : one output per class in the image (remove ALL instances of that class)

Outputs:
  - <out_dir>/images/<split>/*.jpg
  - <out_dir>/annotations/<split>_inpainted.json
  - <out_dir>/metadata.jsonl
Optional previews (if --save-preview):
  - <out_dir>/previews/<split>/*_preview.jpg   (Original | Mask overlay | Inpainted)

Requirements (from the "Inpainting Anything" stack):
  - segment_anything  (pip -e)
  - LaMa deps         (pip -r lama/requirements.txt)
  - Pretrained weights:
      * SAM ckpt (e.g., ./pretrained_models/sam_vit_h_4b8939.pth)
      * LaMa big-lama directory (e.g., ./pretrained_models/big-lama)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# -----------------------------
# COCO / LVIS deps
# -----------------------------
try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
except Exception as e:
    print("ERROR: pycocotools is required. pip install pycocotools", file=sys.stderr)
    raise

try:
    from lvis import LVIS
    HAVE_LVIS = True
except Exception:
    HAVE_LVIS = False

# -----------------------------
# Optional: SAM + LaMa helper imports
# -----------------------------
_SAM_OK = True
_LAMA_OK = True
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception:
    _SAM_OK = False

try:
    # inpainting-anything LaMa wrapper:
    #   inpaint_img_with_lama(img_rgb, mask_uint8(0/255), lama_config, lama_ckpt, device) -> np.uint8 RGB
    from lama_inpaint import inpaint_img_with_lama
except Exception:
    _LAMA_OK = False

print("_SAM_OK", _SAM_OK, "_LAMA_OK", _LAMA_OK)


# -----------------------------
# Utilities
# -----------------------------
def rle_or_poly_to_mask(seg, h, w) -> np.ndarray:
    if isinstance(seg, list):  # polygons
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(seg, dict) and "counts" in seg:
        rle = seg
    else:
        rle = maskUtils.frPyObjects(seg, h, w)
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def instance_centroid(mask_uint8: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask_uint8 > 0)
    if len(xs) == 0:
        return -1, -1
    cx = int(xs.mean())
    cy = int(ys.mean())
    return cx, cy


def pick_instances(
    anns: List[Dict[str, Any]],
    H: int,
    W: int,
    strategy: str,
    k_remove: int,
    area_max_ratio: float,
    per_class_weight: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    candidates = []
    img_area = H * W
    for a in anns:
        if a.get("iscrowd", 0) == 1:
            continue
        area = a.get("area", None)
        if not area or area <= 0:
            seg = a.get("segmentation", None)
            if seg is None:
                continue
            m = rle_or_poly_to_mask(seg, H, W)
            area = float(m.sum())
        if area / img_area <= area_max_ratio:
            candidates.append((a, area))

    if not candidates:
        return []

    if strategy == "random":
        random.shuffle(candidates)
        return [x[0] for x in candidates[:k_remove]]

    weights = []
    if strategy == "sizebased":
        weights = [1.0 / max(1.0, area) for _, area in candidates]
    elif strategy == "hardnegative":
        eps = 1e-6
        for a, _ in candidates:
            w = 1.0
            if per_class_weight and a["category_id"] in per_class_weight:
                w = 1.0 / (eps + float(per_class_weight[a["category_id"]]))
            weights.append(w)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    s = float(sum(weights))
    if s <= 0:
        random.shuffle(candidates)
        return [x[0] for x in candidates[:k_remove]]

    probs = [w / s for w in weights]
    idxs = np.random.choice(len(candidates), size=min(k_remove, len(candidates)), replace=False, p=probs)
    return [candidates[i][0] for i in idxs]


def build_mask_union(anns: List[Dict[str, Any]], H: int, W: int, dilate_px: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    for a in anns:
        seg = a.get("segmentation", None)
        if seg is None:
            continue
        m = rle_or_poly_to_mask(seg, H, W)
        mask[m > 0] = 255
    if dilate_px > 0 and mask.max() > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


def resolve_image_path(base_dir: Path, file_name: str) -> Optional[Path]:
    p1 = base_dir / file_name
    if p1.exists():
        return p1
    p2 = base_dir / os.path.basename(file_name)
    if p2.exists():
        return p2
    p3 = base_dir.parent / file_name
    if p3.exists():
        return p3
    return None


# -----------------------------
# SAM wrapper (optional)
# -----------------------------
class SamRefiner:
    def __init__(self, model_type: str, ckpt_path: str, device: str = "cuda"):
        if not _SAM_OK:
            raise RuntimeError("segment_anything is not installed or import failed. Install the SAM repo.")
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {ckpt_path}")
        sam = sam_model_registry[model_type](checkpoint=ckpt_path)
        self.device = device
        self.predictor = SamPredictor(sam.to(device))

    def refine_mask(self, img_rgb: np.ndarray, click_xy: Tuple[int, int]) -> np.ndarray:
        """Return a binary mask (uint8 {0,255}) from a single positive click."""
        x, y = click_xy
        if x < 0 or y < 0:
            return np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        self.predictor.set_image(img_rgb)
        point_coords = np.array([[x, y]])
        point_labels = np.array([1], dtype=np.int32)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        idx = int(np.argmax(scores))
        m = masks[idx].astype(np.uint8) * 255
        return m


# -----------------------------
# Inpainting (LaMa)
# -----------------------------
def lama_inpaint(img_bgr: np.ndarray, mask_uint8: np.ndarray, lama_config: str, lama_ckpt: str, device: str) -> np.ndarray:
    if not _LAMA_OK:
        raise RuntimeError("LaMa wrapper not found. Ensure `lama_inpaint.inpaint_img_with_lama` is importable.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_bin = (mask_uint8 > 0).astype(np.uint8) * 255
    out_rgb = inpaint_img_with_lama(img_rgb, mask_bin, lama_config, lama_ckpt, device=device)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr


# -----------------------------
# Preview helpers
# -----------------------------
def color_overlay(img_bgr: np.ndarray, mask_uint8: np.ndarray, color_bgr=(0, 0, 255), alpha: float = 0.55) -> np.ndarray:
    """Blend a solid color over masked pixels."""
    ov = img_bgr.copy().astype(np.float32)
    col = np.array(color_bgr, dtype=np.float32)
    m = (mask_uint8 > 0)
    if m.any():
        ov[m] = (1.0 - alpha) * ov[m] + alpha * col
    return ov.clip(0, 255).astype(np.uint8)


def hstack_with_pad(imgs: List[np.ndarray], pad: int = 6, pad_color=(255, 255, 255)) -> np.ndarray:
    """Horizontally stack images with vertical separators."""
    if not imgs:
        return None
    h = min(i.shape[0] for i in imgs)
    # Resize to same height (keep aspect)
    resized = []
    for im in imgs:
        if im.shape[0] != h:
            w = int(round(im.shape[1] * (h / im.shape[0])))
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        resized.append(im)
    if pad > 0:
        sep = np.full((h, pad, 3), pad_color, dtype=np.uint8)
        rows = []
        for k, im in enumerate(resized):
            rows.append(im)
            if k != len(resized) - 1:
                rows.append(sep)
        out = np.hstack(rows)
    else:
        out = np.hstack(resized)
    return out


def maybe_downscale_max_edge(img: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_edge:
        return img
    scale = max_edge / m
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def save_preview_triptych(
    preview_dir: Path,
    base_name_stem: str,
    img_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    inpaint_bgr: np.ndarray,
    alpha: float,
    max_edge: int,
):
    preview_dir.mkdir(parents=True, exist_ok=True)
    overlay = color_overlay(img_bgr, mask_uint8, color_bgr=(0, 0, 255), alpha=alpha)
    trip = hstack_with_pad([img_bgr, overlay, inpaint_bgr], pad=6, pad_color=(255, 255, 255))
    trip = maybe_downscale_max_edge(trip, max_edge)
    out_path = preview_dir / f"{base_name_stem}_preview.jpg"
    cv2.imwrite(str(out_path), trip)


# -----------------------------
# Main pipeline
# -----------------------------
def process_dataset(args):
    import torch
    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.sam_refine and not _SAM_OK:
        raise RuntimeError("You passed --sam-refine but segment_anything isn't available. Install the SAM repo.")
    if not _LAMA_OK:
        raise RuntimeError("LaMa wrapper not importable. Ensure LaMa deps are installed and paths are correct.")

    images_dir = Path(args.images_dir)
    out_images_dir = Path(args.out_dir) / "images" / args.split_name
    out_anns_dir = Path(args.out_dir) / "annotations"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_anns_dir.mkdir(parents=True, exist_ok=True)

    # Preview dir
    preview_dir = Path(args.preview_dir) if args.preview_dir else (Path(args.out_dir) / "previews" / args.split_name)
    if args.save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # Print args
    #print("========== make_inpainted_dataset_inpainting_anything.py :: arguments ==========")
    #print(json.dumps(vars(args), indent=2, sort_keys=True))
    #print("================================================================================")
    #sys.stdout.flush()

    meta_fp = open(Path(args.out_dir) / "metadata.jsonl", "w", encoding="utf-8")

    # Load JSON
    with open(args.ann_file, "r") as f:
        ann_json = json.load(f)

    is_coco = args.dataset.lower() == "coco"
    is_lvis = args.dataset.lower() == "lvis"

    # API / helpers
    if is_coco:
        api = COCO(args.ann_file)
        get_img_anns = lambda img_id: api.loadAnns(api.getAnnIds(imgIds=[img_id], iscrowd=None))
        categories = list(ann_json.get("categories", []))
        cat_name_from_id = {c["id"]: c["name"] for c in categories}
    elif is_lvis:
        if HAVE_LVIS:
            api = LVIS(args.ann_file)
            get_img_anns = lambda img_id: api.load_anns(api.get_ann_ids(img_ids=[img_id]))
        else:
            imgid_to_anns = {}
            for a in ann_json.get("annotations", []):
                imgid_to_anns.setdefault(a["image_id"], []).append(a)
            get_img_anns = lambda img_id: imgid_to_anns.get(img_id, [])
        categories = list(ann_json.get("categories", []))
        cat_name_from_id = {c["id"]: c.get("name", str(c["id"])) for c in categories}
    else:
        meta_fp.close()
        raise ValueError("--dataset must be 'coco' or 'lvis'")

    per_class_weight = None
    if args.class_weights and Path(args.class_weights).exists():
        with open(args.class_weights, "r") as f:
            per_class_weight = {int(k): float(v) for k, v in json.load(f).items()}

    # Optional SAM refiner
    sam_refiner = None
    if args.sam_refine:
        sam_refiner = SamRefiner(args.sam_model_type, args.sam_ckpt, args.device)

    # New JSON shell
    new_images: List[Dict[str, Any]] = []
    new_annotations: List[Dict[str, Any]] = []
    new_json = {
        "info": ann_json.get("info", {}),
        "licenses": ann_json.get("licenses", []),
        "categories": categories,
        "images": new_images,
        "annotations": new_annotations,
    }

    # ID generators
    def next_ids(start: int):
        i = start
        while True:
            yield i
            i += 1

    start_ann_id = max([a["id"] for a in ann_json.get("annotations", [])], default=0) + 1
    start_img_id = max([img["id"] for img in ann_json.get("images", [])], default=0) + 1
    ann_id_gen = next_ids(start_ann_id)
    img_id_gen = next_ids(start_img_id)

    count_processed = 0

    for im in ann_json.get("images", []):
        print("Processing", count_processed)
        if rng.rand() > args.process_prob:
            continue

        src_path = resolve_image_path(images_dir, im["file_name"])
        if src_path is None or not src_path.exists():
            continue

        img_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        img_area = H * W

        anns = get_img_anns(im["id"])
        if not anns:
            continue

        # ---- Strategy: 'all' -> one edited image per class present ----
        if args.strategy == "all":
            seen = set()
            cat_ids = []
            for a in anns:
                if a.get("iscrowd", 0) == 1:
                    continue
                cid = int(a["category_id"])
                if cid not in seen:
                    seen.add(cid)
                    cat_ids.append(cid)

            for cid in cat_ids:
                class_insts = [a for a in anns if a.get("iscrowd", 0) != 1 and int(a["category_id"]) == cid]
                if not class_insts:
                    continue
                # Skip if any instance too large
                too_big = False
                for a in class_insts:
                    area = a.get("area", None)
                    if area is None or area <= 0:
                        m = rle_or_poly_to_mask(a.get("segmentation"), H, W)
                        area = float(m.sum())
                    if area / img_area > args.area_max:
                        too_big = True
                        break
                if too_big:
                    continue

                if args.sam_refine and sam_refiner is not None:
                    full_mask = np.zeros((H, W), dtype=np.uint8)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    for a in class_insts:
                        m_gt = rle_or_poly_to_mask(a.get("segmentation"), H, W) * 255
                        cx, cy = instance_centroid(m_gt)
                        if cx < 0:
                            continue
                        m_sam = sam_refiner.refine_mask(img_rgb, (cx, cy))
                        full_mask[m_sam > 0] = 255
                    mask_union = full_mask
                else:
                    mask_union = build_mask_union(class_insts, H, W, args.dilate)

                if mask_union.max() == 0:
                    continue

                try:
                    img_inp = lama_inpaint(img_bgr, mask_union, args.lama_config, args.lama_ckpt, args.device)
                except Exception as e:
                    print(f"[WARN] LaMa inpaint failed (class {cid}): {e}")
                    continue

                # Save image
                base = Path(im["file_name"]).name
                stem, ext = os.path.splitext(base)
                new_name = f"{stem}_rmALL_{cid}{ext or '.jpg'}"
                out_path = Path(args.out_dir) / "images" / args.split_name / new_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(out_path), img_inp):
                    continue

                # Save preview if requested
                if args.save_preview:
                    save_preview_triptych(
                        preview_dir=preview_dir,
                        base_name_stem=f"{stem}_rmALL_{cid}",
                        img_bgr=img_bgr,
                        mask_uint8=mask_union,
                        inpaint_bgr=img_inp,
                        alpha=args.preview_alpha,
                        max_edge=args.preview_max_edge,
                    )

                # JSON + meta
                new_img_id = next(img_id_gen)
                new_im = dict(im)
                new_im["id"] = new_img_id
                new_im["file_name"] = str(Path("images") / args.split_name / new_name)
                new_images.append(new_im)

                removed_ids = {a["id"] for a in class_insts}
                for a in anns:
                    if a["id"] in removed_ids:
                        continue
                    new_a = dict(a)
                    new_a["id"] = next(ann_id_gen)
                    new_a["image_id"] = new_img_id
                    new_annotations.append(new_a)

                meta = {
                    "orig_image_id": im["id"],
                    "new_image_id": new_img_id,
                    "orig_file": im["file_name"],
                    "new_file": new_im["file_name"],
                    "removed": [
                        {
                            "ann_id": a["id"],
                            "category_id": a["category_id"],
                            "category_name": cat_name_from_id.get(a["category_id"], str(a["category_id"])),
                            "area": float(a.get("area", 0.0)),
                            "bbox": a.get("bbox", None),
                        } for a in class_insts
                    ],
                    "strategy": "all",
                    "area_max": args.area_max,
                    "dilate_px": args.dilate,
                    "backend": "sam_lama",
                    "sam_refine": bool(args.sam_refine),
                }
                meta_fp.write(json.dumps(meta, ensure_ascii=False) + "\n")

                count_processed += 1
                if args.max_images and count_processed >= args.max_images:
                    break

            if args.max_images and count_processed >= args.max_images:
                break
            continue
        # ---- /all ----

        # ---- Single-variant strategies ----
        selected = pick_instances(
            anns,
            H,
            W,
            strategy=args.strategy,
            k_remove=args.k_remove,
            area_max_ratio=args.area_max,
            per_class_weight=per_class_weight,
        )
        if not selected:
            continue

        if args.sam_refine and sam_refiner is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask_union = np.zeros((H, W), dtype=np.uint8)
            for a in selected:
                m_gt = rle_or_poly_to_mask(a.get("segmentation"), H, W) * 255
                cx, cy = instance_centroid(m_gt)
                if cx < 0:
                    continue
                m_sam = sam_refiner.refine_mask(img_rgb, (cx, cy))
                if args.dilate > 0 and m_sam.max() > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.dilate + 1, 2 * args.dilate + 1))
                    m_sam = cv2.dilate(m_sam, k, iterations=1)
                mask_union[m_sam > 0] = 255
        else:
            mask_union = build_mask_union(selected, H, W, args.dilate)

        if mask_union.max() == 0:
            continue

        try:
            img_inp = lama_inpaint(img_bgr, mask_union, args.lama_config, args.lama_ckpt, args.device)
        except Exception as e:
            print(f"[WARN] LaMa inpaint failed: {e}")
            continue

        # Save image
        base = Path(im["file_name"]).name
        stem, ext = os.path.splitext(base)
        new_name = f"{stem}_rm{len(selected)}_{args.strategy}{ext or '.jpg'}"
        out_path = Path(args.out_dir) / "images" / args.split_name / new_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), img_inp):
            continue

        # Save preview if requested
        if args.save_preview:
            save_preview_triptych(
                preview_dir=preview_dir,
                base_name_stem=f"{stem}_rm{len(selected)}_{args.strategy}",
                img_bgr=img_bgr,
                mask_uint8=mask_union,
                inpaint_bgr=img_inp,
                alpha=args.preview_alpha,
                max_edge=args.preview_max_edge,
            )

        # JSON + meta
        new_img_id = next(img_id_gen)
        new_im = dict(im)
        new_im["id"] = new_img_id
        new_im["file_name"] = str(Path("images") / args.split_name / new_name)
        new_images.append(new_im)

        removed_ids = {a["id"] for a in selected}
        for a in anns:
            if a["id"] in removed_ids:
                continue
            new_a = dict(a)
            new_a["id"] = next(ann_id_gen)
            new_a["image_id"] = new_img_id
            new_annotations.append(new_a)

        meta = {
            "orig_image_id": im["id"],
            "new_image_id": new_img_id,
            "orig_file": im["file_name"],
            "new_file": new_im["file_name"],
            "removed": [
                {
                    "ann_id": a["id"],
                    "category_id": a["category_id"],
                    "category_name": cat_name_from_id.get(a["category_id"], str(a["category_id"])),
                    "area": float(a.get("area", 0.0)),
                    "bbox": a.get("bbox", None),
                } for a in selected
            ],
            "strategy": args.strategy,
            "area_max": args.area_max,
            "dilate_px": args.dilate,
            "backend": "sam_lama",
            "sam_refine": bool(args.sam_refine),
        }
        meta_fp.write(json.dumps(meta, ensure_ascii=False) + "\n")

        count_processed += 1
        if args.max_images and count_processed >= args.max_images:
            break
        # ---- /single-variant ----

    meta_fp.close()

    out_json_name = ("instances" if is_coco else "lvis") + f"_{args.split_name}_inpainted.json"
    with open(Path(args.out_dir) / "annotations" / out_json_name, "w", encoding="utf-8") as f:
        json.dump(new_json, f)

    print(f"Done. Edited {count_processed} images.")
    print(f"Images     -> {Path(args.out_dir) / 'images' / args.split_name}")
    print(f"Annotations-> {Path(args.out_dir) / 'annotations' / out_json_name}")
    print(f"Metadata   -> {Path(args.out_dir) / 'metadata.jsonl'}")
    if args.save_preview:
        print(f"Previews   -> {preview_dir}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Create inpainted COCO/LVIS datasets using SAM (optional) + LaMa (Inpainting Anything)."
    )
    p.add_argument("--dataset", choices=["coco", "lvis"], required=True)
    p.add_argument("--images-dir", required=True, help="Directory containing images for the split.")
    p.add_argument("--ann-file", required=True, help="Path to COCO/LVIS annotation JSON.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--split-name", required=True, help="Split name used in output path (e.g., train2017).")

    # Removal & sampling
    p.add_argument("--strategy", choices=["random", "sizebased", "hardnegative", "all"], default="sizebased")
    p.add_argument("--k-remove", type=int, default=1, help="Number of instances to remove per edited image (ignored for 'all').")
    p.add_argument("--area-max", type=float, default=0.30, help="Reject instances larger than this image-area fraction.")
    p.add_argument("--dilate", type=int, default=5, help="Mask dilation in pixels.")
    p.add_argument("--class-weights", type=str, default=None, help="JSON {category_id: weight} for 'hardnegative'.")

    # SAM (optional)
    p.add_argument("--sam-refine", action="store_true", help="Use SAM to re-segment each selected instance from one click (centroid).")
    p.add_argument("--sam-model-type", type=str, default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'])
    p.add_argument("--sam-ckpt", type=str, default="./pretrained_models/sam_vit_h_4b8939.pth")

    # LaMa (required)
    p.add_argument("--lama-config", type=str, default="./lama/configs/prediction/default.yaml")
    p.add_argument("--lama-ckpt", type=str, default="./pretrained_models/big-lama")

    # Device
    p.add_argument("--device", type=str, default="cuda")

    # Which images to process
    p.add_argument("--process-prob", type=float, default=1.0, help="Probability to edit a given image.")
    p.add_argument("--max-images", type=int, default=0, help="Stop after this many edited images (0 = no limit).")
    p.add_argument("--seed", type=int, default=123)

    # Previews
    p.add_argument("--save-preview", action="store_true", help="Save side-by-side preview (orig | mask overlay | inpainted).")
    p.add_argument("--preview-dir", type=str, default=None, help="Directory to save previews (default: <out>/previews/<split>).")
    p.add_argument("--preview-alpha", type=float, default=0.55, help="Alpha for mask overlay in preview.")
    p.add_argument("--preview-max-edge", type=int, default=2000, help="Downscale preview if max(H,W) exceeds this. 0 disables.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        print("========== make_inpainted_dataset_inpainting_anything.py :: arguments ==========")
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        print("================================================================================")
        sys.stdout.flush()
    except Exception:
        print(args)
    process_dataset(args)

