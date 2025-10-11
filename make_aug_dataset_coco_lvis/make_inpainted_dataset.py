#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create inpainted datasets for COCO or LVIS by removing selected instances and filling with inpainting.

Supports backends:
  - 'opencv_telea' / 'opencv_ns'                 (OpenCV fast local inpaint)
  - 'diffusers'                                  (StableDiffusionInpaintPipeline)
  - 'controlnet'                                 (StableDiffusionControlNetInpaintPipeline + lllyasviel/sd-controlnet-inpaint)
  - 'qwen_controlnet'                            (QwenImageControlNetInpaintPipeline + QwenImageControlNetModel)

Strategies:
  - 'random'       : remove k instances at random (size-limited)
  - 'sizebased'    : prefer smaller instances (p ∝ 1/area)
  - 'hardnegative' : prefer classes with lower provided weights
  - 'all'          : for each image, produce one edited image per class present,
                     removing ALL instances of that class (subject to area limit)

Paper alignment (Not Using the Car to See the Sidewalk — Quantifying & Controlling Context Effects...):
  - Remove object with GT mask then inpaint (§3.1)
  - Skip very large objects (default area <= 30% of image)
  - Dilate mask slightly (default 5 px)
  - Instance selection strategies: random / sizebased / hardnegative / all

Outputs:
  - <out_dir>/images/<split>/*.jpg               (edited images)
  - <out_dir>/annotations/<name>.json            (COCO/LVIS JSON with removed instances deleted)
  - <out_dir>/metadata.jsonl                     (one line per new image with details of removed instances)

-----------------------------------------------------------------------
Examples

COCO train2017 + Qwen ControlNet + strategy=all:
python make_inpainted_dataset.py \
  --dataset coco \
  --images-dir /data/coco/images/train2017 \
  --ann-file /data/coco/annotations/instances_train2017.json \
  --out-dir /data/out/coco_qwen_inpaint_all \
  --split-name train2017 \
  --strategy all --area-max 0.30 --dilate 5 \
  --backend qwen_controlnet \
  --qwen-base Qwen/Qwen-Image \
  --qwen-controlnet InstantX/Qwen-Image-ControlNet-Inpainting \
  --prompt "clean natural fill, seamless background, consistent lighting" \
  --neg-prompt "blurry, artifacts, distorted, watermark" \
  --sd-steps 30 --qwen-true-cfg-scale 4.0 --qwen-cond-scale 1.0 \
  --dtype bf16 --device cuda

LVIS v1.0 train + SD ControlNet (English prompt) + sizebased:
python make_inpainted_dataset.py \
  --dataset lvis \
  --images-dir /data/coco/images/train2017 \
  --ann-file /data/lvis/lvis_v1_train.json \
  --out-dir /data/out/lvis_ctl_inpaint \
  --split-name train2017 \
  --strategy sizebased --k-remove 1 --area-max 0.30 --dilate 5 \
  --backend controlnet \
  --prompt "clean natural fill, consistent textures and lighting" \
  --neg-prompt "blurry, artifacts, distorted, watermark" \
  --sd-steps 28 --sd-guidance 7.5 --dtype fp16
-----------------------------------------------------------------------
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# -----------------------------
# Optional dependencies
# -----------------------------
try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
except Exception as e:
    print("ERROR: pycocotools is required. Install: pip install pycocotools", file=sys.stderr)
    raise

try:
    from lvis import LVIS  # optional; we can still run without it
    HAVE_LVIS = True
except Exception:
    HAVE_LVIS = False


# -----------------------------
# Inpainting backend
# -----------------------------
class Inpainter:
    """
    backends:
      - 'opencv_telea' / 'opencv_ns'   : fast local inpainting
      - 'diffusers'                    : StableDiffusionInpaintPipeline
      - 'controlnet'                   : StableDiffusionControlNetInpaintPipeline (lllyasviel/sd-controlnet-inpaint)
      - 'qwen_controlnet'              : QwenImageControlNetInpaintPipeline (+ QwenImageControlNetModel)
    """

    def __init__(
        self,
        backend: str = "opencv_telea",
        sd_model_id: str = "runwayml/stable-diffusion-inpainting",
        controlnet_id: str = "lllyasviel/sd-controlnet-inpaint",
        qwen_base_id: str = "Qwen/Qwen-Image",
        qwen_controlnet_id: str = "InstantX/Qwen-Image-ControlNet-Inpainting",
        device: str = "cuda",
        dtype: str = "fp16",
        prompt: str = "clean realistic background, seamless fill, natural textures",
        negative_prompt: str = "blurry, artifacts, distorted, text, watermark, oversmooth",
        steps: int = 30,
        guidance: float = 7.5,
        seed: int = 42,
        # Qwen-specific
        qwen_true_cfg_scale: float = 4.0,
        qwen_cond_scale: float = 1.0,
        invert_mask: bool = False,  # If your mask polarity is reversed
    ):
        self.backend = backend
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.qwen_true_cfg_scale = qwen_true_cfg_scale
        self.qwen_cond_scale = qwen_cond_scale
        self.invert_mask = invert_mask
        self.device = device

        self.pipe = None
        self.ctrl = None

        if backend in ("diffusers", "controlnet", "qwen_controlnet"):
            try:
                import torch  # noqa: F401
            except Exception as e:
                raise RuntimeError("diffusers backends require PyTorch. pip install torch") from e

            import torch
            if dtype == "bf16":
                _dtype = torch.bfloat16
            elif dtype == "fp16":
                _dtype = torch.float16
            else:
                _dtype = torch.float32

            if backend == "diffusers":
                try:
                    from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
                except Exception as e:
                    raise RuntimeError("pip install diffusers") from e
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    sd_model_id, torch_dtype=_dtype
                )
                try:
                    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
                except Exception:
                    pass
                self.pipe.to(device)

            if backend == "controlnet":
                try:
                    from diffusers import (
                        StableDiffusionControlNetInpaintPipeline,
                        ControlNetModel,
                        DPMSolverMultistepScheduler,
                    )
                except Exception as e:
                    raise RuntimeError("pip install diffusers") from e
                self.ctrl = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=_dtype)
                self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    sd_model_id, controlnet=self.ctrl, torch_dtype=_dtype
                )
                try:
                    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
                except Exception:
                    pass
                self.pipe.to(device)

            if backend == "qwen_controlnet":
                try:
                    from diffusers import QwenImageControlNetModel, QwenImageControlNetInpaintPipeline
                except Exception as e:
                    raise RuntimeError("pip install diffusers (recent version)") from e
                self.ctrl = QwenImageControlNetModel.from_pretrained(
                    qwen_controlnet_id, torch_dtype=_dtype
                )
                self.pipe = QwenImageControlNetInpaintPipeline.from_pretrained(
                    qwen_base_id, controlnet=self.ctrl, torch_dtype=_dtype
                )
                self.pipe.to(device)

            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

    @staticmethod
    def _pad_to_multiple_of(img_np, multiple=8, value=0):
        h, w = img_np.shape[:2]
        nh = (h + multiple - 1) // multiple * multiple
        nw = (w + multiple - 1) // multiple * multiple
        if nh == h and nw == w:
            return img_np, (0, 0, 0, 0)
        padded = cv2.copyMakeBorder(img_np, 0, nh - h, 0, nw - w, cv2.BORDER_CONSTANT, value=value)
        return padded, (0, nh - h, 0, nw - w)

    @staticmethod
    def _unpad(img_np, pads):
        top, bottom, left, right = pads
        h, w = img_np.shape[:2]
        return img_np[top:h - bottom if bottom > 0 else h, left:w - right if right > 0 else w]

    def _run_opencv(self, img_bgr, mask_uint8, algo):
        return cv2.inpaint(
            img_bgr,
            (mask_uint8 > 0).astype(np.uint8),
            3,
            cv2.INPAINT_TELEA if algo == "opencv_telea" else cv2.INPAINT_NS,
        )

    def _run_diffusers_inpaint(self, img_bgr, mask_uint8):
        from PIL import Image

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.invert_mask:
            mask_uint8 = 255 - mask_uint8

        img_rgb_pad, pads = self._pad_to_multiple_of(img_rgb, 8, 0)
        mask_pad, _ = self._pad_to_multiple_of(mask_uint8, 8, 0)

        im = Image.fromarray(img_rgb_pad)
        m = Image.fromarray(mask_pad)
        import torch

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        out = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=im,
            mask_image=m,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=generator,
        )
        res_rgb = np.array(out.images[0])
        res_rgb = self._unpad(res_rgb, (0, pads[1], 0, pads[3]))
        return cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

    def _run_controlnet_inpaint(self, img_bgr, mask_uint8):
        from PIL import Image

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.invert_mask:
            mask_uint8 = 255 - mask_uint8

        img_rgb_pad, pads = self._pad_to_multiple_of(img_rgb, 8, 0)
        mask_pad, _ = self._pad_to_multiple_of(mask_uint8, 8, 0)

        # Control image: mark unknown region white (helps model focus on hole)
        control_rgb_pad = img_rgb_pad.copy()
        hole = (mask_pad > 0)
        control_rgb_pad[hole] = 255

        im = Image.fromarray(img_rgb_pad)
        m = Image.fromarray(mask_pad)
        c = Image.fromarray(control_rgb_pad)
        import torch

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        out = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=im,
            mask_image=m,
            control_image=c,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=generator,
        )
        res_rgb = np.array(out.images[0])
        res_rgb = self._unpad(res_rgb, (0, pads[1], 0, pads[3]))
        return cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

    def _run_qwen_controlnet_inpaint(self, img_bgr, mask_uint8):
        """Qwen Image + ControlNet inpainting backend (matches your example semantics)."""
        from PIL import Image

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.invert_mask:
            mask_uint8 = 255 - mask_uint8

        img_rgb_pad, pads = self._pad_to_multiple_of(img_rgb, 8, 0)
        mask_pad, _ = self._pad_to_multiple_of(mask_uint8, 8, 0)

        im = Image.fromarray(img_rgb_pad)
        m = Image.fromarray(mask_pad)
        import torch

        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        out = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            control_image=im,
            control_mask=m,
            controlnet_conditioning_scale=self.qwen_cond_scale,
            width=im.size[0],
            height=im.size[1],
            num_inference_steps=self.steps,
            true_cfg_scale=self.qwen_true_cfg_scale,
            generator=generator,
        )
        res_rgb = np.array(out.images[0])
        res_rgb = self._unpad(res_rgb, (0, pads[1], 0, pads[3]))
        return cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

    def inpaint(self, img_bgr: np.ndarray, mask_uint8: np.ndarray) -> np.ndarray:
        if self.backend in ("opencv_telea", "opencv_ns"):
            return self._run_opencv(img_bgr, mask_uint8, self.backend)
        elif self.backend == "diffusers":
            return self._run_diffusers_inpaint(img_bgr, mask_uint8)
        elif self.backend == "controlnet":
            return self._run_controlnet_inpaint(img_bgr, mask_uint8)
        elif self.backend == "qwen_controlnet":
            return self._run_qwen_controlnet_inpaint(img_bgr, mask_uint8)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


# -----------------------------
# Mask & selection utilities
# -----------------------------
def rle_or_poly_to_mask(seg, h, w) -> np.ndarray:
    """
    COCO/LVIS 'segmentation' can be:
      - polygon list (list[list[float]])
      - RLE dict (uncompressed or compressed)
    Returns uint8 mask {0,1} as HxW.
    """
    if isinstance(seg, list):  # polygons
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(seg, dict) and "counts" in seg:
        rle = seg
    else:
        # Fallback
        rle = maskUtils.frPyObjects(seg, h, w)
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def _sanitize_for_name(s: str) -> str:
    """Make a safe short token for filenames."""
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_.-]", "", s)
    return s or "unk"


def _instance_area(ann: Dict[str, Any], H: int, W: int) -> float:
    """Return instance area (compute from mask if missing)."""
    area = ann.get("area", None)
    if area is None or area <= 0:
        seg = ann.get("segmentation", None)
        if seg is None:
            return 0.0
        m = rle_or_poly_to_mask(seg, H, W)
        return float(m.sum())
    return float(area)


def pick_instances(
    anns: List[Dict[str, Any]],
    H: int,
    W: int,
    strategy: str,
    k_remove: int,
    area_max_ratio: float,
    per_class_weight: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Filter and sample instance annotations to remove.
    - Reject instances larger than area_max_ratio of the image.
    - strategy:
        random      : uniform over candidates
        sizebased   : sample prob ∝ 1/area (prefer small)
        hardnegative: sample prob ∝ 1/(eps + class_weight[cat]) if provided
    """
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


def build_mask_for_instances(anns: List[Dict[str, Any]], H: int, W: int, dilate_px: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    for a in anns:
        seg = a.get("segmentation", None)
        if seg is None:
            continue
        m = rle_or_poly_to_mask(seg, H, W)
        mask[m > 0] = 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


# -----------------------------
# Robust image path resolution
# -----------------------------
def resolve_image_path(base_dir: Path, file_name: str) -> Optional[Path]:
    """
    Try several common layouts:
      - base_dir / file_name
      - base_dir / basename(file_name)
      - base_dir.parent / file_name (if base_dir already points to split dir)
    """
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
# Main pipeline
# -----------------------------
def process_dataset(args):
    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)

    images_dir = Path(args.images_dir)
    out_images_dir = Path(args.out_dir) / "images" / args.split_name
    out_anns_dir = Path(args.out_dir) / "annotations"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_anns_dir.mkdir(parents=True, exist_ok=True)

    # Load JSON
    with open(args.ann_file, "r") as f:
        ann_json = json.load(f)

    is_coco = args.dataset.lower() == "coco"
    is_lvis = args.dataset.lower() == "lvis"

    # build API / helper
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
            # Fallback: manual grouping
            imgid_to_anns = {}
            for a in ann_json.get("annotations", []):
                imgid_to_anns.setdefault(a["image_id"], []).append(a)
            get_img_anns = lambda img_id: imgid_to_anns.get(img_id, [])
        categories = list(ann_json.get("categories", []))
        cat_name_from_id = {c["id"]: c.get("name", str(c["id"])) for c in categories}
    else:
        raise ValueError("--dataset must be 'coco' or 'lvis'")

    # per-class weights (for 'hardnegative' strategy)
    per_class_weight = None
    if args.class_weights and Path(args.class_weights).exists():
        with open(args.class_weights, "r") as f:
            per_class_weight = {int(k): float(v) for k, v in json.load(f).items()}

    # Build inpainter
    inpainter = Inpainter(
        backend=args.backend,
        sd_model_id=args.sd_model,
        controlnet_id=args.controlnet_id,
        qwen_base_id=args.qwen_base,
        qwen_controlnet_id=args.qwen_controlnet,
        device=args.device,
        dtype=args.dtype,
        prompt=args.prompt,
        negative_prompt=args.neg_prompt,
        steps=args.sd_steps,
        guidance=args.sd_guidance,
        seed=args.sd_seed,
        qwen_true_cfg_scale=args.qwen_true_cfg_scale,
        qwen_cond_scale=args.qwen_cond_scale,
        invert_mask=args.invert_mask,
    )

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

    # Metadata log
    meta_fp = open(Path(args.out_dir) / "metadata.jsonl", "w", encoding="utf-8")

    count_processed = 0
    for im in ann_json.get("images", []):
        if rng.rand() > args.process_prob:
            continue

        src_path = resolve_image_path(images_dir, im["file_name"])
        if src_path is None or not src_path.exists():
            continue

        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        img_area = H * W

        anns = get_img_anns(im["id"])
        if not anns:
            continue

        # ---------- strategy 'all' ----------
        if args.strategy == "all":
            # Collect classes present (excluding crowd)
            cat_ids_present = []
            seen = set()
            for a in anns:
                if a.get("iscrowd", 0) == 1:
                    continue
                cid = int(a["category_id"])
                if cid not in seen:
                    seen.add(cid)
                    cat_ids_present.append(cid)

            # For each class present, remove ALL its instances (subject to area_max)
            for cid in cat_ids_present:
                class_insts = [a for a in anns if a.get("iscrowd", 0) != 1 and int(a["category_id"]) == cid]
                if not class_insts:
                    continue

                # Strict area check: every instance of this class must satisfy area_max
                too_big = False
                for a in class_insts:
                    area = _instance_area(a, H, W)
                    if (area / img_area) > args.area_max:
                        too_big = True
                        break
                if too_big:
                    # Skip this class variant to avoid overly large holes
                    continue

                # Build mask from ALL instances of this class
                mask = build_mask_for_instances(class_insts, H, W, args.dilate)
                if mask.max() == 0:
                    continue

                # Inpaint and save
                img_inp = inpainter.inpaint(img, mask)

                base = Path(im["file_name"]).name
                stem, ext = os.path.splitext(base)
                cat_name = cat_name_from_id.get(cid, str(cid))
                cat_tok = _sanitize_for_name(cat_name)
                new_name = f"{stem}_rmALL_{cat_tok}{ext or '.jpg'}"
                out_path = out_images_dir / new_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(out_path), img_inp)
                if not ok:
                    continue

                # New image entry
                new_img_id = next(img_id_gen)
                new_im = dict(im)
                new_im["id"] = new_img_id
                new_im["file_name"] = str(Path("images") / args.split_name / new_name)
                new_images.append(new_im)

                # Copy annotations except those of this class
                removed_ids = set(a["id"] for a in class_insts)
                for a in anns:
                    if a["id"] in removed_ids:
                        continue
                    new_a = dict(a)
                    new_a["id"] = next(ann_id_gen)
                    new_a["image_id"] = new_img_id
                    new_annotations.append(new_a)

                # Metadata
                removed = [
                    {
                        "ann_id": a["id"],
                        "category_id": a["category_id"],
                        "category_name": cat_name_from_id.get(a["category_id"], str(a["category_id"])),
                        "area": float(a.get("area", _instance_area(a, H, W))),
                        "bbox": a.get("bbox", None),
                    }
                    for a in class_insts
                ]
                meta = {
                    "orig_image_id": im["id"],
                    "new_image_id": new_img_id,
                    "orig_file": im["file_name"],
                    "new_file": new_im["file_name"],
                    "removed": removed,
                    "dilate_px": args.dilate,
                    "backend": args.backend,
                    "strategy": "all",
                    "removed_category_id": cid,
                    "removed_category_name": cat_name,
                    "area_max": args.area_max,
                }
                meta_fp.write(json.dumps(meta, ensure_ascii=False) + "\n")

                count_processed += 1
                if args.max_images and count_processed >= args.max_images:
                    break

            if args.max_images and count_processed >= args.max_images:
                break

            # Done with this source image
            continue
        # ---------- /strategy 'all' ----------

        # Single-variant path (random/sizebased/hardnegative)
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

        # build mask and inpaint
        mask = build_mask_for_instances(selected, H, W, args.dilate)
        img_inp = inpainter.inpaint(img, mask)

        # write image
        base = Path(im["file_name"]).name
        stem, ext = os.path.splitext(base)
        new_name = f"{stem}_rm{len(selected)}_{args.strategy}{ext or '.jpg'}"
        out_path = out_images_dir / new_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), img_inp)
        if not ok:
            continue

        # add image entry (new ID, path relative to dataset root like COCO)
        new_img_id = next(img_id_gen)
        new_im = dict(im)
        new_im["id"] = new_img_id
        new_im["file_name"] = str(Path("images") / args.split_name / new_name)
        new_images.append(new_im)

        # copy annotations except removed
        removed_ids = set(a["id"] for a in selected)
        for a in anns:
            if a["id"] in removed_ids:
                continue
            new_a = dict(a)
            new_a["id"] = next(ann_id_gen)
            new_a["image_id"] = new_img_id
            new_annotations.append(new_a)

        # metadata
        removed = [
            {
                "ann_id": a["id"],
                "category_id": a["category_id"],
                "category_name": cat_name_from_id.get(a["category_id"], str(a["category_id"])),
                "area": float(a.get("area", _instance_area(a, H, W))),
                "bbox": a.get("bbox", None),
            }
            for a in selected
        ]
        meta = {
            "orig_image_id": im["id"],
            "new_image_id": new_img_id,
            "orig_file": im["file_name"],
            "new_file": new_im["file_name"],
            "removed": removed,
            "dilate_px": args.dilate,
            "backend": args.backend,
            "strategy": args.strategy,
            "area_max": args.area_max,
        }
        meta_fp.write(json.dumps(meta, ensure_ascii=False) + "\n")

        count_processed += 1
        if args.max_images and count_processed >= args.max_images:
            break

    meta_fp.close()

    # write JSON
    out_json_name = ("instances" if is_coco else "lvis") + f"_{args.split_name}_inpainted.json"
    with open(out_anns_dir / out_json_name, "w", encoding="utf-8") as f:
        json.dump(new_json, f)

    print(f"Done. Edited {count_processed} images.")
    print(f"Images     -> {out_images_dir}")
    print(f"Annotations-> {out_anns_dir / out_json_name}")
    print(f"Metadata   -> {Path(args.out_dir) / 'metadata.jsonl'}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Create inpainted COCO/LVIS datasets by removing instances and inpainting the holes."
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
    p.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="JSON mapping {category_id: weight} used by 'hardnegative' (lower weight -> more likely to remove).",
    )

    # Backend & models
    p.add_argument(
        "--backend",
        choices=["opencv_telea", "opencv_ns", "diffusers", "controlnet", "qwen_controlnet"],
        default="opencv_telea",
    )
    p.add_argument("--sd-model", type=str, default="runwayml/stable-diffusion-inpainting")
    p.add_argument("--controlnet-id", type=str, default="lllyasviel/sd-controlnet-inpaint")
    p.add_argument("--qwen-base", type=str, default="Qwen/Qwen-Image")
    p.add_argument("--qwen-controlnet", type=str, default="InstantX/Qwen-Image-ControlNet-Inpainting")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")

    # Text guidance / sampling for generative backends
    p.add_argument(
        "--prompt",
        type=str,
        default="clean natural fill, seamless background, consistent lighting, realistic textures",
    )
    p.add_argument(
        "--neg-prompt",
        type=str,
        default="blurry, artifacts, distorted, watermark, text, lowres, oversmooth",
    )
    p.add_argument("--sd-steps", type=int, default=30)
    p.add_argument("--sd-guidance", type=float, default=7.5)
    p.add_argument("--sd-seed", type=int, default=42)
    p.add_argument("--qwen-true-cfg-scale", type=float, default=4.0)
    p.add_argument("--qwen-cond-scale", type=float, default=1.0)
    p.add_argument(
        "--invert-mask",
        action="store_true",
        help="Invert mask polarity before inpainting if your mask semantics are reversed.",
    )

    # Which images to process
    p.add_argument("--process-prob", type=float, default=1.0, help="Probability to edit a given image.")
    p.add_argument("--max-images", type=int, default=0, help="Stop after this many edited images (0 = no limit).")
    p.add_argument("--seed", type=int, default=123)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)

