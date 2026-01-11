#!/usr/bin/env python3
"""
Evaluate LLaVA-OneVision as a COCO-80 multi-label (multi-object presence) classifier.

Modes:
  - yesno : per-class scoring using next-token logits (A/B). Produces AP/mAP + P/R/F1.
  - list  : single prompt returns a set of labels. Produces P/R/F1 (no meaningful AP/mAP).
  - hybrid: list -> candidate labels, then yesno scoring only for candidates + sampled negatives.
            Produces P/R/F1 + approx mAP (zero-fill unscored labels => lower bound).

COCO folder layout expected:
  coco_root/
    train2017/
    val2017/
    annotations/
      instances_train2017.json
      instances_val2017.json
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from pycocotools.coco import COCO
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


# ---------- COCO 80 classes (official names used by COCO) ----------
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]


# ---------- Helpers ----------
def set_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def coco_categories_in_annotation(coco: COCO) -> Tuple[List[int], List[str], Dict[int, int]]:
    """
    COCO category IDs are not contiguous; build a stable mapping.
    """
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cats_sorted = sorted(cats, key=lambda x: x["id"])
    cat_ids_sorted = [c["id"] for c in cats_sorted]
    cat_names_sorted = [c["name"] for c in cats_sorted]
    catid_to_index = {cid: i for i, cid in enumerate(cat_ids_sorted)}
    return cat_ids_sorted, cat_names_sorted, catid_to_index


def extract_json_object(text: str) -> Optional[dict]:
    """
    Heuristic JSON extraction: find first '{' and last '}' and parse.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start:end + 1].strip()
    snippet = re.sub(r"^```(json)?\s*", "", snippet)
    snippet = re.sub(r"\s*```$", "", snippet)
    try:
        return json.loads(snippet)
    except Exception:
        return None


def normalize_label(s: str) -> str:
    return s.strip().lower()


# ---------- Prompt builders ----------
def build_yesno_prompt(processor: AutoProcessor, label: str) -> str:
    """
    Use A/B to encourage single-token answers:
      A = Yes, B = No
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    "You are answering questions about whether COCO object categories appear in the image.\n"
                    f"Question: Is there a '{label}' in this image (in the COCO sense)?\n"
                    "Answer with only one character:\n"
                    "A = Yes\n"
                    "B = No"
                )},
            ],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def build_list_prompt(processor: AutoProcessor, allowed_labels: List[str]) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    "Identify which COCO-80 classes are present in the image.\n"
                    "Only use labels from this list exactly (no synonyms, no extra words):\n"
                    + ", ".join(allowed_labels) +
                    "\n\nReturn ONLY a JSON object exactly like:\n"
                    "{\"labels\": [\"dog\", \"person\"]}\n"
                    "If none are present, return: {\"labels\": []}"
                )},
            ],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


# ---------- Inference ----------
@torch.no_grad()
def score_labels_yesno_ab(
    model: LlavaOnevisionForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    labels: List[str],
    device: str,
    chunk_size: int,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    """
    Returns score per label = P(A|prompt,image) where A=Yes, B=No using next-token logits.
    """
    tok = processor.tokenizer

    a_ids = tok.encode(" A", add_special_tokens=False)
    b_ids = tok.encode(" B", add_special_tokens=False)
    if len(a_ids) != 1 or len(b_ids) != 1:
        raise RuntimeError(
            "Tokenizer did not map ' A'/' B' to single tokens. "
            "Try changing scoring tokens (e.g. '1'/'0') or inspect tokenizer behavior."
        )
    a_id, b_id = a_ids[0], b_ids[0]

    scores = np.zeros((len(labels),), dtype=np.float32)

    for start in range(0, len(labels), chunk_size):
        chunk = labels[start:start + chunk_size]
        prompts = [build_yesno_prompt(processor, lab) for lab in chunk]
        images = [image] * len(chunk)

        inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        use_amp = (device.startswith("cuda") and amp_dtype in (torch.float16, torch.bfloat16))
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else torch.no_grad()

        with ctx:
            out = model(**inputs)
            next_logits = out.logits[:, -1, :]  # (B, V)

        two = next_logits[:, [a_id, b_id]]
        prob_a = torch.softmax(two, dim=-1)[:, 0]
        scores[start:start + len(chunk)] = prob_a.float().cpu().numpy()

    return scores


@torch.no_grad()
def predict_labels_list_mode(
    model: LlavaOnevisionForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    allowed_labels: List[str],
    device: str,
    amp_dtype: torch.dtype,
    max_new_tokens: int,
) -> List[str]:
    """
    Single prompt: model returns JSON {"labels":[...]}.
    """
    prompt = build_list_prompt(processor, allowed_labels)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    use_amp = (device.startswith("cuda") and amp_dtype in (torch.float16, torch.bfloat16))
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else torch.no_grad()

    with ctx:
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = processor.decode(output[0], skip_special_tokens=True)
    obj = extract_json_object(text)

    allowed_map = {normalize_label(x): x for x in allowed_labels}

    if obj is None or "labels" not in obj or not isinstance(obj["labels"], list):
        text_norm = normalize_label(text)
        found = []
        for key, original in allowed_map.items():
            if key in text_norm:
                found.append(original)
        return sorted(set(found))

    preds = []
    for item in obj["labels"]:
        if not isinstance(item, str):
            continue
        key = normalize_label(item)
        if key in allowed_map:
            preds.append(allowed_map[key])
    return sorted(set(preds))


def choose_hybrid_label_subset(
    all_labels: List[str],
    candidates: List[str],
    num_negatives: int,
    seed: int,
    image_id: int,
    max_candidates: int,
) -> List[str]:
    """
    Pick subset S = candidates (capped) ∪ sampled negatives (from remaining labels).
    Sampling is deterministic per image_id.
    """
    cand = list(dict.fromkeys(candidates))
    if max_candidates > 0:
        cand = cand[:max_candidates]

    cand_set = set(cand)
    remaining = [l for l in all_labels if l not in cand_set]

    rng = np.random.default_rng(seed + int(image_id))
    if num_negatives > 0 and len(remaining) > 0:
        k = min(num_negatives, len(remaining))
        neg = rng.choice(remaining, size=k, replace=False).tolist()
    else:
        neg = []

    subset = cand + neg
    subset = list(dict.fromkeys(subset))
    return subset


# ---------- Evaluation ----------
@dataclass
class EvalResults:
    num_images: int
    num_classes: int
    mode: str
    threshold: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    mAP: Optional[float] = None
    mAP_name: Optional[str] = None
    ap_per_class: Optional[List[float]] = None
    class_names: Optional[List[str]] = None


def evaluate(
    coco_root: str,
    split: str,
    model_id: str,
    mode: str,
    device: str,
    dtype_str: str,
    chunk_size: int,
    limit: Optional[int],
    threshold: float,
    max_new_tokens: int,
    save_json: Optional[str],
    hybrid_negatives: int,
    hybrid_seed: int,
    hybrid_max_candidates: int,
) -> EvalResults:
    ann_file = os.path.join(coco_root, "annotations", f"instances_{split}.json")
    img_dir = os.path.join(coco_root, split)
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"Missing annotation file: {ann_file}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Missing image directory: {img_dir}")

    coco = COCO(ann_file)
    _, cat_names_sorted, catid_to_index = coco_categories_in_annotation(coco)

    labels = cat_names_sorted
    num_classes = len(labels)

    img_ids = coco.getImgIds()
    if limit is not None:
        img_ids = img_ids[:limit]

    amp_dtype = set_torch_dtype(dtype_str)

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=(amp_dtype if amp_dtype in (torch.float16, torch.bfloat16) else torch.float32),
        low_cpu_mem_usage=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # >>> CHANGE REQUESTED: set padding side to left (recommended for generation)
    processor.tokenizer.padding_side = "left"

    N = len(img_ids)
    y_true = np.zeros((N, num_classes), dtype=np.int32)
    y_pred = np.zeros((N, num_classes), dtype=np.int32)
    y_score = np.zeros((N, num_classes), dtype=np.float32) if mode in ("yesno", "hybrid") else None

    name_to_idx = {name: i for i, name in enumerate(labels)}

    for i, img_id in enumerate(tqdm(img_ids, desc=f"Evaluating ({mode})")):
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(img_dir, info["file_name"])
        image = Image.open(path).convert("RGB")

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        present = {a["category_id"] for a in anns}
        for cid in present:
            if cid in catid_to_index:
                y_true[i, catid_to_index[cid]] = 1

        if mode == "yesno":
            scores = score_labels_yesno_ab(
                model=model,
                processor=processor,
                image=image,
                labels=labels,
                device=device,
                chunk_size=chunk_size,
                amp_dtype=amp_dtype,
            )
            y_score[i] = scores
            y_pred[i] = (scores >= threshold).astype(np.int32)

        elif mode == "list":
            pred_labels = predict_labels_list_mode(
                model=model,
                processor=processor,
                image=image,
                allowed_labels=labels,
                device=device,
                amp_dtype=amp_dtype,
                max_new_tokens=max_new_tokens,
            )
            idxs = [name_to_idx[lab] for lab in pred_labels if lab in name_to_idx]
            y_pred[i, idxs] = 1

        elif mode == "hybrid":
            candidates = predict_labels_list_mode(
                model=model,
                processor=processor,
                image=image,
                allowed_labels=labels,
                device=device,
                amp_dtype=amp_dtype,
                max_new_tokens=max_new_tokens,
            )

            subset = choose_hybrid_label_subset(
                all_labels=labels,
                candidates=candidates,
                num_negatives=hybrid_negatives,
                seed=hybrid_seed,
                image_id=img_id,
                max_candidates=hybrid_max_candidates,
            )

            subset_scores = score_labels_yesno_ab(
                model=model,
                processor=processor,
                image=image,
                labels=subset,
                device=device,
                chunk_size=chunk_size,
                amp_dtype=amp_dtype,
            )

            for lab, sc in zip(subset, subset_scores.tolist()):
                y_score[i, name_to_idx[lab]] = float(sc)

            y_pred[i] = (y_score[i] >= threshold).astype(np.int32)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    mAP = None
    mAP_name = None
    ap_per_class = None

    if mode == "yesno":
        valid = y_true.sum(axis=0) > 0
        ap = np.full((num_classes,), np.nan, dtype=np.float32)
        ap[valid] = average_precision_score(y_true[:, valid], y_score[:, valid], average=None)
        mAP = float(np.nanmean(ap))
        mAP_name = "mAP_full80"
        ap_per_class = [float(x) if np.isfinite(x) else float("nan") for x in ap.tolist()]

    elif mode == "hybrid":
        valid = y_true.sum(axis=0) > 0
        ap = np.full((num_classes,), np.nan, dtype=np.float32)
        ap[valid] = average_precision_score(y_true[:, valid], y_score[:, valid], average=None)
        mAP = float(np.nanmean(ap))
        mAP_name = "approx_mAP_zero_fill_unscored"
        ap_per_class = [float(x) if np.isfinite(x) else float("nan") for x in ap.tolist()]

    results = EvalResults(
        num_images=N,
        num_classes=num_classes,
        mode=mode,
        threshold=threshold,
        precision_micro=float(p_micro),
        recall_micro=float(r_micro),
        f1_micro=float(f1_micro),
        precision_macro=float(p_macro),
        recall_macro=float(r_macro),
        f1_macro=float(f1_macro),
        mAP=mAP,
        mAP_name=mAP_name,
        ap_per_class=ap_per_class,
        class_names=labels,
    )

    if save_json is not None:
        out = {
            "num_images": results.num_images,
            "num_classes": results.num_classes,
            "mode": results.mode,
            "threshold": results.threshold,
            "precision_micro": results.precision_micro,
            "recall_micro": results.recall_micro,
            "f1_micro": results.f1_micro,
            "precision_macro": results.precision_macro,
            "recall_macro": results.recall_macro,
            "f1_macro": results.f1_macro,
            "mAP": results.mAP,
            "mAP_name": results.mAP_name,
            "class_names": results.class_names,
            "ap_per_class": results.ap_per_class,
            "hybrid_negatives": hybrid_negatives if mode == "hybrid" else None,
            "hybrid_seed": hybrid_seed if mode == "hybrid" else None,
            "hybrid_max_candidates": hybrid_max_candidates if mode == "hybrid" else None,
        }
        with open(save_json, "w") as f:
            json.dump(out, f, indent=2)

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_root", type=str, required=True, help="Path to COCO root folder")
    p.add_argument("--split", type=str, default="val2017", help="val2017 or train2017")
    p.add_argument("--model_id", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")

    p.add_argument("--mode", type=str, choices=["yesno", "list", "hybrid"], default="yesno",
                   help="yesno=per-class scoring (AP/mAP), list=single prompt (P/R/F1), hybrid=list+cand scoring")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--chunk_size", type=int, default=16, help="Label prompts per forward pass (yesno/hybrid scoring)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of images for quick runs")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for y_pred in yesno/hybrid")
    p.add_argument("--max_new_tokens", type=int, default=128, help="For list/hybrid list step generation")
    p.add_argument("--save_json", type=str, default=None)

    # Hybrid-specific
    p.add_argument("--hybrid_negatives", type=int, default=16,
                   help="How many negative labels to sample per image for scoring in hybrid mode")
    p.add_argument("--hybrid_seed", type=int, default=123,
                   help="Seed for deterministic negative sampling in hybrid mode")
    p.add_argument("--hybrid_max_candidates", type=int, default=20,
                   help="Cap on how many candidate labels from list step to score (0 = no cap)")

    return p.parse_args()


def main():
    args = parse_args()

    res = evaluate(
        coco_root=args.coco_root,
        split=args.split,
        model_id=args.model_id,
        mode=args.mode,
        device=args.device,
        dtype_str=args.dtype,
        chunk_size=args.chunk_size,
        limit=args.limit,
        threshold=args.threshold,
        max_new_tokens=args.max_new_tokens,
        save_json=args.save_json,
        hybrid_negatives=args.hybrid_negatives,
        hybrid_seed=args.hybrid_seed,
        hybrid_max_candidates=args.hybrid_max_candidates,
    )

    print("\n=== Results ===")
    print(f"Mode: {res.mode}")
    print(f"Images: {res.num_images} | Classes: {res.num_classes}")
    print(f"Micro P/R/F1: {res.precision_micro:.4f} / {res.recall_micro:.4f} / {res.f1_micro:.4f}")
    print(f"Macro P/R/F1: {res.precision_macro:.4f} / {res.recall_macro:.4f} / {res.f1_macro:.4f}")
    if res.mAP is not None:
        print(f"{res.mAP_name}: {res.mAP:.4f}")
    if args.save_json:
        print(f"Saved JSON: {args.save_json}")


if __name__ == "__main__":
    main()

