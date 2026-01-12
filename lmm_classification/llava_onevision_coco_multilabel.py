#!/usr/bin/env python3
import os
import re
import json
import time
import argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from pycocotools.coco import COCO

from sklearn.metrics import precision_recall_fscore_support, average_precision_score


# ----------------------------
# Text normalization / cleaning
# ----------------------------
def norm_phrase(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def singularize_basic(w: str) -> str:
    w = w.strip().lower()
    if w == "knives":
        return "knife"
    if w == "mice":
        return "mouse"
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("ves") and len(w) > 3:
        return w[:-3] + "f"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def clean_items(items: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for it in items:
        it = norm_phrase(it)
        it = singularize_basic(it)
        if len(it) <= 1:
            continue
        if it not in seen:
            seen.add(it)
            cleaned.append(it)
    return cleaned


# ----------------------------
# Robust parsing of model output
# ----------------------------
def _collect_strings_from_json(x, out: List[str]):
    """Collect string VALUES from nested JSON (do NOT collect dict keys)."""
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, list):
        for v in x:
            _collect_strings_from_json(v, out)
    elif isinstance(x, dict):
        for v in x.values():
            _collect_strings_from_json(v, out)


def parse_objects_any(text: str) -> Tuple[List[str], str]:
    """
    Robustly parse objects from:
      - multiple JSON arrays in one output (non-greedy)
      - multiple JSON dicts in one output (non-greedy)
      - quoted strings
      - fallback split
    Returns: (items, parse_mode)
    """
    t = text.strip()
    t = re.sub(r"^```(json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    # 1) Parse ALL JSON arrays (non-greedy) - handles multi-block outputs
    arrays = re.findall(r"\[[\s\S]*?\]", t)
    merged = []
    for a in arrays:
        try:
            obj = json.loads(a)
            out = []
            _collect_strings_from_json(obj, out)
            merged.extend(out)
        except Exception:
            pass
    if merged:
        merged = [str(x).strip().lower() for x in merged if isinstance(x, str)]
        return merged, "json_any_multi"

    # 2) Parse ALL JSON dicts (non-greedy)
    dicts = re.findall(r"\{[\s\S]*?\}", t)
    merged = []
    for d in dicts:
        try:
            obj = json.loads(d)
            out = []
            _collect_strings_from_json(obj, out)
            merged.extend(out)
        except Exception:
            pass
    if merged:
        merged = [str(x).strip().lower() for x in merged if isinstance(x, str)]
        return merged, "json_any_multi"

    # 3) Quoted strings fallback
    quoted = re.findall(r'"([^"]+)"', t)
    if quoted:
        return [q.strip().lower() for q in quoted], "quoted"

    # 4) Last fallback: split by commas/semicolons/newlines and drop long phrases
    t2 = re.sub(r"[\[\]\{\}\"]", " ", t)
    parts = re.split(r"[,;\n]+", t2)
    items = []
    for p in parts:
        p = p.strip().lower()
        if not p:
            continue
        if len(p.split()) > 4:
            continue
        items.append(p)
    return items, "fallback"


# ----------------------------
# COCO mapping
# ----------------------------
def build_alias_dict() -> Dict[str, str]:
    # Your current aliases
    return {
        "sofa": "couch",
        "settee": "couch",
        "television": "tv",
        "mobile phone": "cell phone",
        "phone": "cell phone",
        "smartphone": "cell phone",
        "cellphone": "cell phone",
        "bike": "bicycle",
        "motorbike": "motorcycle",
        "mug": "cup",
        "teddy": "teddy bear",
        "stuffed bear": "teddy bear",
        "stuffed toy": "teddy bear",
        "traffic signal": "traffic light",
        "signal": "traffic light",
        # plural safety
        "knives": "knife",
        "forks": "fork",
        "spoons": "spoon",
        "cups": "cup",
        "bottles": "bottle",
    }


def map_to_coco(
    objects: List[str],
    coco_set: Dict[str, str],
    alias_to_coco: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    mapped = set()
    unmapped = []

    for obj in objects:
        o = norm_phrase(obj)
        if not o:
            continue

        if o in coco_set:
            mapped.add(coco_set[o])
            continue

        if o in alias_to_coco:
            mapped.add(alias_to_coco[o])
            continue

        # strict word-boundary alias match
        hit_alias = None
        for alias, coco_lab in alias_to_coco.items():
            if re.search(rf"\b{re.escape(alias)}\b", o):
                hit_alias = coco_lab
                break
        if hit_alias is not None:
            mapped.add(hit_alias)
            continue

        # multiword COCO contains match
        hit = None
        for coco_norm, coco_official in coco_set.items():
            if " " in coco_norm and re.search(rf"\b{re.escape(coco_norm)}\b", o):
                hit = coco_official
                break

        if hit is not None:
            mapped.add(hit)
        else:
            unmapped.append(obj)

    return sorted(mapped), unmapped


# ----------------------------
# Model load helpers
# ----------------------------
def parse_dtype(s: str):
    s = s.lower()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def infer_input_device(model) -> torch.device:
    """
    With device_map='auto', model can be sharded. Pick a sensible device for inputs.
    """
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        devs = []
        for d in model.hf_device_map.values():
            if d in ("cpu", "disk", "meta"):
                continue
            devs.append(d)
        if devs:
            d0 = devs[0]
            if isinstance(d0, int):
                return torch.device(f"cuda:{d0}")
            if isinstance(d0, str):
                return torch.device(d0)

    try:
        return model.device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_processor(model_id: str, dtype: torch.dtype, load_4bit: bool):
    if load_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if dtype in (torch.float16, torch.bfloat16) else torch.float32,
        )
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()
    else:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        ).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    return model, processor


# ----------------------------
# Prompt builders
# ----------------------------
def build_prompt_main_objects(processor) -> str:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "List the object categories that are actually visible in the image.\n"
                "Return ONLY a JSON array of at most 12 UNIQUE short nouns (1-2 words).\n"
                "If you are unsure, omit the object.\n"
                "No explanations. No extra text."
            )},
            {"type": "image"},
        ],
    }]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def build_prompt_small_objects(processor) -> str:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "List ONLY small objects that are CLEARLY visible (utensils/containers/handheld items).\n"
                "Return ONLY a JSON array of at most 8 UNIQUE short nouns (1-2 words).\n"
                "If you are not 100% sure an object is present, DO NOT include it.\n"
                "Do NOT guess utensils.\n"
                "No explanations. No extra text."
            )},
            {"type": "image"},
        ],
    }]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def build_prompt_caption(processor) -> str:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in one short sentence."},
            {"type": "image"},
        ],
    }]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


# ----------------------------
# Batched generation
# ----------------------------
@torch.no_grad()
def generate_batch(
    model,
    processor,
    images: List[Image.Image],
    prompts: List[str],
    max_new_tokens: int = 96,
    repetition_penalty: float = 1.2,
) -> List[str]:
    assert len(images) == len(prompts)
    inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)

    dev = infer_input_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    L = inputs["input_ids"].shape[1]
    gen_only = out[:, L:]
    texts = processor.batch_decode(gen_only, skip_special_tokens=True)
    return [t.strip() for t in texts]


# ----------------------------
# Baselines (batched)
# ----------------------------
def predict_single_pass_batch(
    model, processor,
    images: List[Image.Image],
    coco_set: Dict[str, str], alias_to_coco: Dict[str, str],
    max_new_tokens=96, repetition_penalty=1.2,
) -> List[Dict[str, Any]]:
    prompt = build_prompt_main_objects(processor)
    prompts = [prompt] * len(images)
    raws = generate_batch(model, processor, images, prompts, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty)

    outs = []
    for raw in raws:
        items, mode = parse_objects_any(raw)
        items = clean_items(items)
        mapped, unmapped = map_to_coco(items, coco_set, alias_to_coco)
        outs.append({
            "mapped": mapped,
            "unmapped": unmapped,
            "items": items,
            "parse_modes": [mode],
            "raws": [raw],
            "prompts": [prompt],
        })
    return outs


def predict_two_pass_gated_batch(
    model, processor,
    images: List[Image.Image],
    coco_set: Dict[str, str], alias_to_coco: Dict[str, str],
    max_new_tokens_main=96, max_new_tokens_small=64,
    repetition_penalty=1.2, small_cap=4,
) -> List[Dict[str, Any]]:
    p1 = build_prompt_main_objects(processor)
    p2 = build_prompt_small_objects(processor)

    raws1 = generate_batch(
        model, processor, images, [p1] * len(images),
        max_new_tokens=max_new_tokens_main, repetition_penalty=repetition_penalty
    )
    raws2 = generate_batch(
        model, processor, images, [p2] * len(images),
        max_new_tokens=max_new_tokens_small, repetition_penalty=repetition_penalty
    )

    outs = []
    anchors = {"person", "dining table", "toilet", "sink", "bed", "couch", "chair"}

    for raw1, raw2 in zip(raws1, raws2):
        items1, m1 = parse_objects_any(raw1)
        items2, m2 = parse_objects_any(raw2)

        items1 = clean_items(items1)
        items2 = clean_items(items2)

        mapped1, _ = map_to_coco(items1, coco_set, alias_to_coco)
        mapped2, _ = map_to_coco(items2, coco_set, alias_to_coco)

        if len(set(mapped1) & anchors) == 0:
            mapped = mapped1
        else:
            mapped = sorted(set(mapped1) | set(mapped2[:max(0, small_cap)]))

        outs.append({
            "mapped": mapped,
            "items": sorted(set(items1 + items2)),
            "parse_modes": [m1, m2],
            "raws": [raw1, raw2],
            "prompts": [p1, p2],
            "mapped_main": mapped1,
            "mapped_small": mapped2,
        })

    return outs


# ----------------------------
# COCO helpers (paths-based)
# ----------------------------
def load_coco_from_paths(ann_file: str, img_dir: str) -> Tuple[COCO, str, str]:
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(ann_file)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(img_dir)
    coco = COCO(ann_file)
    return coco, ann_file, img_dir


def get_gt_labels_for_img(coco: COCO, img_id: int) -> List[str]:
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    present_cids = sorted({a["category_id"] for a in anns})
    cats = coco.loadCats(present_cids)
    return sorted([c["name"] for c in cats])


def build_label_index(coco: COCO) -> Tuple[List[int], List[str], Dict[int, int], Dict[str, int]]:
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cats_sorted = sorted(cats, key=lambda x: x["id"])
    cat_ids_sorted = [c["id"] for c in cats_sorted]
    coco_labels = [c["name"] for c in cats_sorted]
    catid_to_index = {cid: i for i, cid in enumerate(cat_ids_sorted)}
    label_to_idx = {name: i for i, name in enumerate(coco_labels)}
    return cat_ids_sorted, coco_labels, catid_to_index, label_to_idx


def gt_multihot(coco: COCO, img_ids_subset: List[int], catid_to_index: Dict[int, int], num_classes: int) -> np.ndarray:
    Y = np.zeros((len(img_ids_subset), num_classes), dtype=np.int32)
    for i, iid in enumerate(img_ids_subset):
        ann_ids = coco.getAnnIds(imgIds=[iid])
        anns = coco.loadAnns(ann_ids)
        present = {a["category_id"] for a in anns}
        for cid in present:
            if cid in catid_to_index:
                Y[i, catid_to_index[cid]] = 1
    return Y


# ----------------------------
# Metrics
# ----------------------------
def report_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, name: str):
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="macro", zero_division=0
    )

    # mAP note: this is "degenerate AP" because we only have binary predictions, not confidence scores.
    ap_list = []
    for c in range(Y_true.shape[1]):
        if Y_true[:, c].sum() == 0:
            continue
        ap = average_precision_score(Y_true[:, c], Y_pred[:, c])
        ap_list.append(ap)
    map_macro = float(np.mean(ap_list)) if ap_list else 0.0

    print(f"\n=== {name} ===")
    print(f"Micro P/R/F1: {p_micro:.4f} / {r_micro:.4f} / {f1_micro:.4f}")
    print(f"Macro P/R/F1: {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
    print(f"Macro mAP (bin-scores): {map_macro:.4f}")


# ----------------------------
# Sanity dump
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def dump_sanity_artifact(
    out_dir: str,
    img_id: int,
    image: Image.Image,
    caption: str,
    gt_labels: List[str],
    pred_labels: List[str],
    payload: Dict[str, Any],
):
    ensure_dir(out_dir)

    fig = plt.figure(figsize=(10, 7))
    plt.imshow(image)
    plt.axis("off")

    text = (
        f"img_id={img_id}\n\n"
        f"CAPTION: {caption}\n\n"
        f"GT ({len(gt_labels)}): {', '.join(gt_labels)}\n\n"
        f"PRED ({len(pred_labels)}): {', '.join(pred_labels)}\n"
    )
    fig.text(0.02, 0.02, text, fontsize=10, va="bottom", ha="left", wrap=True)

    png_path = os.path.join(out_dir, f"{img_id}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)

    json_path = os.path.join(out_dir, f"{img_id}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


# ----------------------------
# JSONL writer
# ----------------------------
class JsonlWriter:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.f = None
        if path:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            self.f = open(path, "a", buffering=1)  # line-buffered

    def write(self, obj: Dict[str, Any]):
        if not self.f:
            return
        self.f.write(json.dumps(obj) + "\n")

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # NEW: explicit COCO-format annotation + image dir
    parser.add_argument("--ann_file", type=str, required=True,
                        help="Path to COCO-format annotation JSON (e.g., instances_val2017.json)")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory containing images referenced by ann_file (joined with info['file_name'])")

    parser.add_argument("--model_id", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--load_4bit", action="store_true", help="Enable 4-bit quantization (bitsandbytes required)")

    parser.add_argument("--mode", type=str, default="two_gated", choices=["single", "two_gated"])
    parser.add_argument("--k", type=int, default=30,
                        help="Max number of images to evaluate after slicing. Use -1 for no cap.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=123)

    # sharding/slicing
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index into the (optionally shuffled) img_id list")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End index (exclusive). -1 means 'to the end'.")

    # batching
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation (increase until you hit OOM)")

    parser.add_argument("--max_new_tokens_main", type=int, default=96)
    parser.add_argument("--max_new_tokens_small", type=int, default=64)
    parser.add_argument("--small_cap", type=int, default=4)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--verbose_every", type=int, default=10)

    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--sanity_out", type=str, default="sanity_outputs")

    parser.add_argument("--out_jsonl", type=str, default="",
                        help="Write per-image outputs (pred/raw/prompts/gt) to a JSONL file")

    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    if not torch.cuda.is_available():
        dtype = torch.float32

    coco, ann_file, img_dir = load_coco_from_paths(args.ann_file, args.img_dir)
    _, coco_labels, catid_to_index, label_to_idx = build_label_index(coco)

    coco_set = {norm_phrase(x): x for x in coco_labels}
    alias_to_coco = build_alias_dict()

    print("COCO ann:", ann_file)
    print("COCO img:", img_dir)
    print("Num classes:", len(coco_labels))
    print("Model:", args.model_id)
    print("Mode:", args.mode, "| 4bit:", args.load_4bit, "| dtype:", dtype)
    print("Batch size:", args.batch_size)
    print(f"Slicing: start_idx={args.start_idx}, end_idx={args.end_idx}, k={args.k}, shuffle={args.shuffle}")

    model, processor = load_model_and_processor(args.model_id, dtype=dtype, load_4bit=args.load_4bit)

    img_ids = coco.getImgIds()
    if args.shuffle:
        rng = np.random.RandomState(args.seed)
        rng.shuffle(img_ids)

    start = max(0, args.start_idx)
    end = len(img_ids) if args.end_idx is None or args.end_idx < 0 else min(len(img_ids), args.end_idx)
    subset = img_ids[start:end]

    if args.k is not None and args.k > 0:
        subset = subset[:args.k]
    # args.k <= 0 => no cap

    if len(subset) == 0:
        print("Empty subset after slicing. Exiting.")
        return

    Y_true = gt_multihot(coco, subset, catid_to_index, num_classes=len(coco_labels))
    Y_pred = np.zeros_like(Y_true, dtype=np.int32)

    sanity_dir = None
    if args.sanity_check:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        sanity_dir = ensure_dir(os.path.join(args.sanity_out, f"{args.mode}_{stamp}"))
        print("Sanity dump dir:", sanity_dir)

    caption_prompt = build_prompt_caption(processor) if args.sanity_check else None
    jsonl_path = args.out_jsonl.strip() if args.out_jsonl.strip() else None
    writer = JsonlWriter(jsonl_path)

    n = len(subset)
    for base in range(0, n, args.batch_size):
        batch_img_ids = subset[base: base + args.batch_size]

        batch_images = []
        batch_infos = []
        for img_id in batch_img_ids:
            info = coco.loadImgs(img_id)[0]
            path = os.path.join(img_dir, info["file_name"])
            image = Image.open(path).convert("RGB")
            batch_images.append(image)
            batch_infos.append(info)

        if args.mode == "single":
            outs = predict_single_pass_batch(
                model, processor, batch_images,
                coco_set, alias_to_coco,
                max_new_tokens=args.max_new_tokens_main,
                repetition_penalty=args.repetition_penalty,
            )
        else:
            outs = predict_two_pass_gated_batch(
                model, processor, batch_images,
                coco_set, alias_to_coco,
                max_new_tokens_main=args.max_new_tokens_main,
                max_new_tokens_small=args.max_new_tokens_small,
                repetition_penalty=args.repetition_penalty,
                small_cap=args.small_cap,
            )

        captions = None
        if args.sanity_check:
            captions = generate_batch(
                model, processor, batch_images, [caption_prompt] * len(batch_images),
                max_new_tokens=48, repetition_penalty=1.1
            )

        for j, (img_id, info, out) in enumerate(zip(batch_img_ids, batch_infos, outs)):
            i = base + j  # row index into Y_true/Y_pred
            mapped = out["mapped"]

            for lab in mapped:
                if lab in label_to_idx:
                    Y_pred[i, label_to_idx[lab]] = 1

            if args.verbose_every and (i % args.verbose_every == 0):
                print(f"[{args.mode}] {i}/{n} img_id={img_id} mapped={mapped}")

            gt = get_gt_labels_for_img(coco, img_id)

            record = {
                "img_id": int(img_id),
                "file_name": info["file_name"],
                "mode": args.mode,
                "model_id": args.model_id,
                "gt_labels": gt,
                "pred_labels": mapped,
                "items": out.get("items", []),
                "parse_modes": out.get("parse_modes", []),
                "raws": out.get("raws", []),
                "prompts": out.get("prompts", []),
                "mapped_main": out.get("mapped_main", None),
                "mapped_small": out.get("mapped_small", None),
            }
            if captions is not None:
                record["caption"] = captions[j]

            writer.write(record)

            if args.sanity_check:
                dump_sanity_artifact(
                    sanity_dir,
                    img_id,
                    batch_images[j],
                    captions[j],
                    gt,
                    mapped,
                    record,
                )

    report_metrics(Y_true, Y_pred, name=f"{args.mode} | {args.model_id}")

    if args.sanity_check:
        print(f"\nSanity artifacts written to: {sanity_dir}")
        print("Each image has: <img_id>.png and <img_id>.json")

    if jsonl_path:
        print(f"\nWrote JSONL to: {jsonl_path}")

    writer.close()


if __name__ == "__main__":
    main()

