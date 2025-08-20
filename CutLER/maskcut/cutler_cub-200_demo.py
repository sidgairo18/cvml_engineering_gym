#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extended by assistant: add CUB-200-2011 download/setup, sample 10 train images per class,
# and generate overlaid pseudo-masks using MaskCut/DINO.

import os
import sys
sys.path.append('../')
import argparse
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms
from scipy import ndimage
from detectron2.utils.colormap import random_color

import dino  # model
from third_party.TokenCut.unsupervised_saliency_detection import metric
from crf import densecrf
from maskcut import maskcut

# -----------------------------
# Image transformation applied to all images
# -----------------------------
ToTensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def vis_mask(input_np: np.ndarray, mask: np.ndarray, mask_color):
    fg = mask > 0.5
    rgb = np.copy(input_np)
    rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)


# -----------------------------
# CUB-200-2011 utilities
# -----------------------------
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_ARCHIVE_NAME = "CUB_200_2011.tgz"
CUB_FOLDER_NAME = "CUB_200_2011"


def download_and_extract_cub(dest_root: Path) -> Path:
    dest_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = dest_root / CUB_FOLDER_NAME
    if dataset_dir.exists() and (dataset_dir / "images").exists():
        return dataset_dir

    archive_path = dest_root / CUB_ARCHIVE_NAME

    # Download if needed
    if not archive_path.exists():
        print(f"Downloading CUB-200-2011 to {archive_path} ...")
        try:
            urllib.request.urlretrieve(CUB_URL, archive_path)
        except Exception as e:
            print("\nFailed to download automatically. If the link requires auth, download manually\n"
                  f"and place {CUB_ARCHIVE_NAME} into {dest_root}. Error: {e}")
            raise

    # Extract
    print(f"Extracting {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory
        def safe_extract(tar_obj, path=".", members=None, *, numeric_owner=False):
            for member in tar_obj.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
            tar_obj.extractall(path, members, numeric_owner=numeric_owner)
        safe_extract(tar, path=dest_root)

    if not dataset_dir.exists():
        # Some mirrors extract to a nested folder; search for it
        for p in dest_root.rglob("images.txt"):
            if (p.parent / "images").exists():
                return p.parent
        raise FileNotFoundError("CUB_200_2011 folder not found after extraction.")

    return dataset_dir


def load_cub_split(dataset_dir: Path):
    """Return dicts: image_id->(rel_path, class_id, is_train), and mappings."""
    images_txt = dataset_dir / "images.txt"
    labels_txt = dataset_dir / "image_class_labels.txt"
    split_txt = dataset_dir / "train_test_split.txt"
    classes_txt = dataset_dir / "classes.txt"

    # id -> rel path
    id_to_rel = {}
    with images_txt.open() as f:
        for line in f:
            idx, rel = line.strip().split()
            id_to_rel[int(idx)] = rel

    # id -> class id
    id_to_cls = {}
    with labels_txt.open() as f:
        for line in f:
            idx, cls = line.strip().split()
            id_to_cls[int(idx)] = int(cls)

    # id -> is_train (1 train, 0 test)
    id_to_train = {}
    with split_txt.open() as f:
        for line in f:
            idx, tr = line.strip().split()
            id_to_train[int(idx)] = bool(int(tr))

    # class id -> class name
    clsid_to_name = {}
    with classes_txt.open() as f:
        for line in f:
            clsid, name = line.strip().split()
            clsid_to_name[int(clsid)] = name

    # Build list of (abs_path, class_id, class_name) for train split
    train_by_cls = {}
    for img_id, rel in id_to_rel.items():
        if id_to_train[img_id]:
            clsid = id_to_cls[img_id]
            clsname = clsid_to_name[clsid]
            abs_path = dataset_dir / "images" / rel
            train_by_cls.setdefault((clsid, clsname), []).append(abs_path)

    return train_by_cls


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser('MaskCut Demo on CUB-200-2011')

    # Model / method args (kept compatible with original script)
    parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'],
                        help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'],
                        help='which features')
    parser.add_argument('--patch-size', type=int, default=8, choices=[16, 8],
                        help='patch size')
    parser.add_argument('--tau', type=float, default=0.15,
                        help='threshold used for producing binary graph')
    parser.add_argument('--fixed_size', type=int, default=480,
                        help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain_path', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--N', type=int, default=3,
                        help='the maximum number of pseudo-masks per image')
    parser.add_argument('--cpu', action='store_true', help='use cpu')

    # New args for dataset & batching
    parser.add_argument('--dataset_root', type=str, default='./data',
                        help='where to download/extract CUB-200-2011')
    parser.add_argument('--output_path', type=str, default='./outputs_cub200',
                        help='output directory for overlays')
    parser.add_argument('--samples-per-class', type=int, default=10,
                        help='number of random train images per class')
    parser.add_argument('--seed', type=int, default=0, help='random seed for sampling')

    # Optional single-image mode (kept for parity)
    parser.add_argument('--img-path', type=str, default=None,
                        help='single image visualization; if provided, dataset loop is skipped')

    args = parser.parse_args()
    print(args)

    # -----------------------------
    # Prepare model (DINO backbone)
    # -----------------------------
    if args.pretrain_path is not None:
        url = args.pretrain_path
    if args.vit_arch == 'base' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        feat_dim = 384
    else:
        raise ValueError("Unsupported combination of vit-arch and patch-size for this demo.")

    print(f"Load {args.vit_arch} pre-trained feature from {url} ...")
    backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
    backbone.eval()
    if not args.cpu:
        backbone.cuda()

    # -----------------------------
    # Single image mode (original behavior)
    # -----------------------------
    if args.img_path is not None:
        ensure_dir(Path(args.output_path))
        with torch.no_grad():
            bipartitions, _, I_new = maskcut(args.img_path, backbone, args.patch_size, args.tau,
                                             N=args.N, fixed_size=args.fixed_size, cpu=args.cpu)
        I = Image.open(args.img_path).convert('RGB')
        width, height = I.size
        pseudo_mask_list = []
        for bipartition in bipartitions:
            pseudo_mask = densecrf(np.array(I_new), bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)

            if not args.cpu:
                mask1 = torch.from_numpy(bipartition).cuda()
                mask2 = torch.from_numpy(pseudo_mask).cuda()
            else:
                mask1 = torch.from_numpy(bipartition)
                mask2 = torch.from_numpy(pseudo_mask)
            if metric.IoU(mask1, mask2) < 0.5:
                pseudo_mask = pseudo_mask * -1

            pseudo_mask[pseudo_mask < 0] = 0
            pseudo_mask = Image.fromarray(np.uint8(pseudo_mask * 255))
            pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

            pseudo_mask = pseudo_mask.astype(np.uint8)
            upper = np.max(pseudo_mask)
            lower = np.min(pseudo_mask)
            thresh = upper / 2.0
            pseudo_mask[pseudo_mask > thresh] = upper
            pseudo_mask[pseudo_mask <= thresh] = lower
            pseudo_mask_list.append(pseudo_mask)

        input_np = np.array(I)
        for pseudo_mask in pseudo_mask_list:
            input_np = np.array(vis_mask(input_np, pseudo_mask, random_color(rgb=True)))
        ensure_dir(Path(args.output_path))
        Image.fromarray(input_np).save(os.path.join(args.output_path, "demo.jpg"))
        sys.exit(0)

    # -----------------------------
    # Dataset mode (CUB-200-2011 train split)
    # -----------------------------
    random.seed(args.seed)
    dataset_dir = download_and_extract_cub(Path(args.dataset_root))
    print(f"CUB-200-2011 ready at: {dataset_dir}")

    train_by_cls = load_cub_split(dataset_dir)
    print(f"Found {len(train_by_cls)} classes in train split.")

    out_root = Path(args.output_path)
    ensure_dir(out_root)

    # Iterate classes and sample images
    total_processed = 0
    for (clsid, clsname), img_paths in sorted(train_by_cls.items(), key=lambda x: x[0][0]):
        print(f"Running clsid: {clsid}, clsname: {clsname} ...")
        sample = random.sample(img_paths, k=min(args.samples_per_class, len(img_paths)))
        class_out = out_root / f"{clsid:03d}_{clsname}"
        ensure_dir(class_out)

        for img_path in sample:
            try:
                with torch.no_grad():
                    bipartitions, _, I_new = maskcut(str(img_path), backbone, args.patch_size, args.tau,
                                                     N=args.N, fixed_size=args.fixed_size, cpu=args.cpu)

                I = Image.open(img_path).convert('RGB')
                width, height = I.size
                pseudo_mask_list = []
                for bipartition in bipartitions:
                    pseudo_mask = densecrf(np.array(I_new), bipartition)
                    pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)

                    if not args.cpu:
                        mask1 = torch.from_numpy(bipartition).cuda()
                        mask2 = torch.from_numpy(pseudo_mask).cuda()
                    else:
                        mask1 = torch.from_numpy(bipartition)
                        mask2 = torch.from_numpy(pseudo_mask)
                    if metric.IoU(mask1, mask2) < 0.5:
                        pseudo_mask = pseudo_mask * -1

                    pseudo_mask[pseudo_mask < 0] = 0
                    pseudo_mask = Image.fromarray(np.uint8(pseudo_mask * 255))
                    pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

                    pseudo_mask = pseudo_mask.astype(np.uint8)
                    upper = np.max(pseudo_mask)
                    lower = np.min(pseudo_mask)
                    thresh = upper / 2.0
                    pseudo_mask[pseudo_mask > thresh] = upper
                    pseudo_mask[pseudo_mask <= thresh] = lower
                    pseudo_mask_list.append(pseudo_mask)

                input_np = np.array(I)
                for pseudo_mask in pseudo_mask_list:
                    input_np = np.array(vis_mask(input_np, pseudo_mask, random_color(rgb=True)))

                rel_name = img_path.name
                save_name = rel_name.rsplit('.', 1)[0] + "_overlay.jpg"
                Image.fromarray(input_np).save(class_out / save_name)
                total_processed += 1
            except Exception as e:
                print(f"[WARN] Failed on {img_path}: {e}")
                continue
        print(f"Done clsid: {clsid}, clsname: {clsname}!")

    print(f"Done. Saved overlays to: {out_root}  | Total images processed: {total_processed}")

