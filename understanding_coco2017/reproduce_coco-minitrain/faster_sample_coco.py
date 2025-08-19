import json
import argparse
from random import shuffle
from tqdm import tqdm

from dataloader import CocoDataset
from sampler_utils import get_coco_object_size_info, get_coco_class_object_counts, get_coco_object_size_info_kp

# set seeds
import random, numpy as np

# default area ranges defined in coco
areaRng = [32 ** 2, 96 ** 2, 1e5 ** 2]


########## additional helpers ##########
from collections import Counter

def _size_bucket(area, areaRng):
    if area < areaRng[0]: return 'S'
    if area < areaRng[1]: return 'M'
    return 'L'

def build_reference_histogram(dataset_train, areaRng):
    """Full-dataset histogram over keys: f'{cat}_{S/M/L}'."""
    ref = Counter()
    for ann in dataset_train.coco.anns.values():
        w, h = ann['bbox'][2], ann['bbox'][3]
        if w < 1 or h < 1:
            continue
        kk = f"{ann['category_id']}_{_size_bucket(w*h, areaRng)}"
        ref[kk] += 1
    return ref

def build_image_bins(dataset_train, areaRng):
    """For each image id, precompute a Counter over the same keys as the reference."""
    per_img = {}
    for im_id, anns in dataset_train.coco.imgToAnns.items():
        c = Counter()
        for ann in anns:
            w, h = w, h = ann['bbox'][2], ann['bbox'][3]
            if w < 1 or h < 1:
                continue
            kk = f"{ann['category_id']}_{_size_bucket(w*h, areaRng)}"
            c[kk] += 1

        per_img[im_id] = c
    return per_img

def evaluate_subset(per_img_bins, ref_hist, subset_ids, objective='spread'):
    """Return (score, sampled_hist). Lower score is better."""
    sampler = Counter()
    for iid in subset_ids:
        sampler += per_img_bins[iid]
    
    # Must cover all reference keys; if not, return a bad score to be skipped.
    if any(k not in sampled for k in ref_hist):
        return float('inf'), sampled

    # ratios over the full-dataset counts
    ratios = [sampled[k] / float(ref_hist[k]) for k in ref_hist]
    if objective == 'spread':
        score = (max(ratios) - min(ratios))
    else:  # 'l2'
        # penalize deviation from a uniform scaling (their mean)
        mean_r = sum(ratios) / len(ratios)
        score = sum((r - mean_r) ** 2 for r in ratios) / len(ratios)

    return score, sampled



########## additional helpers end ##########

# NEW fast sampler
def sampling_fast(dataset_train, parser, areaRng):
    ref_hist = build_reference_histogram(dataset_train, areaRng)
    per_img_bins = build_image_bins(dataset_train, areaRng)

    all_ids = list(dataset_train.coco.imgToAnns.keys())
    best_score = float('inf')
    best_subset = None

    # pre-create a Numpy array for fast permutation
    all_ids_np = np.array(all_ids)

    rng = np.random.default_rng(parser.seed)
    ran = range(parser.run_count)
    for _ in (tqdm(ran) if parser.debug else ran):
        rng.shuffle(all_ids_np)
        subset_ids = all_ids_np[:parser.sample_image_count]
        score, _ = evaluate_subset(per_img_bins, ref_hist, subset_ids, parser.objective)
        if score < best_score:
            best_score = score
            best_subset = subset_ids.copy()
            if parser.debug:
                print(f"New best score: {best_score:.6f}")
            if best_score <= parser.early_stopp_diff and parser.objective == 'spread':
                break

    # Return dict: image_id -> list of annots (like old sampling())
    imgs_best_sample = {int(i): dataset_train.coco.imgToAnns[int(i)] for i in best_subset}
    return imgs_best_sample


# New fast but greedy sampling()
def sampling_greedy(dataset_train, parser, areaRng):
    ref_hist = build_reference_histogram(dataset_train, areaRng)
    per_img_bins = build_image_bins(dataset_train, areaRng)

    # Normalize ref_hist to a target *proportion* vector for convenience
    keys = list(ref_hist.keys())
    ref_counts = np.array([ref_hist[k] for k in keys], dtype=np.float64)
    # We want sampled_counts to be proportional to ref_counts

    # Prebuild per-image vectors aligned to keys
    img_ids = list(per_img_bins.keys())
    M = len(img_ids)
    K = len(keys)
    X = np.zeros((M, K), dtype=np.int32)
    for i, iid in enumerate(img_ids):
        row = per_img_bins[iid]
        for j, k in enumerate(keys):
            X[i, j] = row.get(k, 0)

    target_n_imgs = parser.sample_image_count
    chosen = np.zeros(M, dtype=bool)
    cur = np.zeros(K, dtype, np.int64)

    rng = np.random.default_rng(parser.seed)
    order = rng.permutation(M) # mild randomization to break ties

    def score_vec(v):
        # ratios vs ref; fall back to small eps to avoid /0 (ref_hist should be >0 anyway)
        r = (v / np.maximum(ref_counts, 1e-9))
        if parser.objective == 'spread':
            return r.max() - r.min()
        else:
            m = r.mean()
            return ((r-m) **2).mean()

    # warm start: pick a few images randomly to avoid zero-vector degeneracy
    warm = min(100, target_n_images//100 +1)
    for i in order[:warm]:
        chosen[i] = True
        cur += X[i]

    while chosen.sum() < target_n_imgs:
        # Evaluate marginal gain of adding each remaining image on a small candidate pool
        # To keep it fast, only consider a random beam each step.

        pool = rng.choice(np.where(~chosen)[0], size=min(2048, (~chosen).sum()), replace=False)
        best_i, best_s = -1, float('inf')
        for i in pool:
            s = score_vec(cur+X[i])
            if s < best_s:
                best_s, best_i = s, i

        chosen[best_i] = True
        cur += X[best_i]

        if parser.debug and chosen.sum() % 1000 == 0:
            print(f"[greedy] chosen={chosen.sum()} score={best_s:.6f}")
        if parser.objective == 'spread' and best_s <= parser.early_stop_diff:
            break

    if parser.debug:
        print(f"[greed] early stopped at chosen={chosen.sum()} score={best_s:.6f}")
        print("Will fill the remaining now with same algorithm")

    # FILL PHASE: ensure we reach target_n_imgs even if we broke early
    while chosen.sum() < target_n_imgs:
        pool = rng.choice(np.where(~chosen)[0], size=min(2048, (~chosen).sum()), replace=False)
        best_i, best_s = -1, float('inf')
        for i in pool:
            s = score_vec(cur+X[i]) # same scoring as before
            if s < best_s:
                best_s, best_i = s, i
        chosen[best_i] = True
        cur+= X[best_i]
    
    # check
    selected = np.where(chosen)[0]
    assert len(selected) >= target_n_imgs, "Greedy sampler picked fewer than target images."

    selected_ids = [img_ids[i] for i in np.where(chosen)[0][:target_n_imgs]]
    imgs_best_sample = {int(i): dataset_train.coco.imgToAnns[int(i)] for i in selected_ids}
    return imgs_best_sample


# OLD slow sampler
def sampling(dataset_train, parser):
    # get coco class based on object counts
    annot_dict = get_coco_class_object_counts(dataset_train)
    if parser.debug:
        print(f"COCO object counts in each class:\n{annot_dict}")

    # fig = plt.figure()
    # axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
    # axes.plot(np.arange(0,80,1), np.divide(annot_dict.values(), float(len(dataset_train.image_ids)) ), 'r')
    # axes.plot(np.arange(0,80,1), np.divide(annot_dict.values(), float(len(dataset_train.image_ids)) ), 'r')
    # axes.set_xlabel('Class Id')
    # axes.set_ylabel('Annot Count')
    # axes.set_title('MiniCoco vs Coco 2017 train set')
    # fig.show()
    # fig.savefig("minicoco_vs_coco_train2017_annot.png")

    # here extract object sizes
    size_dict = get_coco_object_size_info(dataset_train)
    if parser.debug:
        print(f"COCO object counts in each class for different sizes (S,M,L):\n{size_dict}")

    # now sample!!

    imgs_best_sample = {}
    ratio_list = []
    best_diff = 1_000_000
    keys = []
    # get all keys in coco train set, total image count!
    for k, v in dataset_train.coco.imgToAnns.items():
        keys.append(k)

    for rr in tqdm(range(parser.run_count)):
        imgs = {}
        # shuffle keys
        shuffle(keys)

        # select first N images
        for i in keys[:parser.sample_image_count]:
            imgs[i] = dataset_train.coco.imgToAnns[i]

        # now check for category based annotations
        # annot_sampled = np.zeros(90, int)
        annot_sampled = {}
        for k, v in imgs.items():
            for it in v:
                area = it['bbox'][2] * it['bbox'][3]
                cat = it['category_id']
                if area < areaRng[0]:
                    kk = str(cat) + "_S"
                elif area < areaRng[1]:
                    kk = str(cat) + "_M"
                else:
                    kk = str(cat) + "_L"

                if kk in annot_sampled:
                    annot_sampled[kk] += 1
                else:
                    annot_sampled[kk] = 1

        if parser.debug:
            print(f"Sampled Annotations dict:\n {annot_sampled}")

        # calculate ratios
        ratios_obj_count = {}
        # ratios_obj_size = {}

        failed_run = False
        for k, v in size_dict.items():
            if not k in annot_sampled:
                failed_run = True
                break

            ratios_obj_count[k] = annot_sampled[k] / float(v)

        if failed_run:
            continue
        ratio_list.append(ratios_obj_count)

        min_ratio = min(ratios_obj_count.values())
        max_ratio = max(ratios_obj_count.values())

        diff = max_ratio - min_ratio

        if diff < best_diff:
            best_diff = diff
            imgs_best_sample = imgs

        if parser.debug:
            print(f"Best difference:{best_diff}")

    return imgs_best_sample

def sampling_kp(dataset_train, parser):
    # get coco class based object counts
    annot_dict = get_coco_class_object_counts(dataset_train)

    if parser.debug:
        print(f"COCO object counts in each class:\n{annot_dict}")

    # here extract object sizes.
    size_dict = get_coco_object_size_info_kp(dataset_train)
    if parser.debug:
        print(f"COCO object counts in each class for different sizes (S,M,L):\n{size_dict}")

    # now sample!!
    imgs_best_sample = {}
    ratio_list = []
    best_diff = 1_000_000
    keys = []
    # get all keys in coco train set, total image count!
    for k, v in dataset_train.coco.imgToAnns.items():
        keys.append(k)

    for rr in tqdm(range(parser.run_count)):
        imgs = {}

        # shuffle keys
        shuffle(keys)

        # select first N images
        for i in keys[:parser.sample_image_count]:
            imgs[i] = dataset_train.coco.imgToAnns[i]

        # now check for category based annotations
        # annot_sampled = np.zeros(90, int)
        annot_sampled = {}

        for k, v in imgs.items():
            for it in v:
                area = it['bbox'][2] * it['bbox'][3]
                cat = it['category_id']
                num_kp = it['num_keypoints']
                if num_kp == 0:  # if there is no KP annots then no need to include
                    continue

                if area < areaRng[0]:
                    kk = str(cat) + "_S"
                elif area < areaRng[1]:
                    kk = str(cat) + "_M"
                else:
                    kk = str(cat) + "_L"

                kk += f"_{num_kp}"
                if kk in annot_sampled:
                    annot_sampled[kk] += 1
                else:
                    annot_sampled[kk] = 1

        if parser.debug:
            print(f"Sampled Annotations dict:\n {annot_sampled}")

        # calculate ratios
        ratios_obj_count = {}
        # ratios_obj_size = {}

        failed_run = False
        for k, v in size_dict.items():
            if not k in annot_sampled:
                failed_run = True
                break

            ratios_obj_count[k] = annot_sampled[k] / float(v)

        if failed_run:
            continue

        ratio_list.append(ratios_obj_count)

        min_ratio = min(ratios_obj_count.values())
        max_ratio = max(ratios_obj_count.values())

        diff = max_ratio - min_ratio

        if diff < best_diff:
            best_diff = diff
            imgs_best_sample = imgs

        if parser.debug:
            print(f"Best difference:{best_diff}")

    return imgs_best_sample


def main(args=None):
    parser = argparse.ArgumentParser(description='Mini COCO Sampling Script')
    parser.add_argument('--coco_path', help='Path to COCO directory', 
                        default='/BS/')
    parser.add_argument('--save_file_name', help='Save file name', default="person_keypoints_train2017_minicoco")
    parser.add_argument('--save_format', help='Save to json or csv', default="json")
    parser.add_argument('--sample_image_count', help='How many images you want to sample', type=int, default=15000)
    parser.add_argument('--run_count', help='How many times you want to run sampling', type=int, default=200000)
    parser.add_argument('--debug', help='Print useful info', action='store_true')
    parser.add_argument('--sample_kp', help='Sample Keypoints', action='store_true')

    # new flags
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--early_stop_diff', type=float, default=0.02,
                        help='Stop when (max_ratio - min_ratio) <= this.')
    parser.add_argument('--objective', choices=['spread', 'l2'], default='spread',
                        help='spread = max(r)-min(r) (original); l2 = mean squared error of ratios.')
    parser.add_argument('--sampler', choices=['random', 'greedy'], default='random',
                        help='random = your approach (faster version); greedy = faster convergence, no big run_count needed.')
    parser.add_argument('--save_ids', action='store_true',
                        help='Also save selected image ids to <save_file_name>_image_ids.txt')


    parser = parser.parse_args(args)
    dataset_train = CocoDataset(parser.coco_path, set_name='train2017', sample_kp=parser.sample_kp)

    if parser.sample_kp:
        imgs_best_sample = sampling_kp(dataset_train, parser)
    else:
        if parser.sampler == "greedy":
            imgs_best_sample = sampling_greedy(dataset_train, parser, areaRng)
            # with greedy set --run_count=1
        else:
            imgs_best_sample = sampling_fast(dataset_train, parser, areaRng)
        # older logic
        #imgs_best_sample = sampling(dataset_train, parser)

    # save image ids
    if parser.save_ids:
        out = f"{parser.save_file_name}_image_ids.txt"
        with open(out, "w") as f:
            for iid in imgs_best_sample.keys():
                f.write(f"{iid}\n")

    if parser.save_format == 'csv':
        # now write to csv file
        csv_file = open(f"{parser.save_file_name}.csv", 'w')

        for k, v in imgs_best_sample.items():
            f_name = dataset_train.coco.imgs[k]['file_name']
            for ann in v:
                bbox = ann['bbox']
                class_id = ann['category_id']
                write_str = f_name + ',' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(
                        bbox[3]) + ',' + \
                                str(dataset_train.labels[dataset_train.coco_labels_inverse[class_id]]) + ',' + '0' + '\n'
                csv_file.write(write_str)

        csv_file.close()


    elif parser.save_format == 'json':
        mini_coco = {}
        annots = []
        imgs = []
        # add headers like info, licenses etc.
        mini_coco["info"] = dataset_train.coco.dataset['info']
        mini_coco["licenses"] = dataset_train.coco.dataset['licenses']
        mini_coco["categories"] = dataset_train.coco.dataset['categories']
        for k, v in imgs_best_sample.items():
            f_name = dataset_train.coco.imgs[k]['file_name']
            im_id = int(f_name[:-4])
            for ann in dataset_train.coco.imgToAnns[im_id]:
                annots.append(ann)
            imgs.append(dataset_train.coco.imgs[im_id])

        mini_coco['images'] = imgs
        mini_coco['annotations'] = annots


        with open(f"{parser.save_file_name}.json", 'w') as jsonf:
            json.dump(mini_coco, jsonf)

    exit(0)

if __name__ == '__main__':
    random.seed(parser.seed)
    np.random.seed(parser.seed)
    main()
