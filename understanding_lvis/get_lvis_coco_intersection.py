from pycocotools.coco import COCO
from lvis import LVIS

def compute_lvis_coco_overlap(lvis_train_json, lvis_val_json,
                              coco_train_json, coco_val_json):
    """
    Compute number of common images between LVIS and COCO splits.

    Args:
      lvis_train_json: path to lvis_v1_train.json
      lvis_val_json: path to lvis_v1_val.json
      coco_train_json: path to instances_train2017.json
      coco_val_json: path to instances_val2017.json

    Returns:
      dict with counts of overlaps
    """
    # load APIs
    lvis_train = LVIS(lvis_train_json)
    lvis_val   = LVIS(lvis_val_json)
    coco_train = COCO(coco_train_json)
    coco_val   = COCO(coco_val_json)

    # sets of image ids
    lvis_train_ids = set(lvis_train.imgs.keys())
    lvis_val_ids   = set(lvis_val.imgs.keys())
    coco_train_ids = set(coco_train.imgs.keys())
    coco_val_ids   = set(coco_val.imgs.keys())

    # overlaps
    overlaps = {
        "lvis_train ∩ coco_train": len(lvis_train_ids & coco_train_ids),
        "lvis_train ∩ coco_val":   len(lvis_train_ids & coco_val_ids),
        "lvis_val ∩ coco_train":   len(lvis_val_ids & coco_train_ids),
        "lvis_val ∩ coco_val":     len(lvis_val_ids & coco_val_ids),
    }
    return overlaps

base_dir = "/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/"
lvis_train_json = base_dir+"lvis/annotations/lvis_v1_train.json"
lvis_val_json = base_dir+"lvis/annotations/lvis_v1_val.json"
coco_train_json = base_dir+"coco/annotations/instances_train2017.json"
coco_val_json = base_dir+"coco/annotations/instances_val2017.json"
print("Running overlaps ...")
overlaps = compute_lvis_coco_overlap(
        lvis_train_json=lvis_train_json,
        lvis_val_json=lvis_val_json,
        coco_train_json=coco_train_json,
        coco_val_json=coco_val_json
)
print(overlaps)
