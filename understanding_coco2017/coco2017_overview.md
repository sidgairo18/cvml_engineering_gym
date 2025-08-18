# COCO 2017 Dataset Overview

The **COCO (Common Objects in Context) 2017** dataset is a large-scale benchmark dataset widely used for computer vision tasks such as object detection, instance segmentation, keypoint detection, captioning, and panoptic segmentation.

---

## Splits & Sizes (Instances)

* **train2017**: \~118k images
* **val2017**: 5k images
* **test2017**: \~40.7k images (annotations withheld; evaluation via server)

Additional tasks (captions, keypoints, stuff, panoptic) provide separate annotations for the same train/val splits.

---

## Folder Structure

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
    captions_{train,val}2017.json
    stuff_{train,val}2017.json
    panoptic_{train,val}2017.json
    panoptic_{train,val}2017/   # PNG masks for panoptic segmentation
  {train,val,test}2017/
    000000xxxxx.jpg
```

---

## Annotation JSON Schema (Instances)

* **images**

  * `id`, `file_name`, `height`, `width`
* **annotations**

  * `id`
  * `image_id`
  * `category_id`
  * `bbox`: `[x, y, w, h]` (top-left corner, width, height)
  * `segmentation`: polygon(s) or RLE mask
  * `area`: pixel area of the object
  * `iscrowd`: {0 = normal, 1 = crowd/ambiguous}
* **categories**

  * `id`, `name`, `supercategory`
  * 80 object categories ("things")

---

## Keypoints

* Defined only for `person` (category\_id = 1)
* **17 keypoints** per person
* Stored as `[x, y, visibility] * 17`
* Visibility flags: `0` (not labeled), `1` (labeled but not visible), `2` (visible)

---

## Stuff & Panoptic

* **Stuff:** Amorphous background regions (e.g., sky, grass)
* **Panoptic:** Combines "things" and "stuff" into a unified per-pixel segmentation map (JSON + PNG masks)

---

## Common Gotchas

* `bbox` format is **\[x, y, w, h]**, but many frameworks expect **\[x1, y1, x2, y2]**. Convert carefully.
* `category_id` values are **not continuous 0..79**; always map via the `categories` list.
* `iscrowd=1` annotations use RLE masks; evaluation handles them specially.
* **Test sets**: no ground-truth annotations are available.

---

## Typical Uses

* **Detection & Segmentation**: `instances_*.json`
* **Keypoints**: `person_keypoints_*.json`
* **Captioning**: `captions_*.json`
* **Stuff Segmentation**: `stuff_*.json`
* **Panoptic Segmentation**: `panoptic_*.json` + PNG masks

---

## References

* [COCO dataset official site](https://cocodataset.org)
* [Dataset paper](https://arxiv.org/abs/1405.0312)
* [pycocotools GitHub](https://github.com/cocodataset/cocoapi)


Note: Created with ChatGPT_5.
