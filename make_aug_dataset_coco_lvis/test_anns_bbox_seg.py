import json

config_path = "/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/lvis/annotations/lvis_v1_train.json"
print("loading anns ...")
with open(config_path, "r") as f:
    ann_json = json.load(f)
print("annotations loaded!")


bboxs, segs = 0, 0

print("parsing annotations ...")
for a in ann_json.get("annotations", []):
    s, b = a.get("segmentation", None), a.get("bbox", None)
    
    print(s, b)
    if s is not None and len(s) > 0:
        segs += 1
    if b is not None:
        bboxs += 1

print(config_path)
print("annotations parsed", bboxs-segs, bboxs, segs)
