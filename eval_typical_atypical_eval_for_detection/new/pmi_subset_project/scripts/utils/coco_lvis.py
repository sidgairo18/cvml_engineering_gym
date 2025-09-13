import json, os
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

class DatasetAPI:
    def __init__(self, dataset:str, images_root:str, ann_path:str):
        self.dataset = dataset
        self.images_root = images_root
        self.ann_path = ann_path
        self.coco = COCO(ann_path)
        self.cat_id_to_name = {c['id']: c['name'] for c in self.coco.loadCats(self.coco.getCatIds())}
        self.name_to_cat_id = {v:k for k,v in self.cat_id_to_name.items()}

    def iter_instances(self, cat_name=None, min_area=1.0):
        cat_ids = None
        if cat_name is not None:
            if isinstance(cat_name, (list, tuple)):
                cat_ids = self.coco.getCatIds(catNms=list(cat_name))
            else:
                cat_ids = self.coco.getCatIds(catNms=[cat_name])
        img_ids = self.coco.getImgIds()
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            anns = [a for a in anns if a.get('area',1.0)>=min_area and a.get('iscrowd',0)==0]
            if not anns:
                continue
            img = self.coco.loadImgs([img_id])[0]
            yield img, anns

    def image_path(self, img):
        return os.path.join(self.images_root, img['file_name'])

    @staticmethod
    def ann_to_mask(h, w, ann):
        if 'segmentation' in ann:
            rle = None
            if isinstance(ann['segmentation'], list):
                rle = mask_utils.frPyObjects(ann['segmentation'], h, w)
                rle = mask_utils.merge(rle)
            elif isinstance(ann['segmentation'], dict):
                rle = ann['segmentation']
            else:
                return None
            m = mask_utils.decode(rle).astype('uint8')
            return m
        return None
