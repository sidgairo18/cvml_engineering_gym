import numpy as np

def compute_ioa(mask_a, mask_b):
    inter = np.logical_and(mask_a>0, mask_b>0).sum()
    area_a = (mask_a>0).sum() + 1e-6
    return inter / area_a

def bbox_from_mask(m):
    ys, xs = np.where(m>0)
    if len(xs)==0 or len(ys)==0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)]
