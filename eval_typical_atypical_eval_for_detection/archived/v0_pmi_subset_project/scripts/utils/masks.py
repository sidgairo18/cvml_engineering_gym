import numpy as np
import cv2

def compute_ioa(mask_a, mask_b):
    # IOA of A on B: |A ∩ B| / |A|
    inter = np.logical_and(mask_a>0, mask_b>0).sum()
    area_a = (mask_a>0).sum() + 1e-6
    return inter / area_a

def bbox_from_mask(m):
    ys, xs = np.where(m>0)
    if len(xs)==0 or len(ys)==0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)]

def feather_alpha(mask, radius=7):
    # build soft alpha from binary mask by blurring edges
    m = (mask>0).astype(np.float32)
    alpha = cv2.GaussianBlur(m, (0,0), radius)
    alpha = alpha / (alpha.max()+1e-6)
    return np.clip(alpha, 0, 1)
