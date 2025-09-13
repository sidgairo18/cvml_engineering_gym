import numpy as np
import random
from .masks import compute_ioa

def sample_position(bg_h, bg_w, fg_mask, existing_masks, max_tries=30, ioa_thresh=0.4):
    fh, fw = fg_mask.shape
    for _ in range(max_tries):
        y = random.randint(0, max(0, bg_h - fh))
        x = random.randint(0, max(0, bg_w - fw))
        # compute IOA of fg on all existing masks
        max_ioa = 0.0
        for em in existing_masks:
            # place fg mask at (x,y) and check intersection region
            y1, y2 = y, y+fh
            x1, x2 = x, x+fw
            # fast overlap check
            h = min(y2, em.shape[0]) - y1
            w = min(x2, em.shape[1]) - x1
            if h<=0 or w<=0: 
                continue
            sub_em = em[y1:y2, x1:x2]
            sub_fg = fg_mask[:h, :w]
            inter = np.logical_and(sub_em>0, sub_fg>0).sum()
            area_fg = (sub_fg>0).sum()+1e-6
            ioa = inter / area_fg
            if ioa > max_ioa:
                max_ioa = ioa
        if max_ioa <= ioa_thresh:
            return x, y
    return None

def place_relation(bg_h, bg_w, mask_a, mask_b, relation='next to', existing_masks=None, ioa_thresh=0.4):
    if existing_masks is None:
        existing_masks = []
    existing_masks = list(existing_masks)
    # naive: place A first, then B near A depending on relation
    pos_a = sample_position(bg_h, bg_w, mask_a, existing_masks, ioa_thresh=ioa_thresh)
    if pos_a is None:
        return None, None
    xa, ya = pos_a
    existing_masks.append(translate_mask(mask_a, xa, ya, bg_h, bg_w))
    if relation == 'on':
        # try to put B below A slightly
        xb = xa + np.random.randint(-10, 10)
        yb = min(bg_h - mask_b.shape[0], ya + mask_a.shape[0] + np.random.randint(-5, 15))
        return (xa, ya), (xb, yb)
    else:
        # 'next to'
        xb = min(bg_w - mask_b.shape[1], xa + mask_a.shape[1] + np.random.randint(5, 20))
        yb = ya + np.random.randint(-10, 10)
        return (xa, ya), (xb, yb)

def translate_mask(mask, x, y, H, W):
    out = np.zeros((H,W), dtype=mask.dtype)
    h, w = mask.shape
    x2, y2 = x+w, y+h
    if x<0 or y<0 or x2>W or y2>H:
        return out
    out[y:y2, x:x2] = mask
    return out
