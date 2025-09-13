import numpy as np
import cv2
from .masks import feather_alpha

def paste_with_feather(bg, fg, fg_mask, x, y, feather_radius=7):
    H, W = bg.shape[:2]
    h, w = fg_mask.shape
    x2, y2 = x+w, y+h
    if x<0 or y<0 or x2>W or y2>H:
        return bg.copy(), None
    alpha = feather_alpha(fg_mask, radius=feather_radius)
    out = bg.copy()
    roi = out[y:y2, x:x2]
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR) if fg.shape[:2] != (h,w) else fg
    if fg.ndim == 2:
        fg = np.stack([fg]*3, axis=-1)
    if roi.ndim == 2:
        roi = np.stack([roi]*3, axis=-1)
    alpha3 = np.dstack([alpha]*3)
    roi = (alpha3*fg + (1-alpha3)*roi).astype(bg.dtype)
    out[y:y2, x:x2] = roi
    return out, alpha
