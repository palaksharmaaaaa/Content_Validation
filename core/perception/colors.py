"""core.perception.colors: dominant colours with plain human names.

No neural network is needed for this: colour is measured, not recognised. Pixels are clustered in CIELAB (where distance
matches perceived difference) and each cluster centre is named by its nearest entry in a table of everyday colour names.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# name -> sRGB reference colour
_NAMED: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0), "charcoal": (54, 54, 54), "dark gray": (89, 89, 89), "gray": (128, 128, 128), "light gray": (190, 190, 190),
    "silver": (215, 215, 215), "white": (255, 255, 255), "cream": (255, 248, 220), "ivory": (250, 245, 230), "beige": (225, 205, 170),
    "tan": (200, 165, 120), "khaki": (190, 180, 120), "brown": (110, 70, 40), "dark brown": (65, 40, 25), "chocolate": (90, 55, 30),
    "maroon": (110, 20, 35), "dark red": (140, 15, 15), "red": (215, 30, 35), "crimson": (200, 20, 60), "coral": (245, 110, 90),
    "salmon": (250, 140, 115), "peach": (255, 205, 170), "pink": (245, 150, 185), "light pink": (250, 205, 215), "hot pink": (240, 60, 150),
    "magenta": (200, 30, 160), "orange": (245, 130, 20), "saffron": (250, 150, 30), "amber": (255, 190, 0), "mustard": (210, 170, 30),
    "gold": (212, 175, 55), "yellow": (250, 230, 40), "light yellow": (255, 250, 170), "olive": (110, 115, 40), "lime": (150, 215, 40),
    "light green": (150, 220, 150), "green": (40, 160, 70), "dark green": (20, 85, 45), "forest green": (35, 100, 50), "mint": (160, 235, 205),
    "teal": (20, 130, 130), "turquoise": (50, 205, 200), "cyan": (40, 220, 240), "sky blue": (125, 195, 235), "light blue": (170, 210, 235),
    "blue": (10, 50, 240), "royal blue": (40, 60, 175), "navy": (20, 30, 90), "indigo": (60, 40, 140), "purple": (115, 45, 160),
    "violet": (150, 90, 200), "lavender": (200, 180, 235), "plum": (110, 40, 90),
}
_NAMES = list(_NAMED)
_REF_LAB = cv2.cvtColor(np.array([[list(_NAMED[n]) for n in _NAMES]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def name_color(r: int, g: int, b: int) -> str:
    """Everyday name of an sRGB colour (nearest in CIELAB)."""
    lab = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return _NAMES[int(np.argmin(((_REF_LAB - lab) ** 2).sum(axis=1)))]


def dominant_colors(image_bgr: np.ndarray, k: int = 6, side: int = 96) -> List[Dict[str, Any]]:
    """Up to ``k`` dominant colours as ``{hex, rgb, color_name, coverage_pct}``, largest first. Clusters that get the same
    name are merged so the list reads as distinct colours. Deterministic."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    small = cv2.resize(image_bgr[:, :, :3], (side, side), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    k = max(1, min(k, len(np.unique(lab, axis=0))))
    cv2.setRNGSeed(0)
    _compact, labels, centers = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k).astype(np.float64)
    merged: Dict[str, Dict[str, Any]] = {}
    for idx in np.argsort(counts)[::-1]:
        rgb = cv2.cvtColor(centers[idx].reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
        r, g, b = (int(v) for v in rgb)
        name = name_color(r, g, b)
        entry = merged.setdefault(name, {"hex": f"#{r:02x}{g:02x}{b:02x}", "rgb": (r, g, b), "color_name": name.title(), "coverage_pct": 0.0})
        entry["coverage_pct"] += float(counts[idx] / counts.sum() * 100.0)
    out = sorted(merged.values(), key=lambda e: e["coverage_pct"], reverse=True)
    for e in out:
        e["coverage_pct"] = round(e["coverage_pct"], 1)
    return out
