"""core.imageio: image reading that works for every file name and bit depth.

``cv2.imread`` cannot open paths with non-ASCII characters on Windows (for example ``1920×1080.jpg``) and returns 16-bit
data for some PNGs, which later colour conversions reject. ``imread`` reads the bytes itself and always returns 8-bit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


def imread(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Like ``cv2.imread`` but unicode-safe and always uint8. Returns None if the file cannot be decoded."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img
