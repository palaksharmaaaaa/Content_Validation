"""core.imageio: image reading that works for every file name and bit depth.

``cv2.imread`` cannot open paths with non-ASCII characters on Windows (for example ``1920×1080.jpg``) and returns 16-bit
data for some PNGs, which later colour conversions reject. ``imread`` reads the bytes itself and always returns 8-bit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


def _within_pixel_ceiling(path: Union[str, Path]) -> bool:
    """True if the header says the picture is no larger than ``SAFE_MAX_IMAGE_PIXELS``. OpenCV allocates the whole decoded array, so a
    small PNG that declares a huge canvas must be refused before it is decoded. A file whose header cannot be read is refused too."""
    from PIL import Image

    from core.security import SAFE_MAX_IMAGE_PIXELS

    try:
        with Image.open(path) as im:
            width, height = im.size
    except Exception:
        return False
    return width * height <= SAFE_MAX_IMAGE_PIXELS


def imread(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Like ``cv2.imread`` but unicode-safe and always uint8. Returns None if the file cannot be decoded."""
    if not _within_pixel_ceiling(path):
        return None
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
