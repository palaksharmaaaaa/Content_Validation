"""image_detector.jpeg: what a JPEG's quantisation tables say about how hard it was compressed.

Pixel-noise statistics only mean something on a picture whose grain survived. JPEG quantisation removes fine grain in proportion to
how coarse the tables are, so the compression level has to be known before the grain is read as evidence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

_STD_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]
_STD_CHROMA = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
]
_ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
]


def libjpeg_table(base: List[int], quality: int) -> List[int]:
    q = max(1, min(100, int(quality)))
    scale = 5000 // q if q < 50 else 200 - 2 * q
    return [max(1, min(255, (b * scale + 50) // 100)) for b in base]


def estimate_libjpeg_quality(tables: Dict[int, List[int]]) -> Optional[int]:
    """Quality (1-100) if the luma table equals the libjpeg-scaled standard table, else None."""
    luma = tables.get(0)
    if not luma or len(luma) != 64:
        return None
    luma = [int(v) for v in luma]
    zig = [luma[_ZIGZAG.index(i)] for i in range(64)]  # tolerate zigzag-ordered storage
    for q in range(1, 101):
        ref = libjpeg_table(_STD_LUMA, q)
        if luma == ref or zig == ref:
            return q
    return None




def jpeg_compression(path: str | Path) -> Tuple[bool, Optional[int]]:
    """(is_jpeg, libjpeg-equivalent quality). The quality is None for a JPEG written with its own tables (a camera, Photoshop, ...)
    and for anything that is not a JPEG or cannot be read."""
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return False, None
            return True, estimate_libjpeg_quality(dict(getattr(im, "quantization", {}) or {}))
    except Exception as exc:
        logger.debug("jpeg_compression: ignored %s: %s", type(exc).__name__, exc)
        return False, None
