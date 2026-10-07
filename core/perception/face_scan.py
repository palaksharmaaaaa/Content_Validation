"""core.perception.face_scan: a recall-first, multi-scale face finder for the minor screening.

The shared detector (core.face_detection) is tuned for a clean answer to "how many faces": confidence 0.80, nothing under 24 px.
For child screening a missed face is the costly error and a spurious one costs only a look, so this finder trades precision
for recall: a lower confidence, and a second pass over overlapping tiles that are enlarged before detection, which is how a
face of 12-30 px in a wide shot becomes large enough for the network to see. It reuses the bundled YuNet model.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.face_detection import MODEL_PATH
from core.perception.hub import silence_noise

silence_noise()
logger = logging.getLogger("core.perception.face_scan")

SCORE_THRESHOLD = 0.60      # detector floor; the shared finder uses 0.80
ACCEPT_SCORE = 0.80         # a face this confident is always reported
ACCEPT_SCORE_SMALL = 0.70   # ... and a face under SMALL_SIDE px needs only this: tiny real faces score lower, and tiny false ones are cheap
SMALL_SIDE = 28
NMS_THRESHOLD = 0.30
MIN_SIDE = 12               # smallest face (original pixels) worth reporting
MAX_SIDE = 2560             # larger images are shrunk first: a face is still big enough there, and the tile count stays bounded
TILE = 400                  # tile edge before enlargement
TILE_STRIDE = 300           # tiles overlap by 100 px so a face on a seam is whole in at least one tile
TILE_SCALE = 2.0
MERGE_IOU = 0.40

Box = Tuple[int, int, int, int]  # x, y, w, h


def _iou(a: Box, b: Box) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax2, bx2) - max(a[0], b[0]), min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter)


def merge_scored(scored: List[Tuple[Box, float]], iou: float = MERGE_IOU) -> List[Tuple[Box, float]]:
    """Greedy non-maximum suppression: keep the most confident box of every overlapping group."""
    kept: List[Tuple[Box, float]] = []
    for box, score in sorted(scored, key=lambda s: -s[1]):
        if all(_iou(box, k) < iou for k, _ in kept):
            kept.append((box, score))
    return kept


def merge_boxes(scored: List[Tuple[Box, float]], iou: float = MERGE_IOU) -> List[Box]:
    """``merge_scored`` without the scores."""
    return [b for b, _ in merge_scored(scored, iou)]


class ScreeningFaceFinder:
    """Thread-safe YuNet wrapper; ``find`` returns [] if the model is unavailable."""

    def __init__(self) -> None:
        self._net: Optional["cv2.FaceDetectorYN"] = None
        self._lock = threading.Lock()
        self._failed = False

    def _ensure(self):
        if self._net is None and not self._failed:
            try:
                self._net = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, 5000)
            except Exception as exc:
                self._failed = True
                logger.warning("Screening face finder unavailable: %s", exc)
        return self._net

    def _detect(self, frame: np.ndarray) -> List[Tuple[Box, float]]:
        with self._lock:
            self._net.setInputSize((frame.shape[1], frame.shape[0]))
            _ok, raw = self._net.detect(np.ascontiguousarray(frame, dtype=np.uint8))
        return [((float(r[0]), float(r[1]), float(r[2]), float(r[3])), float(r[-1])) for r in (raw if raw is not None else [])]

    def find(self, image_bgr: np.ndarray) -> List[Box]:
        """Faces as (x, y, w, h) in the pixels of ``image_bgr``: the whole image plus enlarged overlapping tiles, merged."""
        return [b for b, _ in self.find_scored(image_bgr)]

    def find_scored(self, image_bgr: np.ndarray) -> List[Tuple[Box, float]]:
        """Like ``find`` with each face's detector confidence."""
        if image_bgr is None or image_bgr.ndim < 2 or min(image_bgr.shape[:2]) < 16 or self._ensure() is None:
            return []
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
        h0, w0 = image_bgr.shape[:2]
        shrink = MAX_SIDE / max(h0, w0) if max(h0, w0) > MAX_SIDE else 1.0
        work = cv2.resize(image_bgr, (int(w0 * shrink), int(h0 * shrink)), interpolation=cv2.INTER_AREA) if shrink < 1.0 else image_bgr
        h, w = work.shape[:2]

        found: List[Tuple[Box, float]] = [((x, y, bw, bh), s) for (x, y, bw, bh), s in self._detect(work)]
        if max(h, w) > TILE:
            for ty in range(0, max(1, h - TILE + TILE_STRIDE), TILE_STRIDE):
                for tx in range(0, max(1, w - TILE + TILE_STRIDE), TILE_STRIDE):
                    y1, x1 = min(ty, max(0, h - TILE)), min(tx, max(0, w - TILE))
                    tile = work[y1:y1 + TILE, x1:x1 + TILE]
                    if min(tile.shape[:2]) < 16:
                        continue
                    big = cv2.resize(tile, None, fx=TILE_SCALE, fy=TILE_SCALE, interpolation=cv2.INTER_CUBIC)
                    for (x, y, bw, bh), s in self._detect(big):
                        found.append(((x / TILE_SCALE + x1, y / TILE_SCALE + y1, bw / TILE_SCALE, bh / TILE_SCALE), s))

        boxes = []
        for (x, y, bw, bh), score in merge_scored(found):
            x1, y1 = max(0, int(round(x / shrink))), max(0, int(round(y / shrink)))
            x2, y2 = min(w0, int(round((x + bw) / shrink))), min(h0, int(round((y + bh) / shrink)))
            small = min(x2 - x1, y2 - y1) < SMALL_SIDE
            if x2 - x1 >= MIN_SIDE and y2 - y1 >= MIN_SIDE and score >= (ACCEPT_SCORE_SMALL if small else ACCEPT_SCORE):
                boxes.append(((x1, y1, x2 - x1, y2 - y1), score))
        return boxes


_default: Optional[ScreeningFaceFinder] = None
_default_lock = threading.Lock()


def get_screening_finder() -> ScreeningFaceFinder:
    """The process-wide finder."""
    global _default
    with _default_lock:
        if _default is None:
            _default = ScreeningFaceFinder()
        return _default
