"""
core.face_detection: a real face detector (YuNet, a small CNN) shared by the image and video packages.

Replaces the earlier skin-colour heuristic. The model file (``core/models/face_detection_yunet_2023mar.onnx``, 228 KB,
Apache-2.0, from the OpenCV model zoo) is bundled so detection works offline. Detection runs on CPU through OpenCV.

Accuracy on your own media is not measured by this project; YuNet is a published detector, but see docs/LIMITATIONS.md.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.perception.hub import silence_noise

silence_noise()
logger = logging.getLogger("core.face_detection")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD = 0.80      # YuNet confidence below this is discarded
NMS_THRESHOLD = 0.30
MAX_SIDE = 1280             # larger images are downscaled for speed; boxes are mapped back
MIN_FACE_PIXELS = 24        # smaller than this (in the original image) is not reported

Face = Tuple[int, int, int, int, float]  # x, y, w, h, confidence


class FaceFinder:
    """Thread-safe wrapper around ``cv2.FaceDetectorYN`` (the OpenCV network object is not re-entrant)."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self._path = Path(model_path)
        self._net: Optional["cv2.FaceDetectorYN"] = None
        self._lock = threading.Lock()
        self._failed = False

    @property
    def available(self) -> bool:
        """True if the model file exists and loads."""
        return self._ensure() is not None

    def _ensure(self):
        if self._net is None and not self._failed:
            try:
                self._net = cv2.FaceDetectorYN.create(str(self._path), "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, 5000)
            except Exception as exc:
                self._failed = True
                logger.warning("Face detector unavailable (%s): %s", self._path.name, exc)
        return self._net

    def find(self, image_bgr: np.ndarray) -> List[Face]:
        """Faces in a BGR image as (x, y, w, h, confidence) in original pixel coordinates, most confident first."""
        return [(*d["box"], d["score"]) for d in self.find_detailed(image_bgr)]

    def find_detailed(self, image_bgr: np.ndarray) -> List[dict]:
        """Like ``find`` but each face is ``{box: (x, y, w, h), score, landmarks: 5x2 array}`` (right eye, left eye, nose,
        right mouth corner, left mouth corner), all in original pixel coordinates."""
        if image_bgr is None or not hasattr(image_bgr, "shape") or image_bgr.ndim < 2:
            return []
        h, w = image_bgr.shape[:2]
        if h < 16 or w < 16:
            return []
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
        net = self._ensure()
        if net is None:
            return []
        scale = MAX_SIDE / max(h, w) if max(h, w) > MAX_SIDE else 1.0
        frame = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else image_bgr
        with self._lock:
            net.setInputSize((frame.shape[1], frame.shape[0]))
            _ok, raw = net.detect(np.ascontiguousarray(frame, dtype=np.uint8))
        faces: List[dict] = []
        for row in (raw if raw is not None else []):
            x, y, fw, fh = (float(v) / scale for v in row[:4])
            x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
            x2, y2 = min(w, int(round(x + fw))), min(h, int(round(y + fh)))
            if x2 - x1 >= MIN_FACE_PIXELS and y2 - y1 >= MIN_FACE_PIXELS:
                landmarks = np.array(row[4:14], dtype=np.float32).reshape(5, 2) / scale
                faces.append({"box": (x1, y1, x2 - x1, y2 - y1), "score": float(row[-1]), "landmarks": landmarks})
        return sorted(faces, key=lambda f: f["score"], reverse=True)


_default: Optional[FaceFinder] = None
_default_lock = threading.Lock()


def get_face_finder() -> FaceFinder:
    """The process-wide detector (loaded on first use)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = FaceFinder()
        return _default
