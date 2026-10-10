"""core.perception.detector: object detection with RF-DETR Small (Apache-2.0, Roboflow, COCO 80 classes)."""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from core.perception.hub import load_kwargs, silence_noise

silence_noise()
logger = logging.getLogger("core.perception.detector")

MODEL_ID = "Roboflow/rf-detr-small"
MODEL_REVISION = "3bdc465063270f99769da5a1b4c00c68bd2d439d"   # pinned: identical weights on every machine
DEFAULT_THRESHOLD = 0.50

ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
VEHICLES = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}


class ObjectDetector:
    """Loads the model on first use; thread-safe. ``detect`` returns ``[]`` if the model is unavailable."""

    def __init__(self, model_id: str = MODEL_ID):
        self._id = model_id
        self._model = None
        self._proc = None
        self._lock = threading.Lock()
        self._failed = False

    def _ensure(self):
        with self._lock:
            if self._model is None and not self._failed:
                try:
                    import torch
                    from transformers import AutoImageProcessor, AutoModelForObjectDetection

                    self._proc = AutoImageProcessor.from_pretrained(self._id, **load_kwargs(self._id, MODEL_REVISION if self._id == MODEL_ID else None))
                    self._model = AutoModelForObjectDetection.from_pretrained(self._id, **load_kwargs(self._id, MODEL_REVISION if self._id == MODEL_ID else None)).eval()
                    if torch.cuda.is_available():
                        self._model = self._model.to("cuda")
                except Exception as exc:
                    logger.warning("Object detector %s unavailable: %s", self._id, exc)
                    self._failed = True
                    self._model = self._proc = None            # a half-loaded model must not be reported as available
            return self._model

    @property
    def available(self) -> bool:
        """True once the model has loaded."""
        return self._ensure() is not None

    def detect(self, image_bgr: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> List[Dict[str, Any]]:
        """Objects as ``{label, score, box: (x, y, w, h)}`` in original pixels, most confident first."""
        model = self._ensure()
        if model is None or image_bgr is None or image_bgr.ndim != 3:
            return []
        import cv2
        import torch
        from PIL import Image

        h, w = image_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        with self._lock, torch.no_grad():
            inputs = self._proc(images=pil, return_tensors="pt").to(next(model.parameters()).device)
            outputs = model(**inputs)
            res = self._proc.post_process_object_detection(outputs, threshold=threshold, target_sizes=[(h, w)])[0]
        found = []
        for label, score, box in zip(res["labels"], res["scores"], res["boxes"]):
            x1, y1, x2, y2 = (float(v) for v in box)
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            found.append({"label": model.config.id2label[int(label)].lower().strip(), "score": float(score),
                          "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1))})
        return sorted(found, key=lambda d: d["score"], reverse=True)


_default: Optional[ObjectDetector] = None
_default_lock = threading.Lock()


def get_object_detector() -> ObjectDetector:
    """The process-wide detector."""
    global _default
    with _default_lock:
        if _default is None:
            _default = ObjectDetector()
        return _default
