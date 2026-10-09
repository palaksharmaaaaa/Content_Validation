"""
image_detector.face_authenticity: is a detected face a real photograph or a generated one?

A small ResNet-18 trained on face crops (see image_detector.face_training). It is shown the face found by the
YuNet detector plus a margin, resized to 112 x 112, and returns the probability that the face is AI-generated.

Honest scope: it was trained on one public face dataset, so it recognises the generators and camera styles in that
dataset. It is reported as advisory evidence and never changes the main verdict on its own. Without a trained
checkpoint it reports itself as unavailable.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("image_detector.face_authenticity")

CHECKPOINT = Path(__file__).resolve().parent / "models" / "face_authenticity.pt"
INPUT_SIZE = 112
MARGIN = 0.40               # context added around the detector's face box (hair, jaw), as a fraction of its size
MIN_FACE_PIXELS = 64        # smaller faces are too coarse to judge
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_model(pretrained: bool):
    """ResNet-18 with a two-class head (0 = real, 1 = AI-generated)."""
    import torch.nn as nn
    from torchvision import models

    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.fc.in_features, 2))
    return net


def to_tensor(rgb_square: np.ndarray):
    """uint8 RGB (H, W, 3) already at INPUT_SIZE -> normalised float tensor (3, H, W)."""
    import torch

    x = torch.from_numpy(np.array(rgb_square, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std = torch.tensor(_STD).view(3, 1, 1)
    return (x - mean) / std


def crop_region(image_bgr: np.ndarray, box: Sequence[int]) -> np.ndarray:
    """Square BGR crop around ``box`` (x, y, w, h) with MARGIN of context, at the source resolution.
    Training and inference both build their input through this function so that framing is identical."""
    import cv2

    h, w = image_bgr.shape[:2]
    x, y, bw, bh = box[:4]
    side = max(bw, bh) * (1.0 + 2 * MARGIN)
    cx, cy = x + bw / 2.0, y + bh / 2.0
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad = max(0, -x1, -y1, x2 - w, y2 - h)
    if pad:
        image_bgr = cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    return image_bgr[y1:y2, x1:x2]


def crop_face(image_bgr: np.ndarray, box: Sequence[int]) -> np.ndarray:
    """RGB crop of the face (see ``crop_region``) resized to INPUT_SIZE."""
    import cv2

    crop = cv2.resize(crop_region(image_bgr, box), (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


class FaceAuthenticityClassifier:
    """Loads the checkpoint on first use; thread-safe."""

    def __init__(self, checkpoint: Path = CHECKPOINT):
        self._path = Path(checkpoint)
        self._model = None
        self._meta: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._failed = False

    def _ensure(self):
        with self._lock:
            if self._model is None and not self._failed:
                if not self._path.is_file():
                    self._failed = True
                    return None
                try:
                    import torch

                    from core.perception.hub import ensure_not_lfs_pointer

                    ensure_not_lfs_pointer(self._path)
                    state = torch.load(self._path, map_location="cpu", weights_only=True)
                    net = build_model(pretrained=False)
                    net.load_state_dict(state["model_state_dict"])
                    net.eval()
                    self._model, self._meta = net, dict(state.get("meta", {}))
                except Exception as exc:
                    logger.warning("Face authenticity model could not be loaded: %s", exc)
                    self._failed = True
            return self._model

    @property
    def available(self) -> bool:
        """True once a trained checkpoint exists and loads."""
        return self._ensure() is not None

    @property
    def meta(self) -> Dict[str, Any]:
        """Training metadata saved with the checkpoint (validation accuracy, sample counts)."""
        self._ensure()
        return self._meta

    def predict_crops(self, rgb_crops: List[np.ndarray]) -> List[float]:
        """P(AI-generated), in [0, 1], for each INPUT_SIZE RGB crop."""
        net = self._ensure()
        if net is None or not rgb_crops:
            return []
        import torch

        with torch.no_grad():
            logits = net(torch.stack([to_tensor(c) for c in rgb_crops]))
            return [float(p) for p in torch.softmax(logits, dim=1)[:, 1]]

    def analyse(self, image_bgr: np.ndarray, faces: Sequence[Tuple[int, int, int, int, float]]) -> Dict[str, Any]:
        """Judge every face of at least MIN_FACE_PIXELS. Returns ``{available, faces: [{bbox, p_ai}], skipped_small}``."""
        usable = [f for f in faces if min(f[2], f[3]) >= MIN_FACE_PIXELS]
        out: Dict[str, Any] = {"available": self.available, "faces": [], "skipped_small": len(faces) - len(usable)}
        if not out["available"] or not usable:
            return out
        probs = self.predict_crops([crop_face(image_bgr, f) for f in usable])
        out["faces"] = [{"bbox": [int(v) for v in f[:4]], "p_ai": round(p, 4)} for f, p in zip(usable, probs)]
        return out


_default: Optional[FaceAuthenticityClassifier] = None
_default_lock = threading.Lock()


def get_face_authenticity() -> FaceAuthenticityClassifier:
    """The process-wide classifier."""
    global _default
    with _default_lock:
        if _default is None:
            _default = FaceAuthenticityClassifier()
        return _default
