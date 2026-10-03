"""core.perception.face_attributes: facial expression and same-person matching.

Both models come from the OpenCV model zoo (Apache-2.0) and run on CPU through OpenCV's DNN module:
  * expression: MobileFaceNet classifier (angry, disgust, fearful, happy, neutral, sad, surprised), 4.8 MB
  * identity: SFace embeddings (128-d) compared by cosine similarity, 39 MB

Identity here means "is this the same face as that one" across the faces of one analysis. No names are stored and no
identity database exists; embeddings live only in the in-memory result.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger("core.perception.face_attributes")

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
FER_PATH = MODEL_DIR / "facial_expression_recognition_mobilefacenet_2022july.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
EXPRESSIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
SAME_PERSON_COSINE = 0.363     # the threshold published with SFace (about 99.4 % accuracy on LFW at this value)
_STD_POINTS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _align(image_bgr: np.ndarray, landmarks: np.ndarray, size: int = 112) -> Optional[np.ndarray]:
    """Similarity-warp the face so the five landmarks sit on the standard 112 x 112 template."""
    matrix, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, dtype=np.float32), _STD_POINTS, method=cv2.LMEDS)
    return None if matrix is None else cv2.warpAffine(image_bgr, matrix, (size, size), flags=cv2.INTER_LINEAR)


class FaceAttributes:
    """Loads both ONNX models lazily; every method returns empty results if a model file is missing."""

    def __init__(self) -> None:
        self._fer = None
        self._sface = None
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                self._fer = cv2.dnn.readNetFromONNX(str(FER_PATH))
            except Exception as exc:
                logger.warning("Expression model unavailable: %s", exc)
            try:
                self._sface = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
            except Exception as exc:
                logger.warning("Face-recognition model unavailable: %s", exc)

    @property
    def expression_available(self) -> bool:
        """True if the expression model loaded."""
        self._load()
        return self._fer is not None

    @property
    def identity_available(self) -> bool:
        """True if the face-recognition model loaded."""
        self._load()
        return self._sface is not None

    def expression(self, image_bgr: np.ndarray, landmarks: np.ndarray) -> Optional[Dict[str, Any]]:
        """``{label, confidence, scores}`` for one face, or None."""
        self._load()
        aligned = _align(image_bgr, landmarks)
        if self._fer is None or aligned is None:
            return None
        blob = cv2.dnn.blobFromImage(((cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0) - 0.5) / 0.5)
        with self._lock:
            self._fer.setInput(blob)
            logits = self._fer.forward()[0].astype(np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top = int(probs.argmax())
        return {"label": EXPRESSIONS[top], "confidence": round(float(probs[top]), 3),
                "scores": {name: round(float(p), 3) for name, p in zip(EXPRESSIONS, probs)}}

    def embedding(self, image_bgr: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """Unit-length 128-d identity embedding for one face, or None."""
        self._load()
        aligned = _align(image_bgr, landmarks)
        if self._sface is None or aligned is None:
            return None
        with self._lock:
            vec = self._sface.feature(aligned)[0].astype(np.float64)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else None


def group_same_person(embeddings: Sequence[Optional[np.ndarray]], threshold: float = SAME_PERSON_COSINE) -> List[int]:
    """Group index (0, 1, 2 ...) for each embedding: faces whose cosine similarity to a group's first member is at least
    ``threshold`` share a group. ``-1`` for faces without an embedding. Deterministic, greedy, order-stable."""
    firsts: List[np.ndarray] = []
    groups: List[int] = []
    for emb in embeddings:
        if emb is None:
            groups.append(-1)
            continue
        scores = [float(emb @ f) for f in firsts]
        if scores and max(scores) >= threshold:
            groups.append(int(np.argmax(scores)))
        else:
            firsts.append(emb)
            groups.append(len(firsts) - 1)
    return groups


_default: Optional[FaceAttributes] = None
_default_lock = threading.Lock()


def get_face_attributes() -> FaceAttributes:
    """The process-wide instance."""
    global _default
    with _default_lock:
        if _default is None:
            _default = FaceAttributes()
        return _default
