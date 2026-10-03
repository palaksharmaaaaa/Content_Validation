"""core.perception.enrich: turn detector boxes and whole images into recognised, human-readable descriptions."""
from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from core.perception.recognizer import get_recognizer
from core.perception.vocab import INDOOR

MIN_CONFIDENCE = 0.30        # below this the best option of a closed vocabulary is reported as "not sure"
MAX_CROPS = 6                # at most this many boxes per kind are recognised per image
MIN_CROP_SIDE = 32


def _rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _crop(image_bgr: np.ndarray, bbox: Dict[str, Any], pad: float = 0.10) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = (int(bbox[k]) for k in ("x", "y", "width", "height"))
    dx, dy = int(bw * pad), int(bh * pad)
    return image_bgr[max(0, y - dy): min(h, y + bh + dy), max(0, x - dx): min(w, x + bw + dx)]


def recognize_details(image_bgr: np.ndarray, details: List[Dict[str, Any]], vocab: str) -> None:
    """Add ``recognized_as`` (or None when unsure), ``recognition_confidence`` and ``candidates`` to each detection
    (``details`` entries carry a ``bbox`` dict). Boxes are processed largest first; only ``MAX_CROPS`` of them."""
    recognizer = get_recognizer()
    if not details or not recognizer.available:
        return
    order = sorted(range(len(details)), key=lambda i: -(details[i]["bbox"]["width"] * details[i]["bbox"]["height"]))[:MAX_CROPS]
    crops, idx = [], []
    for i in order:
        crop = _crop(image_bgr, details[i]["bbox"])
        if min(crop.shape[:2]) >= MIN_CROP_SIDE:
            crops.append(_rgb(crop))
            idx.append(i)
    for i, top in zip(idx, recognizer.classify(crops, vocab, top_k=3)):
        label, prob = top[0]
        details[i]["recognized_as"] = label if prob >= MIN_CONFIDENCE else None
        details[i]["recognition_confidence"] = round(prob, 3)
        details[i]["candidates"] = [{"label": l, "probability": round(p, 3)} for l, p in top]


def describe_scene(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Scene (place), indoor/outdoor, time of day and photo genre for the whole image. Empty dict if the model is missing."""
    recognizer = get_recognizer()
    if not recognizer.available:
        return {}
    res = recognizer.classify_multi(_rgb(image_bgr), ["scene", "time_of_day", "genre"], top_k=3)
    out: Dict[str, Any] = {}
    for key, name in (("scene", "scene"), ("time_of_day", "time_of_day"), ("genre", "genre")):
        top = res.get(name) or []
        if top:
            out[key] = top[0][0] if top[0][1] >= MIN_CONFIDENCE else None
            out[f"{key}_confidence"] = round(top[0][1], 3)
            out[f"{key}_candidates"] = [{"label": l, "probability": round(p, 3)} for l, p in top]
    scene = out.get("scene")
    out["indoor"] = INDOOR.get(scene) if scene else None
    return out
