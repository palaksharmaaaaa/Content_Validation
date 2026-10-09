"""core.perception.face_texture: the facial texture cue shared by the image and video detectors.

Skin that the bilateral filter barely changes (waxy, over-smoothed) and an upper face with almost no sensor noise are two weak
signs of synthetic or heavily retouched faces. The thresholds are uncalibrated heuristics, so the result is reported as a cue
and never as a trained verdict. A measurement that cannot be taken (a face too small to hold an upper-face patch) is None and
adds no risk, never a made-up value.
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

WAXY_TEXTURE_BELOW = 3.2        # mean |face - bilateral(face)| in grey levels
LOW_NOISE_BELOW = 1.8           # mean |upper face - gaussian(upper face)| in grey levels
HIGH_RISK_AT = 0.70
SUSPICIOUS_AT = 0.45


def texture_cues(face_gray: np.ndarray) -> Optional[Dict[str, object]]:
    """Texture measurements for one grey face crop, or None for an empty crop."""
    if face_gray is None or face_gray.size == 0:
        return None
    h, w = face_gray.shape[:2]
    texture = float(np.mean(cv2.absdiff(face_gray, cv2.bilateralFilter(face_gray, 9, 75, 75))))
    upper = face_gray[max(0, int(h * 0.15)):min(h, int(h * 0.50)), max(0, int(w * 0.20)):min(w, int(w * 0.80))]
    noise = float(np.mean(cv2.absdiff(upper, cv2.GaussianBlur(upper, (3, 3), 0)))) if upper.size else None

    risk = (1.5 if texture < WAXY_TEXTURE_BELOW else 0.0) + (1.0 if noise is not None and noise < LOW_NOISE_BELOW else 0.0)
    return {
        "texture_smoothness": texture,
        "facial_noise_residual": noise,
        "is_waxy_texture": texture < WAXY_TEXTURE_BELOW,
        "score": min(0.95, (risk / 2.5) * 0.75 + 0.15),
    }


def risk_label(mean_score: float) -> str:
    if mean_score >= HIGH_RISK_AT:
        return "HIGH_SYNTHETIC_RISK"
    return "SUSPICIOUS_ARTIFACTS" if mean_score >= SUSPICIOUS_AT else "LOW_RISK_NATURAL_TEXTURE"
