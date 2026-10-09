"""core.frame_scorer: the one per-frame AI-likelihood scorer, used for video frames by both the video detector's built-in path
and the image detector's ``predict_frame``.

A frame is judged from two measurements of its grey plane: the median-filter noise residual (a stand-in for sensor grain; every
camera frame keeps about NOISE_BASELINE of it) and the bilateral-filter residual (surface smoothness). Both are mapped through a
logistic curve and blended. The cut-offs are hand-set heuristics, not calibrated on labelled video.
"""
from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

NOISE_BASELINE = 0.70
NOISE_AI_THRESHOLD = 2.0              # balanced sensitivity: noise above the baseline below this is AI-like
NOISE_AI_THRESHOLD_SENSITIVE = 2.4    # high / aggressive sensitivity
SMOOTH_AI_THRESHOLD = 3.1             # bilateral-filter residual below this reads as over-smoothed (balanced sensitivity)
SMOOTH_AI_THRESHOLD_SENSITIVE = 3.6   # high / aggressive sensitivity
NOISE_WEIGHT, SMOOTH_WEIGHT = 0.55, 0.45
AI_LABEL_AT, AI_LABEL_AT_SENSITIVE = 0.60, 0.50
REAL_LABEL_AT = 0.35
_SENSITIVE = ("high", "aggressive")


def score_frame(frame_bgr: np.ndarray, sensitivity: str = "balanced") -> Dict[str, Any]:
    """``{label, prediction, ai_prob, real_prob, frame_noise}`` for one BGR frame."""
    sensitive = sensitivity in _SENSITIVE
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    noise = float(np.mean(cv2.absdiff(gray, cv2.medianBlur(gray, 3))))
    smooth = float(np.mean(cv2.absdiff(gray, cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75))))

    comp_noise = max(0.2, noise - NOISE_BASELINE)
    noise_thresh = NOISE_AI_THRESHOLD_SENSITIVE if sensitive else NOISE_AI_THRESHOLD
    smooth_thresh = SMOOTH_AI_THRESHOLD_SENSITIVE if sensitive else SMOOTH_AI_THRESHOLD
    p_noise_ai = float(1.0 / (1.0 + np.exp((comp_noise - noise_thresh) * 2.0)))
    p_smooth_ai = float(1.0 / (1.0 + np.exp((smooth - smooth_thresh) * 1.3)))
    score = p_noise_ai * NOISE_WEIGHT + p_smooth_ai * SMOOTH_WEIGHT

    ai_at = AI_LABEL_AT_SENSITIVE if sensitive else AI_LABEL_AT
    label = "LIKELY AI-GENERATED" if score >= ai_at else ("LIKELY REAL" if score <= REAL_LABEL_AT else "UNDECIDED")
    return {"label": label, "prediction": label, "ai_prob": round(score, 3), "real_prob": round(1.0 - score, 3), "frame_noise": round(noise, 3)}
