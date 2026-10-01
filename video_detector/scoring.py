"""
video_detector.scoring: Self-contained scoring and temporal aggregation engine for Video AI Detection.
Contains:
1. Multi-factor temporal evidence pooling (frame spatial ratio, motion variance, diffusion flicker).
2. Epistemic uncertainty estimation via Shannon entropy.
3. Invariant percentage normalization: P(AI) + P(Real) + P(Undecided) = 100.0%.
4. Calibrated threshold decision classification.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from video_detector.config import (
    AI_THRESHOLD_BALANCED,
    AI_THRESHOLD_HIGH,
    DEFAULT_WEIGHT_FLICKER,
    DEFAULT_WEIGHT_FRAME_AI,
    DEFAULT_WEIGHT_WARPING,
    REAL_THRESHOLD,
)


def normalize_percentages(
    ai_val: float,
    real_val: float,
    undecided_val: float,
    min_undecided: float = 4.0,
    decimals: int = 1,
) -> Tuple[float, float, float]:
    """
    Normalizes video percentages to sum exactly to 100.0%.
    """
    ai_clamped = max(0.0, float(ai_val))
    real_clamped = max(0.0, float(real_val))
    u_clamped = max(float(min_undecided), float(undecided_val))

    total = ai_clamped + real_clamped + u_clamped
    if total <= 0.0:
        return 0.0, 0.0, 100.0

    scale = 100.0 / total
    ai_norm = ai_clamped * scale
    real_norm = real_clamped * scale

    ai_pct = round(ai_norm, decimals)
    real_pct = round(real_norm, decimals)
    u_pct = round(max(0.0, 100.0 - (ai_pct + real_pct)), decimals)

    return ai_pct, real_pct, u_pct


def calculate_video_epistemic_uncertainty(prob_ai: float) -> float:
    """Calculates epistemic uncertainty from binary probability using Shannon entropy."""
    p_safe = float(np.clip(prob_ai, 1e-6, 1.0 - 1e-6))
    entropy = -p_safe * math.log2(p_safe) - (1.0 - p_safe) * math.log2(1.0 - p_safe)
    return max(0.0, min(1.0, float(entropy)))


def pool_video_temporal_score(
    mean_frame_ai: float,
    warping_risk: str,
    has_flicker: bool,
    temporal_weights: Optional[Dict[str, float]] = None,
    sensitivity_offset: float = 0.0,
    neural_prob: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Pools frame-level spatial synthesis scores with macro temporal motion variance,
    diffusion flicker cues, and neural transition discriminator probabilities.
    Returns:
        prob_ai: float in [0.01, 0.99]
        prob_real: float in [0.01, 0.99]
    """
    weights = temporal_weights or {}
    w_frame = weights.get("frame_ai_ratio", DEFAULT_WEIGHT_FRAME_AI)
    w_warp = weights.get("motion_warping", DEFAULT_WEIGHT_WARPING)
    w_flick = weights.get("diffusion_flicker", DEFAULT_WEIGHT_FLICKER)
    w_neural = weights.get("neural_temporal", 0.35) if neural_prob is not None else 0.0

    # Normalize weights
    total_w = w_frame + w_warp + w_flick + w_neural
    if total_w > 0:
        w_frame /= total_w
        w_warp /= total_w
        w_flick /= total_w
        w_neural /= total_w

    warping_score = 0.85 if warping_risk in ("HIGH_WARPING_DETECTED", "SUSPICIOUS_FLICKER") else 0.15
    flicker_score = 0.80 if has_flicker else 0.20

    raw_ai_prob = (
        (mean_frame_ai * w_frame)
        + (warping_score * w_warp)
        + (flicker_score * w_flick)
        + ((neural_prob or 0.0) * w_neural)
        + sensitivity_offset
    )
    raw_ai_prob = float(np.clip(raw_ai_prob, 0.01, 0.99))
    raw_real_prob = 1.0 - raw_ai_prob

    return raw_ai_prob, raw_real_prob


def evaluate_video_decision(
    ai_pct: float, real_pct: float, sensitivity: str = "high"
) -> str:
    """Classifies final decision label based on calibrated thresholds."""
    thresh = AI_THRESHOLD_HIGH if sensitivity.lower() in ("high", "aggressive") else AI_THRESHOLD_BALANCED
    if ai_pct >= thresh:
        return "LIKELY AI-GENERATED"
    elif real_pct >= REAL_THRESHOLD:
        return "LIKELY REAL"
    else:
        return "UNDECIDED"
