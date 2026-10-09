"""
video_detector.scoring: Self-contained scoring and temporal aggregation engine for Video AI Detection.
Contains:
1. Multi-factor temporal evidence pooling (frame spatial ratio, motion variance, diffusion flicker).
2. Epistemic uncertainty estimation via Shannon entropy.
3. Invariant percentage normalization: P(AI) + P(Real) + P(Undecided) = 100.0%.
4. Calibrated threshold decision classification.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from core.decision import normalize_percentages  # noqa: F401  (re-exported: other modules import it from here)
from core.shared_results import three_way_label, binary_entropy
from video_detector.config import (
    AI_THRESHOLD_BALANCED,
    AI_THRESHOLD_HIGH,
    DEFAULT_WEIGHT_FLICKER,
    DEFAULT_WEIGHT_FRAME_AI,
    DEFAULT_WEIGHT_WARPING,
    REAL_THRESHOLD,
)


def calculate_video_epistemic_uncertainty(prob_ai: float) -> float:
    """Epistemic uncertainty of a binary probability as its Shannon entropy (see core.shared_results.binary_entropy)."""
    return binary_entropy(prob_ai)


def pool_video_temporal_score(
    mean_frame_ai: float,
    warping_risk: Optional[str],
    has_flicker: Optional[bool],
    temporal_weights: Optional[Dict[str, float]] = None,
    sensitivity_offset: float = 0.0,
    neural_prob: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Pools frame-level spatial synthesis scores with macro temporal motion variance,
    diffusion flicker cues, and neural transition discriminator probabilities.

    NOTE: this is a linear weighted average of hand-assigned heuristic scores, not a
    Bayesian log-odds combination -- distinct from `image_detector.scoring.
    pool_bayesian_log_odds` (additive log-LR summation) and `audio_detector.scoring.
    pool_acoustic_evidence` (its own weighted average over different signals). Each
    modality's evidence model operates on different physical signals and is
    intentionally not unified into one shared function; do not assume their outputs
    or internal math are comparable term-for-term.
    Returns:
        prob_ai: float in [0.01, 0.99]
        prob_real: float in [0.01, 0.99]
    """
    weights = temporal_weights or {}
    w_frame = weights.get("frame_ai_ratio", DEFAULT_WEIGHT_FRAME_AI)
    w_warp = weights.get("motion_warping", DEFAULT_WEIGHT_WARPING)
    w_flick = weights.get("diffusion_flicker", DEFAULT_WEIGHT_FLICKER)
    w_neural = weights.get("neural_temporal", 0.35) if neural_prob is not None else 0.0
    # A cue that could not be measured (too few frames) takes no part: its weight goes to the cues that were.
    if warping_risk is None:
        w_warp = 0.0
    if has_flicker is None:
        w_flick = 0.0

    # Normalize weights
    total_w = w_frame + w_warp + w_flick + w_neural
    if total_w > 0:
        w_frame /= total_w
        w_warp /= total_w
        w_flick /= total_w
        w_neural /= total_w

    warping_score = 0.85 if warping_risk in ("HIGH_WARPING_DETECTED", "SUSPICIOUS_FLICKER", "UNNATURAL_FREEZE") else 0.15
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
    ai_pct: float, real_pct: float, sensitivity: str = "balanced"
) -> str:
    """Classifies final decision label based on calibrated thresholds."""
    return three_way_label(ai_pct, real_pct, sensitivity, AI_THRESHOLD_BALANCED, AI_THRESHOLD_HIGH, REAL_THRESHOLD)
