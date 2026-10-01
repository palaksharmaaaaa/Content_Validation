"""
audio_detector.scoring: Self-contained scoring and acoustic evidence pooling engine for Audio AI Detection.
Contains:
1. Multi-factor acoustic evidence pooling (vocoder cutoff, Wiener flatness, digital silence, HF roll).
2. Epistemic uncertainty estimation via Shannon entropy.
3. Invariant percentage normalization: P(AI) + P(Real) + P(Undecided) = 100.0%.
4. Calibrated threshold decision classification.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from audio_detector.config import (
    AI_THRESHOLD_BALANCED,
    AI_THRESHOLD_HIGH,
    DEFAULT_WEIGHT_FLATNESS,
    DEFAULT_WEIGHT_HF_RATIO,
    DEFAULT_WEIGHT_SILENCE,
    DEFAULT_WEIGHT_VOCODER,
    DIGITAL_SILENCE_RATIO_THRESHOLD,
    REAL_THRESHOLD,
    SYNTHETIC_FLATNESS_HIGH_THRESHOLD,
    SYNTHETIC_FLATNESS_LOW_THRESHOLD,
)


def normalize_percentages(
    ai_val: float,
    real_val: float,
    undecided_val: float,
    min_undecided: float = 4.0,
    decimals: int = 1,
) -> Tuple[float, float, float]:
    """
    Normalizes audio percentages to sum exactly to 100.0%.
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


def calculate_audio_epistemic_uncertainty(prob_ai: float) -> float:
    """Calculates epistemic uncertainty from binary probability using Shannon entropy."""
    p_safe = float(np.clip(prob_ai, 1e-6, 1.0 - 1e-6))
    entropy = -p_safe * math.log2(p_safe) - (1.0 - p_safe) * math.log2(1.0 - p_safe)
    return max(0.0, min(1.0, float(entropy)))


def pool_acoustic_evidence(
    features: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    neural_prob: Optional[float] = None,
    sensitivity_offset: float = 0.0,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """
    Pools acoustic forensic features with neural synthesizer probability.
    Returns:
        prob_ai: float in [0.01, 0.99]
        prob_real: float in [0.01, 0.99]
    """
    w_cfg = weights or {}
    w_vocoder = w_cfg.get("vocoder_cutoff", DEFAULT_WEIGHT_VOCODER)
    w_flatness = w_cfg.get("spectral_flatness", DEFAULT_WEIGHT_FLATNESS)
    w_silence = w_cfg.get("silence_ratio", DEFAULT_WEIGHT_SILENCE)
    w_hf = w_cfg.get("high_freq_roll", DEFAULT_WEIGHT_HF_RATIO)

    total_w = w_vocoder + w_flatness + w_silence + w_hf
    if total_w > 0:
        w_vocoder /= total_w
        w_flatness /= total_w
        w_silence /= total_w
        w_hf /= total_w

    th_cfg = thresholds or {}
    flatness_low_limit = float(th_cfg.get("flatness_synthetic_max", SYNTHETIC_FLATNESS_LOW_THRESHOLD))
    silence_min_limit = float(th_cfg.get("silence_synthetic_min", DIGITAL_SILENCE_RATIO_THRESHOLD))
    vocoder_min = float(th_cfg.get("vocoder_min_hz", 6500))
    vocoder_max = float(th_cfg.get("vocoder_max_hz", 8200))

    # 1. Vocoder cutoff score
    cutoff_hz = float(features.get("cutoff_freq_hz", 0.0))
    has_vocoder = bool(features.get("has_vocoder_cutoff", False) or (vocoder_min <= cutoff_hz <= vocoder_max))
    vocoder_score = 0.90 if has_vocoder else 0.10

    # 2. Flatness score
    flatness = float(features.get("spectral_flatness", 0.05))
    if flatness < flatness_low_limit:
        flatness_score = 0.85
    elif flatness > SYNTHETIC_FLATNESS_HIGH_THRESHOLD:
        flatness_score = 0.70
    else:
        flatness_score = 0.15

    # 3. Digital silence score
    silence_ratio = float(features.get("digital_silence_ratio", 0.0))
    silence_score = 0.85 if silence_ratio > silence_min_limit else 0.15

    # 4. High-frequency roll score
    hf_ratio = float(features.get("high_freq_ratio", 0.05))
    hf_score = 0.70 if hf_ratio < 0.01 else 0.20

    acoustic_prob = (
        (vocoder_score * w_vocoder)
        + (flatness_score * w_flatness)
        + (silence_score * w_silence)
        + (hf_score * w_hf)
    )

    # Blend with neural classifier probability if present
    if neural_prob is not None:
        final_prob = 0.55 * neural_prob + 0.45 * acoustic_prob + sensitivity_offset
    else:
        final_prob = acoustic_prob + sensitivity_offset

    final_prob = float(np.clip(final_prob, 0.01, 0.99))
    prob_real = 1.0 - final_prob

    return final_prob, prob_real


def evaluate_audio_decision(
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
