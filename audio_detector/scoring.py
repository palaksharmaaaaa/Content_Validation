"""
audio_detector.scoring: Self-contained scoring and acoustic evidence pooling engine for Audio AI Detection.
Contains:
1. Multi-factor acoustic evidence pooling (vocoder cutoff, Wiener flatness, digital silence, HF roll).
2. Epistemic uncertainty estimation via Shannon entropy.
3. Invariant percentage normalization: P(AI) + P(Real) + P(Undecided) = 100.0%.
4. Calibrated threshold decision classification.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.decision import normalize_percentages  # noqa: F401  (re-exported: audio_detector.detector and the package API import it from here)
from core.shared_results import three_way_label, binary_entropy
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


def calculate_audio_epistemic_uncertainty(prob_ai: float) -> float:
    """Epistemic uncertainty of a binary probability as its Shannon entropy (see core.shared_results.binary_entropy)."""
    return binary_entropy(prob_ai)


def pool_acoustic_evidence(
    features: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    neural_prob: Optional[float] = None,
    sensitivity_offset: float = 0.0,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """
    Pools acoustic forensic features with neural synthesizer probability.

    NOTE: this is a linear weighted average of hand-assigned heuristic scores, not a
    Bayesian log-odds combination -- distinct from `image_detector.scoring.
    pool_bayesian_log_odds` (additive log-LR summation) and `video_detector.scoring.
    pool_video_temporal_score` (its own weighted average over different signals).
    Each modality's evidence model operates on different physical signals and is
    intentionally not unified into one shared function; do not assume their outputs
    or internal math are comparable term-for-term.
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

    # 1. Vocoder cutoff score
    # features.compute_spectral_features decides this once, and only for a cutoff clearly below the recording's own Nyquist; a cutoff
    # frequency alone is not re-tested here (a plain 16 kHz recording rolls off at 7.5-8 kHz without any vocoder).
    has_vocoder = bool(features.get("has_vocoder_cutoff", False))
    vocoder_score = 0.90 if has_vocoder else 0.10

    # 2. Flatness score
    flatness = float(features.get("spectral_flatness", 0.05))
    if flatness < flatness_low_limit:
        flatness_score = 0.85
    elif flatness > SYNTHETIC_FLATNESS_HIGH_THRESHOLD:
        flatness_score = 0.15          # a noise-like spectrum is what any hissy recording has; it is no evidence of synthesis
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
    ai_pct: float, real_pct: float, sensitivity: str = "balanced"
) -> str:
    """Classifies final decision label based on calibrated thresholds."""
    return three_way_label(ai_pct, real_pct, sensitivity, AI_THRESHOLD_BALANCED, AI_THRESHOLD_HIGH, REAL_THRESHOLD)
