"""
core.bands: the five operational probability bands a score maps to.

Bands apply to the in-distribution P(AI) score only. OOD, open-set attribution, hard-block and
context channels are independent of the band.
"""
from __future__ import annotations

import math
from enum import Enum


class Band(str, Enum):
    """The five probability bands a score maps to; prefer these over a bare percentage."""
    HIGH_CONFIDENCE_SYNTHETIC = "HIGH_CONFIDENCE_SYNTHETIC"
    LEANING_SYNTHETIC = "LEANING_SYNTHETIC"
    INCONCLUSIVE = "INCONCLUSIVE"
    LEANING_AUTHENTIC = "LEANING_AUTHENTIC"
    HIGH_CONFIDENCE_AUTHENTIC = "HIGH_CONFIDENCE_AUTHENTIC"

    @property
    def label(self) -> str:
        """Human-readable name of the band."""
        return _LABELS[self]


_LABELS = {
    Band.HIGH_CONFIDENCE_SYNTHETIC: "High-Confidence Synthetic / AI-Generated",
    Band.LEANING_SYNTHETIC: "Leaning Synthetic / Flagged for Review",
    Band.INCONCLUSIVE: "Inconclusive / Indeterminate Evidence",
    Band.LEANING_AUTHENTIC: "Leaning Authentic / Low Anomaly",
    Band.HIGH_CONFIDENCE_AUTHENTIC: "High-Confidence Authentic Physical Capture",
}

HIGH_AI = 0.995
UPPER_INCONCLUSIVE = 0.600
LOWER_INCONCLUSIVE = 0.400
HIGH_REAL = 0.005


# The two "high confidence" bands claim a measured error rate. None has been measured on labelled data, so they stay off until
# an operator who has run services.calibration_cli on their own media sets this to True; until then the strongest band is "Leaning".
HIGH_CONFIDENCE_ENABLED = False


def classify_band(p_ai: float) -> Band:
    """``p_ai`` is a probability fraction in [0, 1] (values outside are clamped; NaN is inconclusive)."""
    if p_ai is None or math.isnan(float(p_ai)):
        return Band.INCONCLUSIVE
    p = max(0.0, min(1.0, float(p_ai)))
    if p >= HIGH_AI:
        return Band.HIGH_CONFIDENCE_SYNTHETIC if HIGH_CONFIDENCE_ENABLED else Band.LEANING_SYNTHETIC
    if p <= HIGH_REAL:
        return Band.HIGH_CONFIDENCE_AUTHENTIC if HIGH_CONFIDENCE_ENABLED else Band.LEANING_AUTHENTIC
    if LOWER_INCONCLUSIVE <= p <= UPPER_INCONCLUSIVE:
        return Band.INCONCLUSIVE
    if p > UPPER_INCONCLUSIVE:
        return Band.LEANING_SYNTHETIC
    return Band.LEANING_AUTHENTIC


def classify_band_from_percent(p_ai_percent: float) -> Band:
    """``p_ai_percent`` is a percentage in [0, 100]."""
    return classify_band(float(p_ai_percent) / 100.0)
