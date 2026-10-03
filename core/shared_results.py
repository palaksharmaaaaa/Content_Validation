"""core.shared_results: result shapes and label rules shared by all three modality packages."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def unknown_attribution(reason: str) -> Dict[str, Any]:
    """The canonical "no generator attributed" result (identical across image, audio and video)."""
    return {
        "attributed_model": "Unknown",
        "model_key": "unknown",
        "confidence": 0.0,
        "attribution_confidence": 0.0,
        "region_of_origin": "Unknown",
        "watermark_detected": False,
        "cues": [reason],
        "top_candidates": [],
    }


def three_way_label(ai_pct: float, real_pct: float, sensitivity: str, ai_balanced: float, ai_high: float, real_threshold: float) -> str:
    """AI / REAL / UNDECIDED label from percentage thresholds; high/aggressive sensitivity uses the lower AI bar."""
    thresh = ai_high if sensitivity.lower() in ("high", "aggressive") else ai_balanced
    if ai_pct >= thresh:
        return "LIKELY AI-GENERATED"
    if real_pct >= real_threshold:
        return "LIKELY REAL"
    return "UNDECIDED"


def shift_probability_by_log_odds(p_ai: float, extra_log_lrs: Optional[Dict[str, float]]) -> Tuple[float, float, List[str]]:
    """Applies capped dimension-check terms to a pooled probability in base-10 log-odds space.

    Audio and video pool by weighted average, so extra evidence cannot be added to a posterior; it shifts the
    logit instead and the result is re-clipped to [0.01, 0.99]. Empty/zero input returns the inputs unchanged.
    Returns (p_ai, p_real, explanatory cues).
    """
    extra_total = sum(float(v) for v in (extra_log_lrs or {}).values() if v)
    if not extra_total:
        return p_ai, 1.0 - p_ai, []
    p_clip = float(np.clip(p_ai, 0.01, 0.99))
    logit = float(np.log10(p_clip / (1.0 - p_clip))) + extra_total
    shifted = float(np.clip(1.0 / (1.0 + 10.0 ** (-logit)), 0.01, 0.99))
    cues = [f"Dimension check '{k}' adjusted log-odds by {float(v):+.2f}" for k, v in (extra_log_lrs or {}).items() if v]
    return shifted, 1.0 - shifted, cues
