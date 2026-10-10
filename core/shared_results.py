"""core.shared_results: result shapes and label rules shared by all three modality packages."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

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
    extra_total = sum(float(v) for v in (extra_log_lrs or {}).values() if v and math.isfinite(float(v)))
    extra_total = max(-1.0, min(1.0, extra_total))          # the dimension checks are capped upstream; this keeps the maths safe regardless
    if not extra_total:
        return p_ai, 1.0 - p_ai, []
    p_clip = float(np.clip(p_ai, 0.01, 0.99))
    logit = float(np.log10(p_clip / (1.0 - p_clip))) + extra_total
    shifted = float(np.clip(1.0 / (1.0 + 10.0 ** (-logit)), 0.01, 0.99))
    cues = [f"Dimension check '{k}' adjusted log-odds by {float(v):+.2f}" for k, v in (extra_log_lrs or {}).items() if v]
    return shifted, 1.0 - shifted, cues


def binary_entropy(prob_ai: float) -> float:
    """Shannon entropy (bits) of a two-outcome probability: 1.0 at P(AI) = 0.5, near 0 when the probability is near 0 or 1."""
    p = float(np.clip(prob_ai, 1e-6, 1.0 - 1e-6))
    return max(0.0, min(1.0, float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))))


NAMING_THRESHOLD = 0.25


def _share(value: float) -> float:
    """A share rounded to two decimals. A share that is exactly halfway (0.125) can come out a hair either side of it depending on the
    library versions that produced it, so the last bits are absorbed first: the same file always reports the same number."""
    return round(float(value) + 1e-9, 2)


def finalize_attribution(
    scores: Dict[str, float],
    catalog: Dict[str, Dict[str, Any]],
    *,
    declared: bool,
    cues: List[str],
    unknown_name: str,
    no_cue_text: str,
    region_of: Any,
    watermark_detected: bool = False,
    declared_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """The shared last step of every attribution engine.

    A generator is named only when the file itself declares one (a visible watermark, a metadata or software claim). Signal statistics
    (spectral slope, motion variance, a cutoff frequency, a canvas size) are shared by many generators and by ordinary cameras and
    codecs, so they can rank candidates but never put a product name on a file by themselves; without a declaration the answer is
    "unknown" and the candidate list is withheld rather than offered as a guess. ``declared_keys`` are the catalogue keys the file
    itself declared: only one of those can be named, even when signal statistics put another generator's score higher.
    """
    total = sum(scores.values())
    ranked = sorted(((k, v / total if total > 0 else v) for k, v in scores.items()), key=lambda kv: kv[1], reverse=True)
    top3 = [{"model": catalog[k]["name"], "confidence": _share(s)} for k, s in ranked[:3]]
    pool = [kv for kv in ranked if declared_keys is None or kv[0] in declared_keys] or ranked
    best_key, best_score = pool[0]
    if not declared or best_score < NAMING_THRESHOLD:
        return {
            "attributed_model": unknown_name,
            "model_key": "unknown",
            "confidence": _share(best_score) if declared else 0.0,
            "cues": cues or [no_cue_text],
            "top_candidates": top3 if declared else [],
        }
    info = catalog[best_key]
    conf = _share(best_score)
    return {
        "attributed_model": info["name"],
        "model_key": best_key,
        "provider": info["provider"],
        "confidence": conf,
        "attribution_confidence": conf,
        "region_of_origin": region_of(best_key),
        "watermark_detected": watermark_detected,
        "cues": cues or [f"The file declares {info['name']}."],
        "top_candidates": top3,
    }
