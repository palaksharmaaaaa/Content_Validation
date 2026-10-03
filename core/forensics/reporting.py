"""core.forensics.reporting: post-detector report pieces shared by the image, audio and video analysis classes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.bands import classify_band_from_percent

UNKNOWN_SOURCE_MIN_AI_PERCENT = 60.0


def summarize_for_evidence_trail(report: Dict[str, Any]) -> List[str]:
    """One-line human strings for FAIL/WARN findings that carry real evidential weight."""
    lines: List[str] = []
    for stage_findings in report.get("findings_by_stage", {}).values():
        for f in stage_findings:
            if f["status"] in ("FAIL", "WARN") and f["evidence_class"] in ("SECURITY", "METADATA_WEAK", "PHYSICAL_SIGNAL"):
                lines.append(f"[{f['evidence_class']}] {f['title']}: {f['detail']}")
    return lines


def add_band_and_open_set(
    out: Dict[str, Any], ai_result: Optional[Dict[str, Any]], attribution: Optional[Dict[str, Any]], medium: str
) -> Dict[str, Any]:
    """Adds ``confidence_band`` and ``attribution_open_set`` to a report dict (mutates and returns it)."""
    ai_pct = (ai_result or {}).get("ai_percentage")
    if ai_pct is not None:
        band = classify_band_from_percent(float(ai_pct))
        out["confidence_band"] = {"band": band.value, "label": band.label, "p_ai_percent": float(ai_pct)}
    else:
        out["confidence_band"] = None
    model_key = (attribution or {}).get("model_key")
    unknown = bool(model_key == "unknown" and ai_pct is not None and float(ai_pct) >= UNKNOWN_SOURCE_MIN_AI_PERCENT)
    out["attribution_open_set"] = {
        "unknown_source": unknown,
        "model_key": model_key,
        "note": (f"AI-leaning {medium} whose generator matches no known profile: UNKNOWN_SOURCE / possible novel generator."
                 if unknown else ""),
    }
    return out
