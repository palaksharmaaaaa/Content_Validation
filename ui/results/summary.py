"""ui.results.summary: the verdict card shown first on every result page.

One screen answers: what is the verdict, how strong is it, why, and how far can it be trusted. Everything else on the
page is supporting detail.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from ui.text import code_safe, md_escape, neutralise_links

from ui.stages import band_badge, collect_findings, ood_text

# (taxonomy state, extra final_status values, st level, headline, what it means)
_VERDICTS = (
    ("AUTHENTIC_SCREENSHOT", ("AUTHENTIC SCREENSHOT",), "info", "Authentic screenshot",
     "A genuine screen capture: operating-system or app interface, no generative synthesis detected."),
    ("AI_ENHANCED_SCREENSHOT", ("AI-ENHANCED SCREENSHOT",), "warning", "Screenshot of AI-edited media",
     "A screen capture displaying media modified by AI tools (face swaps, inpainting, upscaling)."),
    ("AI_GENERATED_SCREENSHOT", ("AI-GENERATED SCREENSHOT",), "error", "Screenshot of AI-generated media",
     "A screen capture displaying fully synthetic media."),
    ("AUTHENTIC_EDITED", ("AUTHENTIC (CONVENTIONALLY EDITED)", "AUTHENTIC_EDITED"), "info", "Real photo, conventionally edited",
     "A camera photograph edited with ordinary software (crop, cut-out, graphic design); no generative AI detected."),
    ("AI_ENHANCED_COMPOSITE", ("PARTIALLY_SYNTHETIC_OR_EDITED", "AI-ENHANCED / COMPOSITE"), "warning", "Partly AI-generated or AI-enhanced",
     "A real capture combined with neural processing (face swap, inpainting, AI upscaling) or a mix of real and synthetic content."),
    ("FULLY_AI_GENERATED", ("LIKELY_SYNTHETIC", "LIKELY AI-GENERATED"), "error", "Likely AI-generated",
     "Signals typical of generative models: over-smooth surfaces or spectra, missing sensor noise, generator metadata."),
    ("AUTHENTIC_REAL_PHOTOGRAPH", ("LIKELY_AUTHENTIC", "LIKELY REAL"), "success", "Likely a real capture",
     "The measured signals (noise, spectrum, motion, metadata) look like a recording from the physical world, with no sign of generative processing. An estimate, not proof."),
)
_NO_CONTENT = ("info", "No usable content", "The file has no variation to analyse (a single flat colour, silence or blank frames), so no verdict is given.")
_UNDETERMINED = ("info", "Inconclusive", "The evidence is balanced or the file is heavily compressed. The engine does not guess.")


def verdict_style(decision: Dict[str, Any]) -> Tuple[str, str, str]:
    """(st level, headline, meaning) for a decision dict."""
    state = decision.get("taxonomy_state")
    status = decision.get("final_status", "UNDETERMINED")
    if status == "BLANK_OR_DEGRADED":
        return _NO_CONTENT
    for tax, statuses, level, headline, meaning in _VERDICTS:
        if state == tax or status in (tax, *statuses):
            return level, headline, meaning
    return _UNDETERMINED


def key_reasons(decision: Dict[str, Any], dim_report: Optional[Dict[str, Any]], limit: int = 4) -> List[str]:
    """The few strongest human-readable reasons: classifier reasons first, then checks that were flagged."""
    reasons = [r for r in decision.get("taxonomy_reasons", []) if r]
    for finding in collect_findings(dim_report, ["file_integrity", "metadata", "formats", "container", "signal", "faces"]):
        if finding.get("status") in ("FAIL", "WARN"):
            reasons.append(f"{finding.get('title', 'Check')}: {finding.get('detail', '')}".strip())
    if not reasons:
        reasons = [r for r in decision.get("evidence_trail", []) if r]
    return reasons[:limit]


def render_summary(item: Dict[str, Any], title: str) -> None:
    """Draw the verdict card: category, AI likelihood, evidence strength, main reasons and a trust note."""
    decision = item.get("decision") or {}
    probs = decision.get("authenticity_probabilities", {})
    p_ai, p_real, p_unsure = (float(probs.get(k, d)) for k, d in (("p_ai", 0.0), ("p_real", 0.0), ("p_undecided", 100.0)))
    level, headline, meaning = verdict_style(decision)

    st.subheader(title)
    getattr(st, level)(f"**{headline}**  \n{meaning}")

    c1, c2, c3 = st.columns(3)
    c1.metric("AI likelihood", f"{p_ai:.0f}%")
    c2.metric("Real likelihood", f"{p_real:.0f}%")
    c3.metric("Unsure", f"{p_unsure:.0f}%")
    st.progress(min(1.0, max(0.0, p_ai / 100.0)))
    st.caption("A heuristic estimate that has not been calibrated on labelled data: use it to rank and triage, not as proof.")

    band = item.get("confidence_band")
    reliability = (item.get("dimension_report") or {}).get("reliability") or {}
    if band:
        _emoji, band_label = band_badge(band.get("band", ""))
        st.markdown(f"**Confidence band:** {band_label}")
    for limiter in reliability.get("limiters", []):
        st.warning(limiter)
    ood = item.get("ood") or {}
    if ood.get("status") == "OUT_OF_DISTRIBUTION":
        st.warning(ood_text(ood))
    if (item.get("attribution_open_set") or {}).get("unknown_source"):
        st.warning("Possible new or unknown generator: no known profile matches. " + item["attribution_open_set"].get("note", ""))

    reasons = key_reasons(decision, item.get("dimension_report"))
    if reasons:
        st.markdown("**Why**")
        for reason in reasons:
            st.markdown(f"- {neutralise_links(reason)}")
