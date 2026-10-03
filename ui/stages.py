"""
ui.stages: modality-agnostic Streamlit renderers for dimension-check reports.

Pure helpers (status_badge, band_badge, findings_to_rows, collect_findings, ood_text) are
separated from the ``render_*`` functions so they can be unit-tested without a Streamlit runtime.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

_STATUS = {
    "PASS": ("✅", "Pass"),
    "WARN": ("⚠", "Warning"),
    "FAIL": ("❌", "Fail"),
    "INFO": ("ℹ", "Info"),
    "NOT_APPLICABLE": ("–", "N/A"),
    "NOT_CALIBRATED": ("•", "Not calibrated"),
    "RECOGNIZED_OOS": ("ℹ", "Recognized (out of scope)"),
    "ERROR": ("❌", "Error"),
}

_CLASS = {
    "PHYSICAL_SIGNAL": "Physical signal",
    "METADATA_WEAK": "Metadata (weak)",
    "SECURITY": "Security",
    "LEGAL_FLAG": "Legal flag",
    "CONTEXT": "Context",
    "RELIABILITY": "Reliability",
}

_BAND = {
    "HIGH_CONFIDENCE_SYNTHETIC": ("🔴", "Very likely AI-generated"),
    "LEANING_SYNTHETIC": ("🟠", "Leaning AI-generated: review recommended"),
    "INCONCLUSIVE": ("⚪", "Inconclusive"),
    "LEANING_AUTHENTIC": ("🟡", "Leaning authentic: few anomalies"),
    "HIGH_CONFIDENCE_AUTHENTIC": ("🟢", "Very likely a real capture"),
}

_SCORE_ELIGIBLE = {"PHYSICAL_SIGNAL", "METADATA_WEAK"}


def status_badge(status: str) -> Tuple[str, str]:
    return _STATUS.get(status, ("❔", status))


def band_badge(band: str) -> Tuple[str, str]:
    return _BAND.get(band, ("❔", band))


def _weight_text(finding: Dict[str, Any]) -> str:
    cls = finding.get("evidence_class", "")
    llr = finding.get("llr")
    if cls not in _SCORE_ELIGIBLE:
        return "Advisory only (never changes the score)"
    if llr is None:
        return "Weak (no score effect)"
    kind = "physical" if cls == "PHYSICAL_SIGNAL" else "weak"
    return f"{llr:+.2f} log-odds ({kind}, capped)"


def findings_to_rows(findings: Iterable[Dict[str, Any]], show_weight: bool = False) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for f in findings:
        emoji, label = status_badge(f.get("status", ""))
        row = {
            "Status": f"{emoji} {label}",
            "Check": f.get("title", f.get("check_id", "")),
            "Class": _CLASS.get(f.get("evidence_class", ""), f.get("evidence_class", "")),
            "Detail": f.get("detail", ""),
        }
        if show_weight:
            row["Evidence weight"] = _weight_text(f)
        rows.append(row)
    return rows


def collect_findings(report: Optional[Dict[str, Any]], stage_names: Iterable[str]) -> List[Dict[str, Any]]:
    by_stage = (report or {}).get("findings_by_stage", {})
    out: List[Dict[str, Any]] = []
    for name in stage_names:
        out.extend(by_stage.get(name, []))
    return out


def ood_text(ood: Optional[Dict[str, Any]]) -> str:
    if not ood:
        return "OOD status unavailable."
    status = ood.get("status")
    if status == "OUT_OF_DISTRIBUTION":
        return (f"Out-of-distribution: embedding distance {ood.get('distance', 0):.2f} exceeds the calibrated threshold "
                f"{ood.get('threshold', 0):.2f}. Treat the score as unreliable; route to human review.")
    if status == "IN_DISTRIBUTION":
        return f"In-distribution (distance {ood.get('distance', 0):.2f} ≤ threshold {ood.get('threshold', 0):.2f})."
    return "OOD gate not fitted: no reference distribution has been built from your media library yet."


# ---------------------------------------------------------------------------------------------
# Streamlit renderers
# ---------------------------------------------------------------------------------------------
def render_gate_banner(gate: Optional[Dict[str, Any]]) -> bool:
    """Renders a top-of-page banner when a gate fired. Returns True if analysis was blocked/short-circuited."""
    if not gate or not gate.get("triggered"):
        return False
    status = gate.get("final_status")
    if status == "HARD_BLOCK_ESCALATE":
        sha = (gate.get("hard_block") or {}).get("sha256", "")
        st.error(
            "**Hard-blocked: escalate per policy.** This file's SHA-256 matches the operator-supplied hard-block list. "
            "Analysis was stopped and no authenticity verdict was produced. Do not open, share or store the file; follow "
            "your trust-and-safety / legal reporting procedure.\n\n" + (f"`SHA-256: {sha}`" if sha else "")
        )
    else:
        rec = gate.get("recognition") or {}
        st.info(
            f"**Recognized {rec.get('type', 'scientific')} data** ({rec.get('description', '')}). "
            "This format is outside the authenticity-scoring scope of the engine, so no real-vs-AI verdict is produced."
        )
    return True


def render_findings_stage(
    title: str,
    caption: str,
    findings: List[Dict[str, Any]],
    show_weight: bool = False,
    advisory_note: Optional[str] = None,
) -> None:
    """One collapsible group of checks. Opens by itself when something was flagged."""
    flagged = [f for f in findings if f.get("status") in ("FAIL", "WARN")]
    if not findings:
        summary = "no checks applied"
    else:
        summary = f"{len(findings)} checks, " + (f"{len(flagged)} flagged" if flagged else "all clear")
    with st.expander(f"{title}: {summary}", expanded=bool(flagged)):
        st.caption(caption + (f" {advisory_note}" if advisory_note else ""))
        if findings:
            st.dataframe(pd.DataFrame(findings_to_rows(findings, show_weight=show_weight)), hide_index=True, width="stretch")
