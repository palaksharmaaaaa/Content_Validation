"""ui.batch_views: summary and file picker for multi-file runs."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from ui.layout import render_table
from ui.results.summary import verdict_style


def _row(result: Dict[str, Any], modality: str) -> Dict[str, Any]:
    name = result.get("filename", "unknown")
    if not result.get("success"):
        return {"File": name, "Verdict": "Could not process", "AI likelihood": None, "Note": result.get("error", "")}
    if result.get("gate_blocked"):
        return {"File": name, "Verdict": "Blocked by a safety gate: not analysed", "AI likelihood": None, "Note": "see the file's own page"}
    decision = result.get("decision", {})
    probs = decision.get("authenticity_probabilities", {})
    band = (result.get("confidence_band") or {}).get("label", "")
    return {
        "File": name,
        "Verdict": verdict_style(decision)[1],
        "AI likelihood": round(float(probs.get("p_ai", 0.0)), 1),
        "Band": band,
        "Origin guess": (decision.get("model_attribution") or {}).get("attributed_model", ""),
    }


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def csv_safe(table: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``table`` in which no text cell can run as a spreadsheet formula (file names and notes come from uploaded files)."""
    def cell(v: Any) -> Any:
        return "'" + v if isinstance(v, str) and v.startswith(_FORMULA_START) else v
    return table.map(cell) if hasattr(table, "map") else table.applymap(cell)


def render_batch_overview(results: List[Dict[str, Any]], modality: str) -> Optional[Dict[str, Any]]:
    """Counts, a comparison table with downloads, and a picker. Returns the chosen result."""
    ok = [r for r in results if r.get("success")]
    flagged = [r for r in ok if not r.get("gate_blocked") and verdict_style(r.get("decision", {}))[0] in ("error", "warning")]
    a, b, c = st.columns(3)
    a.metric("Files", len(results))
    b.metric("Flagged as AI or edited", len(flagged))
    c.metric("Could not process", len(results) - len(ok))

    table = pd.DataFrame([_row(r, modality) for r in results])
    render_table(table.fillna("").to_dict("records"), percent_bars=["AI likelihood"])
    left, right, _ = st.columns([1, 1, 2])
    left.download_button("Download table (CSV)", csv_safe(table).to_csv(index=False).encode("utf-8"),
                         file_name=f"{modality}_summary.csv", mime="text/csv", key=f"csv_{modality}")
    right.download_button(
        "Download reports (JSON)",
        json.dumps([{"file": r.get("filename"), "decision": r.get("decision")} for r in ok], indent=2, default=str),
        file_name=f"{modality}_reports.json", mime="application/json", key=f"json_{modality}",
    )

    names = [f"{row['File']}: {row['Verdict']}" for row in table.to_dict("records")]
    # Choose by position: two files with the same name and verdict have identical labels, and mapping a label back to its
    # first occurrence would open the wrong file.
    chosen = st.selectbox("Open a file", range(len(names)), format_func=lambda i: names[i], key=f"{modality}_selector")
    return results[chosen] if chosen is not None and chosen < len(results) else None
