"""ui.results.dimensions: the nine-dimension forensic breakdown.

Every modality's dossier is a dict ``dimension_1 .. dimension_9``, each holding a title, a description, measured fields and a
``diagnosis``. The fields differ by modality, so the page renders whatever the dossier holds instead of looking for named keys:
a field the dossier does not produce can then never be silently missing, and a dimension that was not measured is shown as
"not measured" rather than as a reassuring tick.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st
from ui.text import code_safe, md_escape, neutralise_links

from image_detector.explain import build_nine_dimensions_dossier

_KEYS = tuple(f"dimension_{i}" for i in range(1, 10))
_SKIP = {"title", "description", "dimension_id", "diagnosis"}
_HIDDEN = {"face_bounding_boxes", "top_candidates", "watermark_details"}          # bulky structures; the Details tab holds them

# Boolean fields and what their value means for the diagnosis colour.
_SUSPICIOUS_WHEN_TRUE = {
    "is_brickwalled", "is_unnaturally_smooth", "has_digital_dead_silence", "has_diffusion_flicker", "is_diffusion_smoothed",
    "is_anomalous_decay", "is_digital_art", "is_synthetic_voice", "is_ai_video", "watermark_detected",
}
_SUSPICIOUS_WHEN_FALSE = {"is_natural_shot_noise", "is_natural_noise"}
_RISKY_VALUES = {"warping_risk": {"HIGH_WARPING_DETECTED", "SUSPICIOUS_FLICKER", "UNNATURAL_FREEZE"}}


def _label(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _fmt(value: Any) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_fmt(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        shown = ", ".join(_fmt(v) for v in value[:8])
        return shown + (f" (+{len(value) - 8} more)" if len(value) > 8 else "") if value else "none"
    return code_safe(value)


def _assessment(d: Dict[str, Any]) -> Optional[str]:
    """``"warn"`` when a flag in the dimension is suspicious, ``"ok"`` when every flag present is unremarkable, ``None`` when it
    holds no flag or the measurement is missing."""
    states = []
    for key, value in d.items():
        if key in _SUSPICIOUS_WHEN_TRUE and isinstance(value, bool):
            states.append(value)
        elif key in _SUSPICIOUS_WHEN_FALSE and isinstance(value, bool):
            states.append(not value)
        elif key in _RISKY_VALUES and isinstance(value, str):
            states.append(value in _RISKY_VALUES[key])
    if not states:
        return None
    return "warn" if any(states) else "ok"


def _render_dimension(d: Dict[str, Any], number: int) -> None:
    st.markdown(f"#### {d.get('title') or f'Dimension {number}'}")
    if d.get("description"):
        st.caption(neutralise_links(d["description"]))
    for key, value in d.items():
        if key not in _SKIP and key not in _HIDDEN:
            st.write(f"• **{_label(key)}:** `{_fmt(value)}`")
    diagnosis = d.get("diagnosis")
    if not diagnosis:
        return
    state = _assessment(d)
    if state == "warn":
        st.warning(f"⚠ {neutralise_links(diagnosis)}")
    elif state == "ok":
        st.success(f"✅ {neutralise_links(diagnosis)}")
    else:
        st.info(neutralise_links(diagnosis))


def _tab_label(d: Dict[str, Any], number: int) -> str:
    title = str(d.get("title") or "")
    name = title.split(":", 1)[1].strip() if ":" in title else title
    name = name.split("&")[0].split("(")[0].strip()
    return f"{number}. {name[:24]}" if name else f"{number}"


def render_nine_dimensions_breakdown(nine_dims: Dict[str, Any], expanded: bool = True) -> None:
    """Render the dossier's nine dimensions as tabs."""
    if not nine_dims:
        return

    st.markdown("#### Nine-dimension breakdown")
    st.caption("What was measured in each of nine areas. Heuristic readings, not a certified compliance check.")

    with st.expander("Show all nine dimensions", expanded=expanded):
        dims = [nine_dims.get(key) or {} for key in _KEYS]
        tabs = st.tabs([_tab_label(d, i) for i, d in enumerate(dims, start=1)])
        for i, (tab, d) in enumerate(zip(tabs, dims), start=1):
            with tab:
                _render_dimension(d, i)


def image_nine_dimensions(item: Dict[str, Any], profile_data: Dict[str, Any], ai_result: Dict[str, Any], content_res: Dict[str, Any]) -> Any:
    """The item's precomputed nine-dimension dossier, or one built on demand from its parts."""
    nine_dims = item.get("nine_dimensions_dossier")
    if not nine_dims and profile_data and ai_result:
        nine_dims = build_nine_dimensions_dossier(
            profile_data=profile_data,
            ai_result=ai_result,
            content_inventory=content_res,
            provenance_result=item.get("provenance_res"),
            attribution_result=item.get("attribution_res"),
        )
    return nine_dims
