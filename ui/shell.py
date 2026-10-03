"""ui.shell: the sidebar (sensitivity, system status, cache wipe)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.atomic_io import purge_ephemeral_cache
from services.system_status import collect_status

SENSITIVITY_CHOICES = {
    "Balanced (recommended)": "balanced",
    "High sensitivity": "high",
    "Aggressive": "aggressive",
}
_SENSITIVITY_HELP = (
    "Balanced: no built-in lean toward AI or real before evidence is weighed.\n\n"
    "High: leans toward AI and tightens thresholds to catch subtle modern generators; more false alarms on ambiguous real media.\n\n"
    "Aggressive: maximum scrutiny, intended for heavily compressed reposts; highest false-alarm risk."
)
_ICON = {"ok": "✅", "info": "ℹ️", "warn": "⚠️"}


def render_sidebar(session_dir: Path) -> str:
    """Draw the sidebar and return the chosen sensitivity key ("balanced" | "high" | "aggressive")."""
    with st.sidebar:
        st.header("Settings")
        label = st.selectbox("Sensitivity", list(SENSITIVITY_CHOICES), index=0, help=_SENSITIVITY_HELP)

        with st.expander("System status"):
            for row in collect_status():
                st.markdown(f"{_ICON.get(row.level, '')} **{row.component}**: {row.state}")
                if row.hint:
                    st.caption(row.hint)

        st.divider()
        if st.button("Clear uploaded files", help="Deletes this session's temporary copies of uploaded and downloaded media."):
            st.success(f"Removed {purge_ephemeral_cache(session_dir)} temporary file(s).")
    return SENSITIVITY_CHOICES[label]
