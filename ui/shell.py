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
            removed = purge_ephemeral_cache(session_dir)
            # The uploaders still hold their files and the session still lists fetched links: reset both, or the next rerun would write
            # the files straight back to disk.
            for tab in ("img", "vid", "aud"):                      # the keys of the three MediaTabSpec tabs in app.py
                st.session_state[f"{tab}_uploader_round"] = st.session_state.get(f"{tab}_uploader_round", 0) + 1
                for suffix in ("_batch_sig", "_batch_results", "_url_items"):
                    st.session_state.pop(f"{tab}{suffix}", None)
            st.success(f"Removed {removed} temporary file(s) and cleared the file lists.")
    return SENSITIVITY_CHOICES[label]
