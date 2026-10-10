"""ui.results.models_panel: which model and algorithm ran at every step, where it came from and what it was trained on."""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from core.model_registry import manifest
from ui.layout import render_table
from ui.text import md_escape, neutralise_links


@st.cache_data(ttl=300, show_spinner=False)
def cached_manifest(modality: str) -> Dict[str, Any]:
    """The model manifest for ``modality``; cached briefly because reading the checkpoints is not free and the page reruns on every click."""
    return manifest(modality)


def render_models_panel(modality: str, expanded: bool = False) -> None:
    """An expander listing every step with its model, variant, origin (pretrained / fine-tuned / hand-written), training data and licence."""
    info = cached_manifest(modality)
    decider = info["models"][0]
    st.caption(neutralise_links(f"Verdict from: {decider['name']} ({decider['origin']}). {info['summary']}"))        # always visible, the table is one click away
    with st.expander("Models, algorithms and training data", expanded=expanded):
        st.markdown(f"**{md_escape(info['summary'])}**")
        st.caption(f"Training samples counted: {info['training_files_total']:,} (what this project trained a model on, held-out samples included). "
                   f"Facts about each model checked against its publisher on {info['verified_on']}; installation facts are read from the files on this machine.")
        render_table([
            {"Step": m["step"], "Model / algorithm": m["name"], "Variant": m["variant"], "Origin": m["origin"] + ("" if m["active"] else " (not installed)"),
             "Trained on": m["training_data"], "Training size": m["training_size"], "Licence": m["licence"]}
            for m in info["models"]
        ])
        for m in info["models"]:
            if m["note"]:
                st.caption(f"{neutralise_links(m['name'])}: {neutralise_links(m['note'])}")
