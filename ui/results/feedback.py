"""ui.results.feedback: tell the app whether a result was right, and export the full report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from audio_detector import AudioSelfImprover
from image_detector import ImageSelfImprover
from video_detector import VideoSelfImprover

# What the user can say the file really is, and the binary label the learners store.
# "Edited or partly AI" counts as not authentic, so it is recorded with the AI class.
TRUTH_CHOICES = {
    "AI-generated": "AI",
    "Real / authentic": "REAL",
    "Edited or partly AI": "AI",
}
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")
_LEARNERS = {"image": ImageSelfImprover, "video": VideoSelfImprover, "audio": AudioSelfImprover}


def _resolve_feedback_modality(modality: str, media_path: str | Path) -> str:
    """'image' | 'video' | 'audio' (auto-detected from the extension when modality == 'auto')."""
    name = str(media_path).lower()
    if modality == "image" or (modality == "auto" and name.endswith(_IMAGE_SUFFIXES)):
        return "image"
    if modality == "video" or (modality == "auto" and name.endswith(_VIDEO_SUFFIXES)):
        return "video"
    return "audio"


def _submit_feedback(media_path: str | Path, modality: str, forensic_data: Any, truth: str, note: str) -> None:
    """Record the user's answer with the owning modality's learner (queued by reference; the file is never copied)."""
    metrics = forensic_data if isinstance(forensic_data, dict) else (forensic_data.to_dict() if hasattr(forensic_data, "to_dict") else {})
    resolved = _resolve_feedback_modality(modality, media_path)
    notes = f"[{truth}] {note}".strip()
    calib = _LEARNERS[resolved]().record_feedback(str(media_path), TRUTH_CHOICES[truth], metrics, notes=notes)
    st.success(f"Saved. The {resolved} engine has now learned from {calib.get('samples_processed', 1)} correction(s).")


def render_feedback(media_path: str | Path, modality: str, forensic_data: Any, unique_key: str) -> None:
    """Draw the 'Was this result right?' form and record the answer with the owning learner."""
    st.markdown("**Was this result right?**")
    st.caption(
        "Tell the app what the file really is. It remembers the file by reference (nothing is copied or uploaded) and nudges its "
        "thresholds; after enough corrections you can retrain on the Learning tab."
    )
    truth = st.radio("This file is", list(TRUTH_CHOICES), index=None, horizontal=True, key=f"{unique_key}_truth")
    note = st.text_input("Note (optional)", key=f"{unique_key}_note", placeholder="What gave it away, or which tool made it")
    if st.button("Save feedback", key=f"{unique_key}_save", disabled=truth is None):
        with st.spinner("Saving"):
            _submit_feedback(media_path, modality, forensic_data, truth, note)


def render_export(media_path: str | Path, modality: str, decision: Dict[str, Any], profile_data: Dict[str, Any],
                  forensic_data: Dict[str, Any], unique_key: str) -> None:
    """Draw the button that downloads the full JSON report."""
    payload = {"media_file": str(media_path), "modality": modality, "decision": decision,
               "file_profile": profile_data, "detector_output": forensic_data}
    st.download_button(
        "Download full report (JSON)",
        data=json.dumps(payload, indent=2, default=str),
        file_name=f"report_{Path(media_path).stem}.json",
        mime="application/json",
        key=f"{unique_key}_download",
    )
