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
_LEARNERS = {"image": ImageSelfImprover, "video": VideoSelfImprover, "audio": AudioSelfImprover}


def _resolve_feedback_modality(modality: str, media_path: str | Path) -> str:
    """'image' | 'video' | 'audio'. ``"auto"`` reads it from the file extension. Anything else raises: guessing would file a
    correction under the wrong learner and silently skew its calibration."""
    from ui.profile_view import modality_of_extension

    name = str(modality).lower()
    if name in _LEARNERS:
        return name
    if name == "auto":
        found = modality_of_extension(Path(str(media_path)).suffix.lower())
        if found:
            return found
        raise ValueError(f"cannot tell the media type of {Path(str(media_path)).name!r} from its extension")
    raise ValueError(f"unknown modality {modality!r}; expected image, video, audio or auto")


def persist_feedback_media(modality: str, media_path: str | Path) -> Path:
    """Keep a copy of a reviewed file under ``<modality>_detector/data/feedback_media/<sha256><ext>`` and return its path.

    The path an upload arrives with is a scratch file that disappears when the session is cleared or swept, so a correction that only
    pointed at it would turn into a missing file at the next retrain. The copy is named by content (the same bytes are stored once),
    stays on this machine, and lives in the git-ignored ``data`` folder. Files above the size limit are never offered for review."""
    import shutil

    from core.hashing import file_sha256

    src = Path(str(media_path))
    folder = Path(__file__).resolve().parents[2] / f"{modality}_detector" / "data" / "feedback_media"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{file_sha256(src)}{src.suffix.lower()[:10]}"
    if not dest.is_file():
        partial = dest.with_name(dest.name + ".partial")
        shutil.copyfile(src, partial)
        partial.replace(dest)
    return dest


def _submit_feedback(media_path: str | Path, modality: str, forensic_data: Any, truth: str, note: str) -> None:
    """Record the user's answer with the owning modality's learner (the reviewed file is kept in the app's data folder)."""
    metrics = forensic_data if isinstance(forensic_data, dict) else (forensic_data.to_dict() if hasattr(forensic_data, "to_dict") else {})
    resolved = _resolve_feedback_modality(modality, media_path)
    notes = f"[{truth}] {note}".strip()
    kept = persist_feedback_media(resolved, media_path)
    calib = _LEARNERS[resolved]().record_feedback(str(kept), TRUTH_CHOICES[truth], metrics, notes=notes)
    st.success(f"Saved. The {resolved} engine has now learned from {calib.get('samples_processed', 1)} correction(s).")


def render_feedback(media_path: str | Path, modality: str, forensic_data: Any, unique_key: str) -> None:
    """Draw the 'Was this result right?' form and record the answer with the owning learner."""
    st.markdown("**Was this result right?**")
    st.caption(
        "Tell the app what the file really is. A copy of the file is kept in this app's data folder on this machine (it is never uploaded "
        "anywhere) so retraining can use it, and the thresholds are nudged. Corrections are shared by everyone who uses this installation."
    )
    truth = st.radio("This file is", list(TRUTH_CHOICES), index=None, horizontal=True, key=f"{unique_key}_truth")
    note = st.text_input("Note (optional)", key=f"{unique_key}_note", placeholder="What gave it away, or which tool made it")
    if st.button("Save feedback", key=f"{unique_key}_save", disabled=truth is None):
        with st.spinner("Saving"):
            _submit_feedback(media_path, modality, forensic_data, truth, note)


def render_export(media_path: str | Path, modality: str, decision: Dict[str, Any], profile_data: Dict[str, Any],
                  forensic_data: Dict[str, Any], unique_key: str) -> None:
    """Draw the button that downloads the full JSON report."""
    from ui.results.models_panel import cached_manifest

    payload = {"media_file": Path(str(media_path)).name, "modality": modality, "decision": decision,
               "models_and_training_data": cached_manifest(modality), "file_profile": profile_data, "detector_output": forensic_data}
    st.download_button(
        "Download full report (JSON)",
        data=json.dumps(payload, indent=2, default=str),
        file_name=f"report_{Path(media_path).stem}.json",
        mime="application/json",
        key=f"{unique_key}_download",
    )
