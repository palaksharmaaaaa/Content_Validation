"""ui.results.feedback: tell the app whether a result was right, and export the full report."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from audio_detector import AudioSelfImprover
from image_detector import ImageSelfImprover
from video_detector import VideoSelfImprover

# What the user can say the file really is, and the binary label the learners store.
# "Edited or partly AI" counts as not authentic, so it is recorded with the AI class.
logger = logging.getLogger(__name__)

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


def evaluate_feedback_label_quality(truth: str, forensic_data: Any) -> dict[str, Any]:
    """Uses Cleanlab confident learning to evaluate whether user-submitted feedback is consistent with detector probabilities or suspect label noise."""
    metrics = forensic_data if isinstance(forensic_data, dict) else (forensic_data.to_dict() if hasattr(forensic_data, "to_dict") else {})

    # Extract probability of AI vs Real
    p_ai: float | None = None
    if "ai_percentage" in metrics:
        p_ai = float(metrics["ai_percentage"]) / 100.0
    elif "ai_probability" in metrics:
        p_ai = float(metrics["ai_probability"])
    elif "probability" in metrics:
        p_ai = float(metrics["probability"])

    if p_ai is None or not (0.0 <= p_ai <= 1.0):
        return {"quality_score": 1.0, "is_suspect_noise": False, "reason": "No probabilistic cues to assess"}

    p_real = 1.0 - p_ai
    user_label = 1 if TRUTH_CHOICES.get(truth) == "AI" else 0

    # Label quality here is the detector's own probability for the label the reviewer chose (cleanlab's default "self_confidence" score
    # for one example). Cleanlab computes it when it is installed; the same number is computed directly when it is not, so the guard
    # below never silently switches off.
    score = p_ai if user_label == 1 else p_real
    source = "directly"
    try:
        import numpy as np
        from cleanlab.rank import get_label_quality_scores

        score = float(get_label_quality_scores(labels=np.array([user_label], dtype=np.int64),
                                               pred_probs=np.array([[p_real, p_ai]], dtype=np.float64))[0])
        source = "by cleanlab"
    except Exception as exc:                       # not installed or failed: keep the direct value
        logger.debug("cleanlab label-quality score unavailable (%s: %s); using the direct value", type(exc).__name__, exc)
    is_suspect = bool(score < 0.20 and max(p_ai, p_real) >= 0.80)
    return {
        "quality_score": round(score, 3),
        "is_suspect_noise": is_suspect,
        "p_ai": round(p_ai, 3),
        "p_real": round(p_real, 3),
        "scored": source,
        "reason": (
            f"The detector is {max(p_ai, p_real):.0%} sure the other way (label quality score {score:.2f}, {source}): "
            f"it saw {p_ai:.1%} AI, you chose '{truth}'."
            if is_suspect
            else "Feedback label is consistent with forensic model indicators."
        ),
    }


def _submit_feedback(media_path: str | Path, modality: str, forensic_data: Any, truth: str, note: str, confirmed: bool = False) -> bool:
    """Record the user's answer with the owning modality's learner (the reviewed file is kept in the app's data folder).

    An answer that contradicts a very confident detector result is not applied until the reviewer confirms it (``confirmed=True``): the
    detector may be the one that is wrong, which is exactly when a correction is most valuable, but one mistaken click should not move
    the thresholds. Returns True if the correction was recorded."""
    metrics = forensic_data if isinstance(forensic_data, dict) else (forensic_data.to_dict() if hasattr(forensic_data, "to_dict") else {})
    resolved = _resolve_feedback_modality(modality, media_path)
    notes = f"[{truth}] {note}".strip()

    quality_info = evaluate_feedback_label_quality(truth, forensic_data)
    suspect = bool(quality_info.get("is_suspect_noise"))
    if suspect and not confirmed:
        st.warning(f"Not saved yet. {quality_info['reason']} If you are sure, tick the confirmation box and save again.")
        return False
    if suspect:
        notes = f"{notes} [confirmed against a confident detector result, label quality score {quality_info['quality_score']}]".strip()

    kept = persist_feedback_media(resolved, media_path)
    calib = _LEARNERS[resolved]().record_feedback(str(kept), TRUTH_CHOICES[truth], metrics, notes=notes)

    st.success(f"Saved. The {resolved} engine has now learned from {calib.get('samples_processed', 1)} correction(s).")
    return True


def render_feedback(media_path: str | Path, modality: str, forensic_data: Any, unique_key: str) -> None:
    """Draw the 'Was this result right?' form and record the answer with the owning learner."""
    st.markdown("**Was this result right?**")
    st.caption(
        "Tell the app what the file really is. A copy of the file is kept in this app's data folder on this machine (it is never uploaded "
        "anywhere) so retraining can use it, and the thresholds are nudged. Corrections are shared by everyone who uses this installation."
    )
    truth = st.radio("This file is", list(TRUTH_CHOICES), index=None, horizontal=True, key=f"{unique_key}_truth")
    note = st.text_input("Note (optional)", key=f"{unique_key}_note", placeholder="What gave it away, or which tool made it")
    confirmed = False
    if truth is not None and evaluate_feedback_label_quality(truth, forensic_data).get("is_suspect_noise"):
        confirmed = st.checkbox(
            "I am sure: apply this correction although the detector is very confident the other way",
            key=f"{unique_key}_confirm",
            help="Corrections that contradict a confident result are held back until you confirm them, so one mistaken click cannot move the thresholds.",
        )
    if st.button("Save feedback", key=f"{unique_key}_save", disabled=truth is None):
        with st.spinner("Saving"):
            _submit_feedback(media_path, modality, forensic_data, truth, note, confirmed=confirmed)


def render_export(media_path: str | Path, modality: str, decision: dict[str, Any], profile_data: dict[str, Any],
                  forensic_data: dict[str, Any], unique_key: str) -> None:
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
