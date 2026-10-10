"""ui.results.learning: the Learning tab - what the app has learned from your corrections, and retraining."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from audio_detector import AudioSelfImprover
from image_detector import ImageSelfImprover
from video_detector import VideoSelfImprover

from image_detector.config import SCORE_UNUSED_WEIGHTS
from ui.layout import render_table

_MODALITIES = (("Image", "image", ImageSelfImprover), ("Video", "video", VideoSelfImprover), ("Audio", "audio", AudioSelfImprover))
_UNUSED_WEIGHTS = {"image": SCORE_UNUSED_WEIGHTS}
_WEIGHT_KEYS = {"image": "feature_weights", "video": "temporal_weights", "audio": "acoustic_weights"}


def render_retrain_panel() -> None:
    """Per-modality "Retrain": fine-tunes on registered files and promotes the result only if validation holds."""
    from audio_detector.config import RETRAIN_SUGGEST_THRESHOLD as threshold  # same value in all three configs

    st.markdown("**Retrain**")
    st.caption("Corrected files are queued by reference. Retraining checks the result on held-out files and keeps the old model if accuracy would drop.")
    for col, (label, mod, _learner) in zip(st.columns(3), _MODALITIES):
        with col:
            retrain = importlib.import_module(f"{mod}_detector.retrain")
            pending = retrain.count_pending_corrections()
            st.metric(f"{label}: files queued", pending)
            if pending >= threshold:
                st.info("Enough new files to retrain.")
            if st.button(f"Retrain {label.lower()} model", key=f"retrain_now_{mod}", disabled=pending == 0):
                with st.spinner(f"Retraining the {label.lower()} model"):
                    try:
                        result = retrain.run_retrain(min_new=1)
                    except Exception as exc:  # show the failure, keep the page alive
                        st.error(f"Retraining failed: {exc}")
                    else:
                        (st.success if result.promoted else st.warning)(result.message)


def _records(learner) -> List[Dict[str, Any]]:
    return list(learner.load_memory())


def _recent_table(records: Dict[str, List[Dict[str, Any]]], limit: int = 15) -> List[Dict[str, str]]:
    rows = []
    for modality, recs in records.items():
        for rec in recs:
            path = rec.get("image_path") or rec.get("video_path") or rec.get("audio_path") or rec.get("media_path") or "media"
            rows.append({
                "When": str(rec.get("timestamp", ""))[:19].replace("T", " "),
                "Type": modality, "File": Path(path).name,
                "Label": rec.get("user_label") or rec.get("ground_truth", "?"),
                "Note": str(rec.get("notes") or rec.get("user_notes", "")),
            })
    return sorted(rows, key=lambda r: r["When"], reverse=True)[:limit]


def render_learning_dashboard() -> None:
    """Draw the Learning tab: calibration state per modality and the retrain panel."""
    st.subheader("Learning")
    st.write("Everything the app has learned from your corrections. It starts blank and only changes when you give feedback.")

    learners = {label: learner() for label, _mod, learner in _MODALITIES}
    records = {label: _records(learner) for label, learner in learners.items()}
    total = sum(len(r) for r in records.values())
    ai = sum(1 for recs in records.values() for r in recs if r.get("user_label") == "AI")

    a, b, c, d = st.columns(4)
    a.metric("Corrections saved", total)
    b.metric("Marked AI", ai)
    c.metric("Marked real", sum(1 for recs in records.values() for r in recs if r.get("user_label") == "REAL"))
    d.metric("Images / videos / audio", " / ".join(str(len(records[label])) for label, _m, _l in _MODALITIES))

    st.divider()
    render_retrain_panel()
    st.divider()

    with st.expander("Current adjustable weights"):
        st.caption("Default priors until you give feedback; they shift slightly with each correction.")
        for (label, mod, _learner), tab in zip(_MODALITIES, st.tabs([m[0] for m in _MODALITIES])):
            with tab:
                calib = learners[label].load_calibration()
                weights = calib.get(_WEIGHT_KEYS[mod], {})
                unused = _UNUSED_WEIGHTS.get(mod, ())
                render_table([{"Signal": k.replace("_", " ") + (" (not used in the score)" if k in unused else ""), "Weight": round(v, 3)}
                              for k, v in weights.items()])
                offsets = calib.get("sensitivity_offsets", {})
                if offsets:
                    st.caption("Offsets: " + ", ".join(f"{k.replace('_', ' ')} {v:+.2f}" for k, v in offsets.items()))

    st.markdown("**Recent corrections**")
    rows = _recent_table(records)
    if rows:
        render_table(rows)
    else:
        st.caption("None yet. Open any result, go to the Feedback tab and tell the app what the file really is.")
