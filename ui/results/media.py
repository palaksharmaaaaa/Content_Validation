"""ui.results.media: media previews (with heatmap / spectrogram) and detection timelines."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from ui.layout import render_table


PREVIEW_MAX_SIDE = 1600


def _preview_source(path: str | Path) -> Any:
    """What to hand ``st.image``: the file itself when it is modest, otherwise a copy shrunk to PREVIEW_MAX_SIDE. Streamlit decodes
    and re-encodes whatever it is given, which for a 60-megapixel photograph costs seconds just to draw a thumbnail."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            if max(im.size) <= PREVIEW_MAX_SIDE:
                return str(path)
            preview = im.convert("RGB") if im.mode not in ("RGB", "L") else im.copy()
            preview.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
            return preview
    except Exception:                       # an unusual format Pillow cannot open: let Streamlit try the original
        return str(path)


def render_image_preview(item: Dict[str, Any], profile_data: Dict[str, Any], ai_result: Dict[str, Any]) -> None:
    """Show the image, plus the spatial anomaly heatmap when one exists."""
    left, right = st.columns(2)
    with left:
        st.image(_preview_source(item["path"]), caption=f"Original ({profile_data.get('width', 0)} x {profile_data.get('height', 0)} px)", width="stretch")
    with right:
        heatmap = ai_result.get("heatmap_rgb")
        if heatmap is not None:
            st.image(heatmap, caption=f"Anomaly map (about {ai_result.get('ai_spatial_area_pct', 0.0)}% of the area looks synthetic)", width="stretch")
        else:
            st.caption("No anomaly map for this format.")


def render_video_preview(item: Dict[str, Any]) -> None:
    """Show the video's keyframe."""
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        st.video(item["path"])
    with right:
        keyframe, keyframe_ai = item.get("tmp_kf_path"), item.get("kf_ai")
        if keyframe and Path(keyframe).is_file():
            if keyframe_ai and keyframe_ai.get("heatmap_rgb") is not None:
                st.image(keyframe_ai["heatmap_rgb"], caption=f"Keyframe anomaly map ({keyframe_ai.get('ai_spatial_area_pct', 0)}% flagged)", width="stretch")
            else:
                st.image(keyframe, caption="Sampled keyframe", width="stretch")


def render_audio_preview(item: Dict[str, Any]) -> None:
    """Play the audio file."""
    left, right = st.columns([1, 1.2], gap="medium")
    with left:
        st.audio(item["path"])
    with right:
        if item.get("spec_img") is not None:
            st.image(item["spec_img"], caption="Spectrogram (frequency over time): look for a hard ceiling or unnaturally smooth bands", width="stretch")


def render_timeline(segments: List[Dict[str, Any]], title: str) -> None:
    """Per-segment verdicts along the clip, as a small table."""
    if not segments:
        return
    st.markdown(f"**{title}**")
    rows = []
    for seg in segments:
        label = str(seg.get("label", ""))
        verdict = "AI-like" if "AI" in label else ("Real-like" if "REAL" in label else "Unsure")
        rows.append({"From (s)": seg.get("start_seconds", 0.0), "To (s)": seg.get("end_seconds", 0.0), "Verdict": verdict})
    render_table(rows)
