"""ui.results.pages: one result page per modality, composed from the sections.

Every page has the same shape so it reads the same for an image, a clip or a recording:

    verdict card                 (summary.py)  what, how strong, why, how far to trust it
    Overview | Evidence | Details | Feedback
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import streamlit as st
from ui.text import code_safe, md_escape, neutralise_links

from ui.results.checks import AUDIO_CHECKS, IMAGE_CHECKS, VIDEO_CHECKS, CheckGroup, render_evidence
from ui.results.content import render_image_type_and_category, render_minor_screening, render_quantified_detections_and_inventory, render_scene_and_content_intelligence
from ui.results.dimensions import image_nine_dimensions, render_nine_dimensions_breakdown
from ui.results.explanation import image_explanation, render_explanation
from ui.results.feedback import render_export, render_feedback
from ui.results.media import render_audio_preview, render_image_preview, render_timeline, render_video_preview
from ui.results.profile import render_audio_signal_specs, render_pre_analysis_specifications, render_video_stream_specs
from ui.results.summary import render_summary
from ui.stages import render_gate_banner


@dataclass(frozen=True)
class _Parts:
    """Everything a page needs, pulled out of a result item (tolerates items from either orchestration path)."""

    item: Dict[str, Any]
    filename: str
    profile: Dict[str, Any]
    result: Dict[str, Any]       # the detector's own output
    content: Dict[str, Any]
    decision: Dict[str, Any]


@dataclass(frozen=True)
class _Page:
    noun: str
    modality: str
    key_prefix: str
    profile_keys: Tuple[str, str]
    result_keys: Tuple[str, str]
    content_keys: Tuple[str, str]
    checks: Sequence[CheckGroup]
    preview: Callable[[_Parts], None]
    timeline: Callable[[_Parts], None]
    details: Callable[[_Parts], None]
    explanation: Callable[[_Parts], Any]


def _first(item: Dict[str, Any], keys: Tuple[str, str]) -> Dict[str, Any]:
    return item.get(keys[0]) or item.get(keys[1]) or {}


def _parts(item: Dict[str, Any], page: _Page) -> _Parts:
    return _Parts(
        item=item, filename=item.get("filename", page.noun.lower()), profile=_first(item, page.profile_keys),
        result=_first(item, page.result_keys), content=_first(item, page.content_keys), decision=item.get("decision") or {},
    )


# ---------------------------------------------------------------- image
def _image_details(p: _Parts) -> None:
    nine = image_nine_dimensions(p.item, p.profile, p.result, p.content)
    render_scene_and_content_intelligence(p.content, modality="image")
    render_pre_analysis_specifications(p.profile, expanded=False, source=p.item.get("source", "User Upload"))
    render_image_type_and_category(p.decision, p.content, p.result, nine_dims=nine)
    render_quantified_detections_and_inventory(p.decision, p.content, p.result, p.profile)
    if nine:
        render_nine_dimensions_breakdown(nine, expanded=False)


IMAGE_PAGE = _Page(
    noun="Image", modality="image", key_prefix="img",
    profile_keys=("img_profile", "file_profile"), result_keys=("ai_result", "ai_detection"),
    content_keys=("content_res", "content_inventory"), checks=IMAGE_CHECKS,
    preview=lambda p: render_image_preview(p.item, p.profile, p.result),
    timeline=lambda p: None,
    details=_image_details,
    explanation=lambda p: image_explanation(p.item, p.profile, p.result, p.content, p.decision, p.filename),
)


# ---------------------------------------------------------------- video / audio
def _stream_details(specs: Callable[[_Parts], None], scene_modality: str) -> Callable[[_Parts], None]:
    def render(p: _Parts) -> None:
        specs(p)
        render_scene_and_content_intelligence(p.content, modality=scene_modality)
        nine = p.item.get("nine_dimensions_dossier")
        if nine:
            render_nine_dimensions_breakdown(nine, expanded=False)

    return render


VIDEO_PAGE = _Page(
    noun="Video", modality="video", key_prefix="vid",
    profile_keys=("vid_profile", "file_profile"), result_keys=("video_result", "temporal_forensics"),
    content_keys=("content_res", "content_inventory"), checks=VIDEO_CHECKS,
    preview=lambda p: render_video_preview(p.item),
    timeline=lambda p: render_timeline(p.result.get("temporal_segments", []), "Timeline"),
    details=_stream_details(lambda p: render_video_stream_specs(p.profile), "video"),
    explanation=lambda p: p.item.get("newbie_explanation"),
)
AUDIO_PAGE = _Page(
    noun="Audio", modality="audio", key_prefix="aud",
    profile_keys=("aud_profile", "file_profile"), result_keys=("audio_result", "acoustic_forensics"),
    content_keys=("content_res", "scene_and_tone"), checks=AUDIO_CHECKS,
    preview=lambda p: render_audio_preview(p.item),
    timeline=lambda p: render_timeline(p.result.get("temporal_segments", []), "Timeline"),
    details=_stream_details(lambda p: render_audio_signal_specs(p.item, p.profile), "audio"),
    explanation=lambda p: p.item.get("newbie_explanation"),
)


def _render_page(item: Dict[str, Any], page: _Page) -> None:
    if not item or not item.get("success"):
        st.error(f"Could not process `{code_safe((item or {}).get('filename', 'this file'))}`: {md_escape((item or {}).get('error', 'unknown error'))}")
        return
    if render_gate_banner(item.get("gate")) and item.get("gate_blocked"):
        return
    p = _parts(item, page)
    render_summary(item, p.filename)
    if page.modality in ("image", "video"):
        render_minor_screening(p.content)

    overview, evidence, details, feedback = st.tabs(["Overview", "Evidence", "Details", "Feedback"])
    with overview:
        page.preview(p)
        page.timeline(p)
        st.markdown("**In plain English**")
        render_explanation(page.explanation(p))
    with evidence:
        render_evidence(item, page.checks)
    with details:
        page.details(p)
    with feedback:
        unique_key = f"{page.key_prefix}_{p.filename}"
        render_feedback(item["path"], page.modality, p.result, unique_key)
        st.divider()
        render_export(item["path"], page.modality, p.decision, p.profile, p.result, unique_key)


def render_image_result(item: Dict[str, Any]) -> None:
    """The full result page for one analysed image."""
    _render_page(item, IMAGE_PAGE)


def render_video_result(item: Dict[str, Any]) -> None:
    """The full result page for one analysed video."""
    _render_page(item, VIDEO_PAGE)


def render_audio_result(item: Dict[str, Any]) -> None:
    """The full result page for one analysed audio file."""
    _render_page(item, AUDIO_PAGE)
