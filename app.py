from pathlib import Path

from PIL import Image, ImageFile
import streamlit as st

from core.logging_filters import install_benign_reset_filter

install_benign_reset_filter()  # hides the harmless Windows 'connection forcibly closed' asyncio traceback

# Enforce strict parsing and decompression bomb ceiling
from core.security import SAFE_MAX_IMAGE_PIXELS
Image.MAX_IMAGE_PIXELS = SAFE_MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False

from services.forensic_service import ForensicService
from ui.results import (
    render_learning_dashboard,
    render_audio_result,
    render_image_result,
    render_video_result,
)
from ui.media_tab import MediaTabSpec, render_media_tab
from ui.layout import inject_css
from ui.shell import render_sidebar
from ui.validators import validate_expected_platform


DETECTOR_CHECKPOINT = Path(__file__).resolve().parent / "image_detector" / "models" / "ai_detector.pt"

import uuid

from core.atomic_io import get_session_cache_dir

# Per-session scratch space in the OS temp directory (never inside the repository), so one
# user's uploads are neither readable by nor wiped by another user's session.
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex
SESSION_CACHE_DIR = get_session_cache_dir(st.session_state["session_id"])


@st.cache_resource
def get_forensic_service() -> ForensicService:
    return ForensicService.get_instance(checkpoint_path=DETECTOR_CHECKPOINT)


@st.cache_resource
def get_ai_detector():
    return get_forensic_service().image_detector


@st.cache_resource
def get_video_detector():
    return get_forensic_service().video_detector


@st.cache_resource
def get_face_detector():
    return get_forensic_service().face_detector


@st.cache_resource
def get_audio_detector():
    return get_forensic_service().audio_detector


@st.cache_resource
def get_audio_content_analyzer():
    return get_forensic_service().audio_content_analyzer


@st.cache_resource
def get_audio_attribution_engine():
    return get_forensic_service().audio_attribution_engine


@st.cache_resource
def get_content_analyzer():
    return get_forensic_service().content_analyzer


@st.cache_resource
def get_attribution_engine():
    return get_forensic_service().attribution_engine


st.set_page_config(page_title="OmniForensics", page_icon="🔍", layout="wide")

inject_css()
sensitivity_key = render_sidebar(SESSION_CACHE_DIR)

st.title("OmniForensics")
st.caption("Checks whether an image, video or audio file is likely AI-generated or edited, and shows the evidence behind each verdict.")

tab_image, tab_video, tab_audio, tab_url, tab_memory = st.tabs(["Image", "Video", "Audio", "Link check", "Learning"])

detector = get_ai_detector()
face_detector = get_face_detector()
content_analyzer = get_content_analyzer()
attribution_engine = get_attribution_engine()
audio_detector = get_audio_detector()

image_detectors_map = {
    "detector": detector,
    "face_detector": face_detector,
    "content_analyzer": content_analyzer,
    "attribution_engine": attribution_engine,
}
video_detectors_map = {
    "detector": detector,
    "video_detector": get_video_detector(),
    "face_detector": face_detector,
    "content_analyzer": content_analyzer,
    "audio_detector": audio_detector,
    "attribution_engine": attribution_engine,
}
audio_detectors_map = {
    "audio_detector": audio_detector,
    "content_analyzer": get_audio_content_analyzer(),
    "attribution_engine": get_audio_attribution_engine(),
}

MEDIA_TABS = (
    (tab_image, MediaTabSpec(
        key="img", modality="image", noun="Image", file_types=["jpg", "jpeg", "jfif", "png", "webp", "bmp", "tif", "tiff"],
        hint="Looks at pixels, noise, metadata and provenance. JPG, PNG, WebP, BMP or TIFF.",
        url_placeholder="https://example.com/photo.jpg", render_result=render_image_result, tag_upload_source=True,
    ), image_detectors_map),
    (tab_video, MediaTabSpec(
        key="vid", modality="video", noun="Video", file_types=["mp4", "mov", "avi", "mkv", "webm"],
        hint="Looks at frame consistency over time, the audio track and encoder traces. MP4, MOV, AVI, MKV or WebM.",
        url_placeholder="https://example.com/clip.mp4", render_result=render_video_result,
    ), video_detectors_map),
    (tab_audio, MediaTabSpec(
        key="aud", modality="audio", noun="Audio", file_types=["mp3", "wav", "m4a", "aac", "flac", "ogg"],
        hint="Looks at the frequency spectrum, voice delivery and speech timeline. MP3, WAV, M4A, AAC, FLAC or OGG.",
        url_placeholder="https://example.com/voice.mp3", render_result=render_audio_result,
    ), audio_detectors_map),
)
for _tab, _spec, _detectors in MEDIA_TABS:
    with _tab:
        render_media_tab(_spec, _detectors, sensitivity_key, SESSION_CACHE_DIR)

with tab_url:
    st.markdown("Checks that a social-media link is well formed and belongs to the platform you expect. It does not open the link.")
    check_url = st.text_input("Link", placeholder="https://www.instagram.com/reel/...", key="platform_url_input")
    expected_platform = st.selectbox("Expected platform", ["Auto", "Instagram", "YouTube", "Facebook", "TikTok", "X"], key="expected_platform_select")
    if check_url:
        url_res = validate_expected_platform(check_url, expected_platform)
        if not url_res["valid_url"]:
            st.error(url_res["message"])
        else:
            st.write(f"Detected platform: **{url_res['platform']}**")
            if expected_platform == "Auto" or url_res["platform_match"]:
                st.success(url_res["message"])
            else:
                st.error(url_res["message"])
        with st.expander("Raw result"):
            st.json(url_res)

with tab_memory:
    render_learning_dashboard()
