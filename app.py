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
from ui.feedback_ui import render_learning_dashboard, render_linear_audio_pipeline_results, render_linear_image_pipeline_results, render_linear_video_pipeline_results
from ui.media_tab import MediaTabSpec, render_media_tab
from ui.validators import validate_expected_platform


DETECTOR_CHECKPOINT = Path(__file__).resolve().parent / "image_detector" / "models" / "ai_detector.pt"

import uuid

from core.atomic_io import get_session_cache_dir, purge_ephemeral_cache

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


st.set_page_config(
    page_title="OmniForensics: Multi-Modal AI & Media Validator",
    page_icon="🔍",
    layout="wide",
)

# Sidebar Configuration
st.sidebar.header("⚙️ Forensic Engine Settings")
sensitivity_option = st.sidebar.selectbox(
    "Detection Sensitivity Mode",
    [
        "Balanced (Recommended -- neutral prior, no AI/real bias before evidence)",
        "High Sensitivity (For Modern AI -- biases toward flagging AI)",
        "Aggressive (Maximum Forensic Scrutiny)",
    ],
    index=0,
)

if "Aggressive" in sensitivity_option:
    sensitivity_key = "aggressive"
elif "High" in sensitivity_option:
    sensitivity_key = "high"
else:
    sensitivity_key = "balanced"

st.sidebar.info(
    f"Active Mode: **{sensitivity_key.upper()}**\n\n"
    "• Balanced (default): No built-in prior toward AI or real before any evidence is evaluated -- the honest starting point for general use.\n"
    "• High: Shifts the prior toward AI, tightening PRNU, spectral slope, and vocoder cutoff thresholds to catch subtle modern generators (Gemini, Midjourney, Flux, Sora, ElevenLabs) at the cost of more false positives on ambiguous real images.\n"
    "• Aggressive: Maximizes scrutiny against compressed social media reposts; highest false-positive risk."
)

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Zero-Disk Storage Manager")
if st.sidebar.button("🧹 Wipe Transient Media Cache"):
    purged = purge_ephemeral_cache(SESSION_CACHE_DIR)
    st.sidebar.success(f"Wiped {purged} transient media file(s) (0 bytes on disk)!")

st.title("🔍 OmniForensics: Multi-Modal Media Authenticity Engine")
st.write(
    "NIST-aligned multi-modal media forensics architecture: "
    "**File Forensics → Provenance & C2PA → Media Quality → Parallel Content Analysis → AI Forensics → Localization → Cross-Modal Sync → Continuous Learning**."
)

tab_image, tab_video, tab_audio, tab_url, tab_memory = st.tabs([
    "🖼️ Image Validation",
    "🎥 Video Validation",
    "🎙️ Audio Validation",
    "🔗 Social URL Checker",
    "🧠 Continuous Learning & Memory",
])

detector = get_ai_detector()
video_detector = get_video_detector()
face_detector = get_face_detector()
audio_detector = get_audio_detector()
content_analyzer = get_content_analyzer()
attribution_engine = get_attribution_engine()

image_detectors_map = {
    "detector": detector,
    "face_detector": face_detector,
    "content_analyzer": content_analyzer,
    "attribution_engine": attribution_engine,
}

video_detectors_map = {
    "detector": detector,
    "video_detector": video_detector,
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


# =====================================================================
# 1-3. IMAGE / VIDEO / AUDIO TABS (one shared implementation, see ui/media_tab.py)
# =====================================================================
MEDIA_TABS = (
    (tab_image, MediaTabSpec(
        key="img", modality="image", noun="Image",
        subheader="📥 Image Ingestion & Spatial Inspection",
        uploader_label="Upload Image(s)", file_types=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        uploader_help="Select one or multiple images simultaneously for instant batch forensic evaluation.",
        url_placeholder="https://example.com/photo1.jpg\nhttps://example.com/photo2.png",
        url_help="Paste one or multiple direct image URLs for batch downloading and forensic analysis.",
        empty_info=(
            "👈 **Upload an image file (or paste image URLs) in the panel above to begin.**\n\n"
            "• **Completely Linear Flow:** Upload image file ➔ Extract each and every detail (Dimensions, DPI, Pixels, EXIF, Colors, Noise) ➔ "
            "Run 9-Dimensions Forensic Analyzer (As described in GLOBAL_IMAGE_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md) ➔ "
            "Identify Image Type & Category ➔ Predict percentages and counts ➔ Result & Beginner Newbie Narrative Explanation."
        ),
        render_result=render_linear_image_pipeline_results, tag_upload_source=True,
    ), image_detectors_map),
    (tab_video, MediaTabSpec(
        key="vid", modality="video", noun="Video",
        subheader="📥 Video Ingestion & Temporal Inspection",
        uploader_label="Upload Video(s)", file_types=["mp4", "mov", "avi", "mkv", "webm"],
        uploader_help="Select one or multiple videos simultaneously for batch temporal and cross-modal evaluation.",
        url_placeholder="https://example.com/video1.mp4\nhttps://example.com/video2.mp4",
        url_help="Paste direct video URLs for batch downloading and forensic evaluation.",
        empty_info=(
            "👈 **Upload one or more videos (or paste video URLs) in the panel above to begin forensic analysis.**\n\n"
            "• **Single Video Mode:** Inspects temporal continuity, keyframe noise, vocoder voice clone sync, and generator signatures.\n"
            "• **Batch Analysis Mode:** Evaluates multiple videos simultaneously with automated comparative risk dashboard and exportable reports."
        ),
        render_result=render_linear_video_pipeline_results,
    ), video_detectors_map),
    (tab_audio, MediaTabSpec(
        key="aud", modality="audio", noun="Audio",
        subheader="📥 Audio Ingestion & Spectral Inspection",
        uploader_label="Upload Audio Recording(s)", file_types=["mp3", "wav", "m4a", "aac", "flac", "ogg"],
        uploader_help="Select one or multiple audio recordings simultaneously for batch vocoder and voice synthesis evaluation.",
        url_placeholder="https://example.com/speech1.mp3\nhttps://example.com/voice2.wav",
        url_help="Paste direct audio URLs for batch downloading and acoustic forensics.",
        empty_info=(
            "👈 **Upload one or more audio files (or paste audio URLs) in the panel above to begin forensic analysis.**\n\n"
            "• **Single Audio Mode:** Inspects brick-wall vocoder cutoffs, vocal delivery tone, and speech timelines.\n"
            "• **Batch Analysis Mode:** Evaluates multiple audio tracks simultaneously with batch comparison metrics and export options."
        ),
        render_result=render_linear_audio_pipeline_results,
    ), audio_detectors_map),
)
for _tab, _spec, _detectors in MEDIA_TABS:
    with _tab:
        render_media_tab(_spec, _detectors, sensitivity_key, SESSION_CACHE_DIR)


# =====================================================================
# 4. SOCIAL PLATFORM & URL CHECKER TAB
# =====================================================================
with tab_url:
    st.subheader("🔗 Social Platform & URL Link Checker")
    st.write("Inspect social media links (Instagram, YouTube, TikTok, Facebook, X) for domain authenticity and format integrity.")

    check_url = st.text_input("Enter Social Media Link", placeholder="https://www.instagram.com/reel/...", key="platform_url_input")
    expected_platform = st.selectbox("Expected Platform", ["Auto", "Instagram", "YouTube", "Facebook", "TikTok", "X"], key="expected_platform_select")

    if check_url:
        url_res = validate_expected_platform(check_url, expected_platform)
        if not url_res["valid_url"]:
            st.error(f"❌ {url_res['message']}")
        else:
            st.write(f"Detected Platform: **{url_res['platform']}**")
            if expected_platform == "Auto" or url_res["platform_match"]:
                st.success(f"✅ {url_res['message']}")
            else:
                st.error(url_res["message"])

        st.json(url_res)


# =====================================================================
# 5. CONTINUOUS LEARNING & FORENSIC MEMORY BANK TAB
# =====================================================================
with tab_memory:
    render_learning_dashboard()