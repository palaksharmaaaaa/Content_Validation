import os
from pathlib import Path
import re

from PIL import Image, ImageFile
import streamlit as st

# Enforce strict parsing and decompression bomb ceiling
from core.security import SAFE_MAX_IMAGE_PIXELS
Image.MAX_IMAGE_PIXELS = SAFE_MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False

from core.forensic_service import ForensicService
from audio_detector import AudioAIDetector
from image_detector import (
    FaceDeepfakeDetector,
    ImageAIDetector,
    ImageContentAnalyzer as ContentAnalyzer,
    ImageModelAttributionEngine as ModelAttributionEngine,
    analyze_image,
)
from video_detector import VideoAIDetector, analyze_video, evaluate_cross_modal_consistency
from ui.batch_ui import (
    process_single_audio,
    process_single_image,
    process_single_video,
    render_batch_file_selector,
    render_batch_overview_table,
    render_batch_summary_dashboard,
    run_batch_pipeline,
)
from core.decision import generate_final_decision
from ui.feedback_ui import (
    profile_media,
    render_analysis_right_panel,
    render_bottom_feedback_panel,
    render_feedback_box,
    render_forensic_dossier,
    render_learning_dashboard,
    render_linear_audio_pipeline_results,
    render_linear_image_pipeline_results,
    render_linear_video_pipeline_results,
    render_media_specs,
    render_pre_analysis_specifications,
    render_scene_and_content_intelligence,
)
from ui.validators import (
    analyze_provenance,
    cleanup_url_download,
    fetch_media_from_url,
    validate_expected_platform,
    validate_file,
)


DETECTOR_CHECKPOINT = Path(__file__).resolve().parent / "image_detector" / "models" / "ai_detector.pt"
# App-owned scratch space for transient uploaded-file copies across all three modalities.
# Deliberately NOT nested under image_detector/, audio_detector/, or video_detector/'s own
# data/ directories -- those are each package's own persistent calibration/feedback store,
# not a shared dumping ground for the orchestration layer's temp files.
SESSION_CACHE_DIR = Path(__file__).resolve().parent / "data" / "session_cache"
SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
    "content_analyzer": content_analyzer,
    "attribution_engine": attribution_engine,
}


# =====================================================================
# 1. SEGREGATED IMAGE TAB (SINGLE & BATCH PROCESSING)
# =====================================================================
with tab_image:
    st.subheader("📥 Image Ingestion & Spatial Inspection")
    img_mode = st.radio(
        "Image Input Method",
        ["Upload Image File(s) (Single or Batch)", "Fetch Image from URL(s)"],
        horizontal=True,
        key="img_mode",
    )

    img_items_to_process = []

    if img_mode == "Upload Image File(s) (Single or Batch)":
        uploaded_imgs = st.file_uploader(
            "Upload Image(s)",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
            accept_multiple_files=True,
            key="uploader_img",
            help="Select one or multiple images simultaneously for instant batch forensic evaluation.",
        )
        if uploaded_imgs:
            for idx, u_img in enumerate(uploaded_imgs):
                clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', u_img.name)
                save_path = str(SESSION_CACHE_DIR / f"img_batch_{idx}_{clean_name}")
                with open(save_path, "wb") as f:
                    f.write(u_img.getbuffer())
                    f.flush()
                    os.fsync(f.fileno())
                img_items_to_process.append({
                    "path": save_path,
                    "filename": u_img.name,
                    "size": u_img.size,
                    "source": "Local Device Upload",
                })
    else:
        img_urls_input = st.text_area(
            "Paste Image URL(s) (One per line or comma-separated)",
            placeholder="https://example.com/photo1.jpg\nhttps://example.com/photo2.png",
            key="img_urls_input",
            help="Paste one or multiple direct image URLs for batch downloading and forensic analysis.",
        )
        if st.button("Fetch & Analyze Image(s)", key="btn_fetch_img") and img_urls_input:
            urls = [u.strip() for u in img_urls_input.replace(",", "\n").splitlines() if u.strip().startswith("http")]
            if not urls:
                st.error("Please enter at least one valid URL starting with http:// or https://")
            else:
                for idx, u in enumerate(urls):
                    with st.spinner(f"Downloading image #{idx+1} from {u[:40]}..."):
                        fetch_res = fetch_media_from_url(u, expected_type="image")
                    if fetch_res.get("success"):
                        p = fetch_res["file_path"]
                        img_items_to_process.append({
                            "path": p,
                            "filename": fetch_res["filename"],
                            "size": Path(p).stat().st_size,
                            "source": f"URL Stream ({u[:35]}...)",
                        })
                    else:
                        st.error(f"❌ Failed to download {u}: {fetch_res.get('error')}")

    # Session caching for image batch
    img_batch_sig = tuple((item["filename"], item.get("size", 0), sensitivity_key) for item in img_items_to_process)
    if img_items_to_process:
        if st.session_state.get("img_batch_sig") != img_batch_sig:
            progress_bar = st.progress(0, text="Initializing batch image analysis...")

            def update_img_progress(curr, total, name):
                progress_bar.progress(curr / total, text=f"Analyzing image {curr}/{total}: {name}...")

            img_results = run_batch_pipeline(
                items=img_items_to_process,
                modality="image",
                detectors=image_detectors_map,
                sensitivity=sensitivity_key,
                cache_dir=SESSION_CACHE_DIR,
                progress_callback=update_img_progress,
            )
            progress_bar.empty()
            st.session_state["img_batch_results"] = img_results
            st.session_state["img_batch_sig"] = img_batch_sig
        else:
            img_results = st.session_state.get("img_batch_results", [])
    else:
        img_results = []

    # Display results in completely linear flow
    if img_results:
        is_batch = len(img_results) > 1

        if is_batch:
            render_batch_summary_dashboard(img_results, modality="image")
            render_batch_overview_table(img_results, modality="image")
            selected_img = render_batch_file_selector(img_results, modality="image", key="img_selector")
        else:
            selected_img = img_results[0]

        if selected_img and selected_img.get("success"):
            render_linear_image_pipeline_results(selected_img)
        elif selected_img and not selected_img.get("success"):
            st.error(f"❌ Failed to process `{selected_img['filename']}`: {selected_img.get('error')}")
    else:
        st.info(
            "👈 **Upload an image file (or paste image URLs) in the panel above to begin.**\n\n"
            "• **Completely Linear Flow:** Upload image file ➔ Extract each and every detail (Dimensions, DPI, Pixels, EXIF, Colors, Noise) ➔ "
            "Run 9-Dimensions Forensic Analyzer (As described in GLOBAL_IMAGE_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md) ➔ "
            "Identify Image Type & Category ➔ Predict percentages and counts ➔ Result & Beginner Newbie Narrative Explanation."
        )


# =====================================================================
# 2. SEGREGATED VIDEO TAB (SINGLE & BATCH PROCESSING)
# =====================================================================
with tab_video:
    st.subheader("📥 Video Ingestion & Temporal Inspection")
    vid_mode = st.radio(
        "Video Input Method",
        ["Upload Video File(s) (Single or Batch)", "Fetch Video from URL(s)"],
        horizontal=True,
        key="vid_mode",
    )

    vid_items_to_process = []

    if vid_mode == "Upload Video File(s) (Single or Batch)":
        uploaded_vids = st.file_uploader(
            "Upload Video(s)",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            accept_multiple_files=True,
            key="uploader_vid",
            help="Select one or multiple videos simultaneously for batch temporal and cross-modal evaluation.",
        )
        if uploaded_vids:
            for idx, u_vid in enumerate(uploaded_vids):
                clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', u_vid.name)
                save_path = str(SESSION_CACHE_DIR / f"vid_batch_{idx}_{clean_name}")
                with open(save_path, "wb") as f:
                    f.write(u_vid.getbuffer())
                    f.flush()
                    os.fsync(f.fileno())
                vid_items_to_process.append({
                    "path": save_path,
                    "filename": u_vid.name,
                    "size": u_vid.size,
                })
    else:
        vid_urls_input = st.text_area(
            "Paste Video URL(s) (One per line or comma-separated)",
            placeholder="https://example.com/video1.mp4\nhttps://example.com/video2.mp4",
            key="vid_urls_input",
            help="Paste direct video URLs for batch downloading and forensic evaluation.",
        )
        if st.button("Fetch & Analyze Video(s)", key="btn_fetch_vid") and vid_urls_input:
            urls = [u.strip() for u in vid_urls_input.replace(",", "\n").splitlines() if u.strip().startswith("http")]
            if not urls:
                st.error("Please enter at least one valid URL starting with http:// or https://")
            else:
                for idx, u in enumerate(urls):
                    with st.spinner(f"Downloading video #{idx+1} from {u[:40]}..."):
                        fetch_res = fetch_media_from_url(u, expected_type="video")
                    if fetch_res.get("success"):
                        p = fetch_res["file_path"]
                        vid_items_to_process.append({
                            "path": p,
                            "filename": fetch_res["filename"],
                            "size": Path(p).stat().st_size,
                        })
                    else:
                        st.error(f"❌ Failed to download {u}: {fetch_res.get('error')}")

    # Session caching for video batch
    vid_batch_sig = tuple((item["filename"], item.get("size", 0), sensitivity_key) for item in vid_items_to_process)
    if vid_items_to_process:
        if st.session_state.get("vid_batch_sig") != vid_batch_sig:
            progress_bar = st.progress(0, text="Initializing batch video analysis...")

            def update_vid_progress(curr, total, name):
                progress_bar.progress(curr / total, text=f"Analyzing video {curr}/{total}: {name}...")

            vid_results = run_batch_pipeline(
                items=vid_items_to_process,
                modality="video",
                detectors=video_detectors_map,
                sensitivity=sensitivity_key,
                cache_dir=SESSION_CACHE_DIR,
                progress_callback=update_vid_progress,
            )
            progress_bar.empty()
            st.session_state["vid_batch_results"] = vid_results
            st.session_state["vid_batch_sig"] = vid_batch_sig
        else:
            vid_results = st.session_state.get("vid_batch_results", [])
    else:
        vid_results = []

    # Display video results
    if vid_results:
        is_batch = len(vid_results) > 1

        if is_batch:
            render_batch_summary_dashboard(vid_results, modality="video")
            render_batch_overview_table(vid_results, modality="video")
            selected_vid = render_batch_file_selector(vid_results, modality="video", key="vid_selector")
        else:
            selected_vid = vid_results[0]

        if selected_vid and selected_vid.get("success"):
            render_linear_video_pipeline_results(selected_vid)
        elif selected_vid and not selected_vid.get("success"):
            st.error(f"❌ Failed to process `{selected_vid['filename']}`: {selected_vid.get('error')}")
    else:
        st.info(
            "👈 **Upload one or more videos (or paste video URLs) in the panel above to begin forensic analysis.**\n\n"
            "• **Single Video Mode:** Inspects temporal continuity, keyframe noise, vocoder voice clone sync, and generator signatures.\n"
            "• **Batch Analysis Mode:** Evaluates multiple videos simultaneously with automated comparative risk dashboard and exportable reports."
        )


# =====================================================================
# 3. SEGREGATED AUDIO TAB (SINGLE & BATCH PROCESSING)
# =====================================================================
with tab_audio:
    st.subheader("📥 Audio Ingestion & Spectral Inspection")
    aud_mode = st.radio(
        "Audio Input Method",
        ["Upload Audio File(s) (Single or Batch)", "Fetch Audio from URL(s)"],
        horizontal=True,
        key="aud_mode",
    )

    aud_items_to_process = []

    if aud_mode == "Upload Audio File(s) (Single or Batch)":
        uploaded_auds = st.file_uploader(
            "Upload Audio Recording(s)",
            type=["mp3", "wav", "m4a", "aac", "flac", "ogg"],
            accept_multiple_files=True,
            key="uploader_aud",
            help="Select one or multiple audio recordings simultaneously for batch vocoder and voice synthesis evaluation.",
        )
        if uploaded_auds:
            for idx, u_aud in enumerate(uploaded_auds):
                clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', u_aud.name)
                save_path = str(SESSION_CACHE_DIR / f"aud_batch_{idx}_{clean_name}")
                with open(save_path, "wb") as f:
                    f.write(u_aud.getbuffer())
                    f.flush()
                    os.fsync(f.fileno())
                aud_items_to_process.append({
                    "path": save_path,
                    "filename": u_aud.name,
                    "size": u_aud.size,
                })
    else:
        aud_urls_input = st.text_area(
            "Paste Audio URL(s) (One per line or comma-separated)",
            placeholder="https://example.com/speech1.mp3\nhttps://example.com/voice2.wav",
            key="aud_urls_input",
            help="Paste direct audio URLs for batch downloading and acoustic forensics.",
        )
        if st.button("Fetch & Analyze Audio(s)", key="btn_fetch_aud") and aud_urls_input:
            urls = [u.strip() for u in aud_urls_input.replace(",", "\n").splitlines() if u.strip().startswith("http")]
            if not urls:
                st.error("Please enter at least one valid URL starting with http:// or https://")
            else:
                for idx, u in enumerate(urls):
                    with st.spinner(f"Downloading audio #{idx+1} from {u[:40]}..."):
                        fetch_res = fetch_media_from_url(u, expected_type="audio")
                    if fetch_res.get("success"):
                        p = fetch_res["file_path"]
                        aud_items_to_process.append({
                            "path": p,
                            "filename": fetch_res["filename"],
                            "size": Path(p).stat().st_size,
                        })
                    else:
                        st.error(f"❌ Failed to download {u}: {fetch_res.get('error')}")

    # Session caching for audio batch
    aud_batch_sig = tuple((item["filename"], item.get("size", 0), sensitivity_key) for item in aud_items_to_process)
    if aud_items_to_process:
        if st.session_state.get("aud_batch_sig") != aud_batch_sig:
            progress_bar = st.progress(0, text="Initializing batch audio analysis...")

            def update_aud_progress(curr, total, name):
                progress_bar.progress(curr / total, text=f"Analyzing audio {curr}/{total}: {name}...")

            aud_results = run_batch_pipeline(
                items=aud_items_to_process,
                modality="audio",
                detectors=audio_detectors_map,
                sensitivity=sensitivity_key,
                cache_dir=SESSION_CACHE_DIR,
                progress_callback=update_aud_progress,
            )
            progress_bar.empty()
            st.session_state["aud_batch_results"] = aud_results
            st.session_state["aud_batch_sig"] = aud_batch_sig
        else:
            aud_results = st.session_state.get("aud_batch_results", [])
    else:
        aud_results = []

    # Display audio results
    if aud_results:
        is_batch = len(aud_results) > 1

        if is_batch:
            render_batch_summary_dashboard(aud_results, modality="audio")
            render_batch_overview_table(aud_results, modality="audio")
            selected_aud = render_batch_file_selector(aud_results, modality="audio", key="aud_selector")
        else:
            selected_aud = aud_results[0]

        if selected_aud and selected_aud.get("success"):
            render_linear_audio_pipeline_results(selected_aud)
        elif selected_aud and not selected_aud.get("success"):
            st.error(f"❌ Failed to process `{selected_aud['filename']}`: {selected_aud.get('error')}")
    else:
        st.info(
            "👈 **Upload one or more audio files (or paste audio URLs) in the panel above to begin forensic analysis.**\n\n"
            "• **Single Audio Mode:** Inspects brick-wall vocoder cutoffs, vocal delivery tone, and speech timelines.\n"
            "• **Batch Analysis Mode:** Evaluates multiple audio tracks simultaneously with batch comparison metrics and export options."
        )


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