import os
from pathlib import Path
import shutil
import tempfile

import cv2
import streamlit as st

from models.ai_audio_detector import AIAudioDetector
from models.ai_image_detector import AIImageDetector
from models.content_analyzer import ContentAnalyzer
from models.face_detector import FaceDeepfakeDetector
from models.model_attribution import ModelAttributionEngine
from scoring.cross_modal_engine import evaluate_cross_modal_consistency
from scoring.decision_engine import generate_final_decision
from ui.feedback_ui import (
    render_analysis_right_panel,
    render_bottom_feedback_panel,
    render_feedback_box,
    render_forensic_dossier,
    render_learning_dashboard,
    render_media_specs,
    render_scene_and_content_intelligence,
)
from utils.media_profiler import profile_media
from validators.file_validator import validate_file
from validators.image_validator import analyze_image
from validators.provenance_validator import analyze_provenance
from validators.url_validator import (
    fetch_media_from_url,
    validate_expected_platform,
)
from validators.video_validator import analyze_video


DETECTOR_CHECKPOINT = Path(__file__).resolve().parent / "models" / "ai_detector.pt"
SESSION_CACHE_DIR = Path(__file__).resolve().parent / "data" / "session_cache"
SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_ai_detector():
    detector = AIImageDetector(checkpoint_path=DETECTOR_CHECKPOINT)
    detector.load()
    return detector


@st.cache_resource
def get_face_detector():
    return FaceDeepfakeDetector()


@st.cache_resource
def get_audio_detector():
    return AIAudioDetector()


@st.cache_resource
def get_content_analyzer():
    return ContentAnalyzer()


@st.cache_resource
def get_attribution_engine():
    return ModelAttributionEngine()


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
        "High Sensitivity (Recommended for Modern AI)",
        "Balanced (Standard)",
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
    "• High: Tightens PRNU, spectral slope, and vocoder cutoff thresholds to detect modern subtle generators (Gemini, Midjourney, Flux, Sora, ElevenLabs).\n"
    "• Aggressive: Maximizes scrutiny against compressed social media reposts."
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
face_detector = get_face_detector()
audio_detector = get_audio_detector()
content_analyzer = get_content_analyzer()
attribution_engine = get_attribution_engine()


# =====================================================================
# 1. SEGREGATED IMAGE TAB
# =====================================================================
with tab_image:
    col_img_left, col_img_right = st.columns([1, 1.2], gap="large")

    img_path = None
    img_file_name = None
    file_res = None
    image_result = None
    ai_result = None
    content_res = None
    provenance_res = None
    attribution_res = None
    decision = None
    img_profile = None

    with col_img_left:
        st.subheader("📥 Image Ingestion & Spatial Inspection")
        img_mode = st.radio("Image Input Method", ["Upload Image File", "Fetch Image from URL"], horizontal=True, key="img_mode")

        if img_mode == "Upload Image File":
            uploaded_img = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
                key="uploader_img",
            )
            if uploaded_img:
                suffix = Path(uploaded_img.name).suffix or ".jpg"
                img_path = str(SESSION_CACHE_DIR / f"active_image{suffix}")
                with open(img_path, "wb") as f:
                    f.write(uploaded_img.getbuffer())
                img_file_name = uploaded_img.name
        else:
            img_url = st.text_input("Paste Direct Image URL", placeholder="https://example.com/photo.jpg", key="img_url_input")
            if st.button("Fetch & Analyze Image", key="btn_fetch_img") and img_url:
                with st.spinner("Downloading image from URL..."):
                    fetch_res = fetch_media_from_url(img_url, expected_type="image")
                if not fetch_res["success"]:
                    st.error(f"❌ {fetch_res['error']}")
                else:
                    img_path = fetch_res["file_path"]
                    img_file_name = fetch_res["filename"]
                    st.success(f"✅ Downloaded {img_file_name} ({fetch_res['size_mb']} MB)")

        if img_path and Path(img_path).is_file():
            file_res = validate_file(img_path)
            if not file_res["readable"]:
                st.error(f"❌ File corrupted or unreadable: {file_res.get('error')}")
            else:
                with st.spinner("Executing File Forensics, C2PA Provenance, and Parallel Content Analysis..."):
                    provenance_res = analyze_provenance(img_path)
                    content_res = content_analyzer.analyze_image_content(img_path)
                    image_result = analyze_image(img_path)

                with st.spinner("Running PRNU noise profiling, ELA, surface texture, 2D Fourier power slope, and memory bank lookup..."):
                    ai_result = detector.predict(img_path, sensitivity=sensitivity_key)

                with st.spinner("Profiling container chunks, visible watermarks, and international AI generator signatures..."):
                    img_profile = profile_media(img_path, modality="image")
                    attribution_res = attribution_engine.attribute_media(
                        img_path,
                        modality="image",
                        forensic_data=ai_result,
                        profile_data=img_profile,
                        provenance_data=provenance_res,
                    )

                decision = generate_final_decision(
                    file_validation=file_res,
                    quality_result=image_result,
                    ai_result=ai_result,
                    content_inventory=content_res,
                    provenance_result=provenance_res,
                    attribution_result=attribution_res,
                )

                # Visual Inspection & Spatial Localization Heatmap
                st.markdown("#### 🖼️ Visual Inspection & Spatial Localization Heatmap")
                col_orig, col_heat = st.columns(2)
                with col_orig:
                    st.image(img_path, caption=f"Original Media ({image_result.get('width')}x{image_result.get('height')})", width="stretch")
                with col_heat:
                    heatmap_rgb = ai_result.get("heatmap_rgb")
                    if heatmap_rgb is not None:
                        ai_area = ai_result.get("ai_spatial_area_pct", 0.0)
                        st.image(heatmap_rgb, caption=f"Spatial Anomaly Map (Estimated AI Area: {ai_area}%)", width="stretch")

                st.caption(
                    "🟢 **Cool / Low Residual:** Natural physical camera sensor noise (PRNU), organic optical grain.\n\n"
                    "🔴 **Hot / High Anomaly Zones:** Neural latent denoising, synthetic bilateral over-smoothing, or localized inpainting."
                )

    with col_img_right:
        if decision and content_res:
            render_analysis_right_panel(decision, content_res, modality="image")
        else:
            st.info(
                "👈 **Upload an image or paste a URL in the left panel to begin forensic analysis.**\n\n"
                "The right panel will automatically render:\n"
                "• Plain-English Executive Authenticity Verdict\n"
                "• Calibrated Authenticity Probabilities ($P(\\text{AI})$ vs $P(\\text{Real})$)\n"
                "• Scene & Content Depiction Intelligence (Entities, Items, Setting, Tone)\n"
                "• International AI Generator Attribution (Gemini, Flux, Midjourney, DALL-E, etc.)\n"
                "• Provenance & C2PA Content Credentials"
            )

    # Bottom single wide panel
    if decision and img_path and Path(img_path).is_file():
        render_bottom_feedback_panel(
            media_path=img_path,
            modality="image",
            forensic_data=ai_result or {},
            profile_data=img_profile or {},
            decision=decision,
            unique_key="img_tab",
        )


# =====================================================================
# =====================================================================
# 2. SEGREGATED VIDEO TAB
# =====================================================================
with tab_video:
    col_vid_left, col_vid_right = st.columns([1, 1.2], gap="large")

    vid_path = None
    vid_file_name = None
    file_res = None
    video_result = None
    audio_result = None
    content_res = None
    provenance_res = None
    attribution_res = None
    cross_modal_res = None
    decision = None
    vid_profile = None
    tmp_kf_path = None

    with col_vid_left:
        st.subheader("📥 Video Ingestion & Temporal Inspection")
        vid_mode = st.radio("Video Input Method", ["Upload Video File", "Fetch Video from URL"], horizontal=True, key="vid_mode")

        if vid_mode == "Upload Video File":
            uploaded_vid = st.file_uploader(
                "Upload Video",
                type=["mp4", "mov", "avi", "mkv", "webm"],
                key="uploader_vid",
            )
            if uploaded_vid:
                suffix = Path(uploaded_vid.name).suffix or ".mp4"
                vid_path = str(SESSION_CACHE_DIR / f"active_video{suffix}")
                with open(vid_path, "wb") as f:
                    f.write(uploaded_vid.getbuffer())
                vid_file_name = uploaded_vid.name
        else:
            vid_url = st.text_input("Paste Direct Video URL", placeholder="https://example.com/video.mp4", key="vid_url_input")
            if st.button("Fetch & Analyze Video", key="btn_fetch_vid") and vid_url:
                with st.spinner("Downloading video from URL..."):
                    fetch_res = fetch_media_from_url(vid_url, expected_type="video")
                if not fetch_res["success"]:
                    st.error(f"❌ {fetch_res['error']}")
                else:
                    vid_path = fetch_res["file_path"]
                    vid_file_name = fetch_res["filename"]
                    st.success(f"✅ Downloaded {vid_file_name} ({fetch_res['size_mb']} MB)")

        if vid_path and Path(vid_path).is_file():
            file_res = validate_file(vid_path)
            if not file_res["readable"]:
                st.error(f"❌ Video corrupted or unreadable: {file_res.get('error')}")
            else:
                st.markdown("#### 🎥 Video Preview")
                st.video(vid_path)

                with st.spinner("Extracting frames, profiling temporal consistency, demuxing audio, and scanning C2PA..."):
                    provenance_res = analyze_provenance(vid_path)
                    video_result = analyze_video(
                        vid_path,
                        sample_count=30,
                        ai_detector=detector,
                        sensitivity=sensitivity_key,
                    )
                    audio_result = audio_detector.analyze_audio_file(vid_path, sensitivity=sensitivity_key)

                # Sample keyframe for content inventory
                cap = cv2.VideoCapture(vid_path)
                ret, sample_frame = cap.read()
                cap.release()
                content_res = {}
                if ret and sample_frame is not None:
                    tmp_kf = SESSION_CACHE_DIR / "temp_kf.jpg"
                    cv2.imwrite(str(tmp_kf), sample_frame)
                    content_res = content_analyzer.analyze_image_content(str(tmp_kf))
                    tmp_kf_path = str(tmp_kf)

                cross_modal_res = evaluate_cross_modal_consistency(video_result, audio_result, content_res)

                with st.spinner("Profiling video stream signatures, corner badges, and international AI generator telltales..."):
                    vid_profile = profile_media(vid_path, modality="video")
                    attribution_res = attribution_engine.attribute_media(
                        vid_path,
                        modality="video",
                        forensic_data=video_result.get("ai_video_rating", {}),
                        profile_data=vid_profile,
                        provenance_data=provenance_res,
                    )

                decision = generate_final_decision(
                    file_validation=file_res,
                    quality_result=video_result,
                    ai_result=None,
                    audio_result=audio_result,
                    content_inventory=content_res,
                    provenance_result=provenance_res,
                    cross_modal_result=cross_modal_res,
                    attribution_result=attribution_res,
                )

                # Visual Keyframe Spatial Inspection & Temporal Heatmap Timeline
                st.markdown("#### 🖼️ Keyframe Spatial Anomaly & Temporal Timeline")
                if tmp_kf_path and Path(tmp_kf_path).is_file():
                    kf_ai = detector.predict(tmp_kf_path, sensitivity=sensitivity_key)
                    col_kf_orig, col_kf_heat = st.columns(2)
                    with col_kf_orig:
                        st.image(tmp_kf_path, caption="Sampled Video Keyframe", width="stretch")
                    with col_kf_heat:
                        kf_heat = kf_ai.get("heatmap_rgb")
                        if kf_heat is not None:
                            st.image(kf_heat, caption=f"Keyframe Spatial Heatmap ({kf_ai.get('ai_spatial_area_pct', 0)}% AI area)", width="stretch")
                    if Path(tmp_kf_path).exists():
                        try:
                            Path(tmp_kf_path).unlink()
                        except OSError:
                            pass

                # Temporal Timeline Display
                segments = video_result.get("temporal_segments", [])
                if segments:
                    st.markdown("##### ⏱️ Video Temporal Timeline Attribution")
                    for s in segments:
                        badge = "🚨 AI GENERATED" if s["label"] == "LIKELY AI-GENERATED" else ("✅ REAL" if s["label"] == "LIKELY REAL" else "❓ UNCERTAIN")
                        st.write(f"• `{s['start_seconds']}s ── {s['end_seconds']}s` ({s['duration_seconds']}s) : **{badge}**")
                    st.caption("Temporal consistency checks analyze inter-frame motion vector continuity to expose diffusion flickering and warped object boundaries.")

    with col_vid_right:
        if decision and content_res:
            render_analysis_right_panel(decision, content_res, modality="video")
        else:
            st.info(
                "👈 **Upload a video or paste a URL in the left panel to begin forensic analysis.**\n\n"
                "The right panel will automatically render:\n"
                "• Plain-English Executive Authenticity Verdict\n"
                "• Calibrated Authenticity Probabilities ($P(\\text{AI})$ vs $P(\\text{Real})$)\n"
                "• Scene & Content Depiction Intelligence (Entities, Items, Setting, Tone)\n"
                "• International AI Generator Attribution (Sora, Kling, Seedance, Runway, Hailuo)\n"
                "• Audio-Visual Cross-Modal Consistency & Synchronization"
            )

    # Bottom single wide panel
    if decision and vid_path and Path(vid_path).is_file():
        render_bottom_feedback_panel(
            media_path=vid_path,
            modality="video",
            forensic_data=video_result or {},
            profile_data=vid_profile or {},
            decision=decision,
            unique_key="vid_tab",
        )


# =====================================================================
# 3. SEGREGATED AUDIO TAB
# =====================================================================
with tab_audio:
    col_aud_left, col_aud_right = st.columns([1, 1.2], gap="large")

    aud_path = None
    aud_file_name = None
    file_res = None
    audio_result = None
    content_res = None
    provenance_res = None
    attribution_res = None
    decision = None
    aud_profile = None

    with col_aud_left:
        st.subheader("📥 Audio Ingestion & Spectral Inspection")
        aud_mode = st.radio("Audio Input Method", ["Upload Audio File", "Fetch Audio from URL"], horizontal=True, key="aud_mode")

        if aud_mode == "Upload Audio File":
            uploaded_aud = st.file_uploader(
                "Upload Audio",
                type=["mp3", "wav", "m4a", "aac", "flac", "ogg"],
                key="uploader_aud",
            )
            if uploaded_aud:
                suffix = Path(uploaded_aud.name).suffix or ".mp3"
                aud_path = str(SESSION_CACHE_DIR / f"active_audio{suffix}")
                with open(aud_path, "wb") as f:
                    f.write(uploaded_aud.getbuffer())
                aud_file_name = uploaded_aud.name
        else:
            aud_url = st.text_input("Paste Direct Audio URL", placeholder="https://example.com/speech.mp3", key="aud_url_input")
            if st.button("Fetch & Analyze Audio", key="btn_fetch_aud") and aud_url:
                with st.spinner("Downloading audio from URL..."):
                    fetch_res = fetch_media_from_url(aud_url, expected_type="audio")
                if not fetch_res["success"]:
                    st.error(f"❌ {fetch_res['error']}")
                else:
                    aud_path = fetch_res["file_path"]
                    aud_file_name = fetch_res["filename"]
                    st.success(f"✅ Downloaded {aud_file_name} ({fetch_res['size_mb']} MB)")

        if aud_path and Path(aud_path).is_file():
            file_res = validate_file(aud_path)
            if not file_res["readable"]:
                st.error(f"❌ Audio corrupted or unreadable: {file_res.get('error')}")
            else:
                st.markdown("#### 🎙️ Audio Player")
                st.audio(aud_path)

                with st.spinner("Extracting audio signal, inspecting vocoder cutoff, and checking C2PA credentials..."):
                    from utils.audio_utils import extract_audio_samples, generate_spectrogram_image
                    samples, sr, duration = extract_audio_samples(aud_path)
                    provenance_res = analyze_provenance(aud_path)
                    content_res = content_analyzer.analyze_audio_content(samples, sr, duration)
                    audio_result = audio_detector.analyze_audio_file(aud_path, sensitivity=sensitivity_key)

                with st.spinner("Profiling vocoder cutoff frequency, acoustic phase, and voice synthesis signatures..."):
                    aud_profile = profile_media(aud_path, modality="audio")
                    attribution_res = attribution_engine.attribute_media(
                        aud_path,
                        modality="audio",
                        forensic_data=audio_result,
                        profile_data=aud_profile,
                        provenance_data=provenance_res,
                    )

                decision = generate_final_decision(
                    file_validation=file_res,
                    quality_result={},
                    audio_result=audio_result,
                    content_inventory=content_res,
                    provenance_result=provenance_res,
                    attribution_result=attribution_res,
                )

                # Visual Acoustic Spectrogram Heatmap & Speech Timeline
                st.markdown("#### 🌊 Acoustic Spectrogram & Spectral Heatmap")
                if samples is not None and len(samples) > 0:
                    spec_img = generate_spectrogram_image(samples, sr)
                    if spec_img is not None:
                        st.image(spec_img, caption="Spectral Heatmap (Frequency vs Time) — Exposing Vocoder Cutoff Lines & Harmonic Smoothing", width="stretch")

                # Speech Timeline Display
                audio_segs = audio_result.get("temporal_segments", [])
                if audio_segs:
                    st.markdown("##### ⏱️ Speech Timeline Attribution")
                    for a_seg in audio_segs:
                        badge = "🚨 AI VOICE" if a_seg["label"] == "LIKELY AI-GENERATED" else ("✅ NATURAL SPEECH" if a_seg["label"] == "LIKELY REAL" else "❓ UNCERTAIN")
                        st.write(f"• `{a_seg['start_seconds']}s ── {a_seg['end_seconds']}s` ({a_seg['duration_seconds']}s) : **{badge}**")
                    st.caption("Acoustic analysis checks for brick-wall vocoder cutoffs (e.g. 7.5kHz/16kHz in ElevenLabs/Suno/CosyVoice), unnaturally flat Wiener entropy, and digital zero silence dropouts.")

    with col_aud_right:
        if decision and content_res:
            render_analysis_right_panel(decision, content_res, modality="audio")
        else:
            st.info(
                "👈 **Upload an audio file or paste a URL in the left panel to begin forensic analysis.**\n\n"
                "The right panel will automatically render:\n"
                "• Plain-English Executive Authenticity Verdict\n"
                "• Calibrated Authenticity Probabilities ($P(\\text{AI})$ vs $P(\\text{Real})$)\n"
                "• Acoustic Scene & Vocal Delivery Tone Intelligence\n"
                "• International Voice/Music Generator Attribution (ElevenLabs, CosyVoice, Suno, Udio)\n"
                "• Provenance & C2PA Content Credentials"
            )

    # Bottom single wide panel
    if decision and aud_path and Path(aud_path).is_file():
        render_bottom_feedback_panel(
            media_path=aud_path,
            modality="audio",
            forensic_data=audio_result or {},
            profile_data=aud_profile or {},
            decision=decision,
            unique_key="aud_tab",
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