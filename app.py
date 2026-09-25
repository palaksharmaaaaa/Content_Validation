import os
from pathlib import Path
import tempfile

import cv2
import streamlit as st

from models.ai_audio_detector import AIAudioDetector
from models.ai_image_detector import AIImageDetector
from models.face_detector import FaceDeepfakeDetector
from scoring.decision_engine import generate_final_decision
from validators.file_validator import validate_file
from validators.image_validator import analyze_image
from validators.url_validator import validate_expected_platform
from validators.video_validator import analyze_video


DETECTOR_CHECKPOINT = Path(__file__).resolve().parent / "models" / "ai_detector.pt"


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


st.set_page_config(
    page_title="OmniForensics: Multi-Modal AI & Media Validator",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 OmniForensics: Multi-Modal Media Authenticity Engine")
st.write(
    "Deep multimodal forensic inspection distinguishing **Authenticity Probabilities** from "
    "**Spatial & Temporal Proportions**, with explicit uncertainty modeling across Image, Video, and Audio."
)

st.subheader("📁 Media Input")

input_method = st.radio(
    "Choose input method",
    ["Upload File", "Paste URL"],
    horizontal=True,
)

uploaded_file = None
media_url = None

if input_method == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload Image or Video",
        type=[
            "jpg", "jpeg", "png", "webp", "bmp", "tiff",
            "mp4", "mov", "avi", "mkv", "webm",
        ],
    )
else:
    media_url = st.text_input(
        "Paste Media URL",
        placeholder="https://www.instagram.com/...",
    )
    expected_platform = st.selectbox(
        "Expected Platform",
        ["Auto", "Instagram", "YouTube", "Facebook", "TikTok", "X"],
    )

if input_method == "Paste URL" and media_url:
    url_result = validate_expected_platform(media_url, expected_platform)
    st.subheader("🔗 URL Validation")

    if not url_result["valid_url"]:
        st.error(f"❌ {url_result['message']}")
    else:
        st.write(f"Detected Platform: **{url_result['platform']}**")
        if expected_platform == "Auto" or url_result["platform_match"]:
            st.success(f"✅ {url_result['message']}")
        else:
            st.error(url_result["message"])

if input_method == "Upload File" and uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        # -----------------------------
        # FILE VALIDATION
        # -----------------------------
        st.subheader("📁 File Integrity & Quality")
        file_result = validate_file(temp_path)

        if not file_result["readable"]:
            st.error(f"❌ File is not readable: {file_result.get('error')}")
            st.stop()
        st.success("✅ File is valid, readable, and within size constraints.")

        detector = get_ai_detector()
        face_detector = get_face_detector()
        audio_detector = get_audio_detector()

        # -----------------------------
        # IMAGE ANALYSIS
        # -----------------------------
        if file_result["media_type"] == "image":
            image_result = analyze_image(temp_path)

            with st.spinner("Executing multi-signal forensic ensemble (PRNU noise, ELA, surface texture, facial cues)..."):
                ai_result = detector.predict(temp_path)

            ai_pct = ai_result.get("ai_percentage", 0.0)
            real_pct = ai_result.get("real_percentage", 0.0)
            undecided_pct = ai_result.get("undecided_percentage", 100.0)
            classification = ai_result.get("prediction", "UNDECIDED")
            ai_spatial_area = ai_result.get("ai_spatial_area_pct", 0.0)

            # Visual Comparison: Original vs Spatial Anomaly Heatmap
            st.subheader("🖼️ Visual Inspection & Spatial Localization Heatmap")
            col_orig, col_heat = st.columns(2)
            with col_orig:
                st.image(uploaded_file, caption=f"Original Media ({image_result.get('width')}x{image_result.get('height')})", use_container_width=True)
            with col_heat:
                heatmap_rgb = ai_result.get("heatmap_rgb")
                if heatmap_rgb is not None:
                    st.image(heatmap_rgb, caption=f"Spatial Anomaly Map (Estimated AI Area: {ai_spatial_area}%)", use_container_width=True)
                else:
                    st.info("Spatial heatmap unavailable.")

            # Metrics Row 1: Authenticity Probabilities
            st.subheader("🤖 Authenticity Probabilities vs Spatial Proportion")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("🤖 P(AI-Generated)", f"{ai_pct:.1f}%")
            mcol2.metric("📷 P(Real / Authentic)", f"{real_pct:.1f}%")
            mcol3.metric("❓ P(Undetermined / OOD)", f"{undecided_pct:.1f}%")
            mcol4.metric("📍 Manipulated Area", f"{ai_spatial_area:.1f}%")

            # Verdict Status
            if classification == "LIKELY AI-GENERATED":
                st.error(f"🚨 **Verdict: {classification}** (Ensemble Confidence: {ai_result.get('confidence', 0)*100:.1f}%)")
            elif classification == "LIKELY REAL":
                st.success(f"✅ **Verdict: {classification}** (Ensemble Confidence: {ai_result.get('confidence', 0)*100:.1f}%)")
            else:
                st.warning(f"⚠️ **Verdict: {classification}** — Evidence is ambiguous or out-of-distribution.")

            # Forensic Evidence & Cues
            cues = ai_result.get("forensic_cues", [])
            if cues:
                st.subheader("🔬 Forensic Evidence & Anomaly Attribution")
                for cue in cues:
                    st.markdown(f"• **{cue}**")

            # Facial Deepfake Analysis
            face_res = ai_result.get("facial_analysis", {})
            if face_res.get("faces_detected", 0) > 0:
                st.subheader("👤 Facial & Deepfake Inspection")
                fcol1, fcol2, fcol3 = st.columns(3)
                fcol1.metric("Faces Detected", face_res["faces_detected"])
                fcol2.metric("Deepfake Risk", face_res["deepfake_risk"])
                fcol3.metric("Facial AI Confidence", f"{face_res.get('facial_ai_confidence', 0)*100:.1f}%")

            # Final Decision Engine Verdict
            decision = generate_final_decision(file_result, image_result, ai_result)
            st.subheader("🎯 Final Decision Verdict")
            if decision["final_status"] == "VALID":
                st.success(f"**Status: {decision['final_status']}** — {decision['reason']}")
            elif decision["final_status"] == "PARTIALLY_VALID":
                st.warning(f"**Status: {decision['final_status']}** — {decision['reason']}")
            else:
                st.error(f"**Status: {decision['final_status']}** — {decision['reason']}")

            # Deep Forensics JSON
            with st.expander("📊 Complete Multi-Modal Forensic Report (JSON)"):
                st.json({
                    "authenticity": {
                        "ai_probability": ai_pct / 100.0,
                        "real_probability": real_pct / 100.0,
                        "undetermined_probability": undecided_pct / 100.0,
                    },
                    "spatial_proportion": {
                        "ai_spatial_area_pct": ai_spatial_area,
                    },
                    "forensic_metrics": ai_result.get("forensic_metrics", {}),
                    "facial_analysis": face_res,
                    "metadata": ai_result.get("metadata_forensics", {}),
                    "decision": decision,
                })

        # -----------------------------
        # VIDEO ANALYSIS
        # -----------------------------
        elif file_result["media_type"] == "video":
            st.subheader("🎥 Video Preview")
            st.video(uploaded_file)

            st.subheader("📊 Video Temporal Profiling & AI Detection")
            with st.spinner("Extracting and analyzing video frames across timeline..."):
                video_result = analyze_video(temp_path, sample_count=30, ai_detector=detector)

            metadata = video_result.get("metadata", {})
            vcol1, vcol2, vcol3, vcol4 = st.columns(4)
            vcol1.metric("Width", metadata.get("width", "N/A"))
            vcol2.metric("Height", metadata.get("height", "N/A"))
            vcol3.metric("FPS", metadata.get("fps", "N/A"))
            vcol4.metric("Duration", f"{metadata.get('duration_seconds', 0)} sec")

            # AI Video Breakdown
            ai_vid = video_result.get("ai_video_rating", {})
            st.subheader("🤖 Video Authenticity Breakdown vs Temporal Duration")
            acol1, acol2, acol3, acol4 = st.columns(4)
            acol1.metric("🤖 P(AI-Generated)", f"{ai_vid.get('ai_percentage', 0.0):.1f}%")
            acol2.metric("📷 P(Real Frames)", f"{ai_vid.get('real_percentage', 0.0):.1f}%")
            acol3.metric("❓ P(Undetermined)", f"{ai_vid.get('undecided_percentage', 100.0):.1f}%")
            ai_dur_pct = ai_vid.get("details", {}).get("ai_duration_pct", 0.0)
            acol4.metric("⏱️ Manipulated Duration", f"{ai_dur_pct:.1f}%")

            # Temporal Segments Timeline
            segments = video_result.get("temporal_segments", [])
            if segments:
                st.subheader("⏱️ Temporal Timeline Attribution")
                for s in segments:
                    badge = "🚨 AI GENERATED" if s["label"] == "LIKELY AI-GENERATED" else ("✅ REAL" if s["label"] == "LIKELY REAL" else "❓ UNCERTAIN")
                    st.write(f"• `{s['start_seconds']}s ── {s['end_seconds']}s` ({s['duration_seconds']}s) : **{badge}**")

            # Audio Analysis of Video File
            audio_result = audio_detector.analyze_audio_file(temp_path)
            st.subheader("🎙️ Audio Track Forensics (Vocoder & Speech)")
            aucol1, aucol2, aucol3 = st.columns(3)
            aucol1.metric("🤖 Audio AI %", f"{audio_result.get('ai_percentage', 0.0):.1f}%")
            aucol2.metric("🎙️ Audio Real %", f"{audio_result.get('real_percentage', 0.0):.1f}%")
            aucol3.metric("❓ Audio Undecided %", f"{audio_result.get('undecided_percentage', 100.0):.1f}%")

            # Final Decision Engine
            decision = generate_final_decision(file_result, video_result, ai_result=ai_vid, audio_result=audio_result)
            st.subheader("🎯 Final Decision Verdict")
            if decision["final_status"] == "VALID":
                st.success(f"**Status: {decision['final_status']}** — {decision['reason']}")
            elif decision["final_status"] == "PARTIALLY_VALID":
                st.warning(f"**Status: {decision['final_status']}** — {decision['reason']}")
            else:
                st.error(f"**Status: {decision['final_status']}** — {decision['reason']}")

            with st.expander("🔍 Complete Video Forensic JSON"):
                st.json({
                    "video_summary": video_result,
                    "audio_summary": audio_result,
                    "final_decision": decision,
                })

    except Exception as e:
        st.error("Something went wrong while processing the file.")
        st.exception(e)

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass