
import tempfile
import os
from pathlib import Path

import streamlit as st

from models.ai_image_detector import AIImageDetector

from validators.file_validator import validate_file
from validators.image_validator import analyze_image
from validators.video_validator import analyze_video
from validators.url_validator import validate_expected_platform


DETECTOR_CHECKPOINT = Path(__file__).resolve().parent / "models" / "ai_detector.pt"


@st.cache_resource
def load_ai_detector(checkpoint_mtime: float):
    detector = AIImageDetector()
    detector.load()
    return detector

st.set_page_config(
    page_title="Media Content Validator",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Media Content Validator")

st.write(
    "Upload an image or video to validate file integrity, "
    "quality, blank content and usability."
)

# uploaded_file = st.file_uploader(
#     "Upload Image or Video",
#     type=[
#         "jpg", "jpeg", "png", "webp", "bmp",
#         "mp4", "mov", "avi", "mkv", "webm"
#     ]
# )

st.subheader("📁 Media Input")

input_method = st.radio(
    "Choose input method",
    ["Upload File", "Paste URL"],
    horizontal=True
)

uploaded_file = None
media_url = None

if input_method == "Upload File":

    uploaded_file = st.file_uploader(
        "Upload Image or Video",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
            "tiff",
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm",
        ]
    )

else:

    media_url = st.text_input(
        "Paste Media URL",
        placeholder="https://www.instagram.com/..."
    )

    expected_platform = st.selectbox(
        "Expected Platform",
        [
            "Auto",
            "Instagram",
            "YouTube",
            "Facebook",
            "TikTok",
            "X",
        ]
    )

if input_method == "Paste URL" and media_url:

    url_result = validate_expected_platform(
        media_url,
        expected_platform
    )

    st.subheader("🔗 URL Validation")

    if not url_result["valid_url"]:

        st.error(
            f"❌ {url_result['message']}"
        )

    else:

        st.write(
            f"Detected Platform: "
            f"**{url_result['platform']}**"
        )

        if expected_platform == "Auto":

            st.success(
                f"✅ {url_result['message']}"
            )

        elif url_result["platform_match"]:

            st.success(
                url_result["message"]
            )

        else:

            st.error(
                url_result["message"]
            )

if input_method == "Upload File" and uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:

        # -----------------------------
        # FILE VALIDATION
        # -----------------------------

        st.subheader("📁 File Validation")

        file_result = validate_file(temp_path)

        st.json(file_result)

        if not file_result["readable"]:
            st.error("❌ File is not readable.")
            st.stop()

        st.success("✅ File is readable.")

        # -----------------------------
        # IMAGE ANALYSIS
        # -----------------------------

        if file_result["media_type"] == "image":

            st.subheader("🖼️ Image Preview")

            st.image(
                uploaded_file,
                caption=uploaded_file.name
            )
            st.subheader("📊 Image Quality Analysis")

            image_result = analyze_image(temp_path)
            # ============================================================
            # AI GENERATED IMAGE DETECTION
            # ============================================================

            st.subheader(
                "🤖 AI-Generated Content Detection"
            )

            try:

                detector = load_ai_detector(DETECTOR_CHECKPOINT.stat().st_mtime)

                with st.spinner(
                    "Analyzing image for AI-generated content..."
                ):

                    ai_result = detector.predict(
                        temp_path
                    )

                ai_probability = (
                    ai_result["ai_probability"]
                )

                real_probability = (
                    ai_result["real_probability"]
                )

                confidence = (
                    ai_result["confidence"]
                )

                classification = (
                    ai_result["prediction"]
                )


                # --------------------------------------------------------
                # Metrics
                # --------------------------------------------------------

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "AI Probability",
                        f"{ai_probability * 100:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Real Probability",
                        f"{real_probability * 100:.2f}%"
                    )

                with col3:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )


                # --------------------------------------------------------
                # Classification
                # --------------------------------------------------------

                if classification == (
                    "LIKELY AI-GENERATED"
                ):

                    st.warning(
                        "🤖 Likely AI-generated content"
                    )

                elif classification == (
                    "LIKELY REAL"
                ):

                    st.success(
                        "📷 Likely real content"
                    )

                else:

                    st.info(
                        "⚠️ AI detection is uncertain"
                    )


                # --------------------------------------------------------
                # Details
                # --------------------------------------------------------

                with st.expander(
                    "AI Detection Details"
                ):

                    st.json(
                        ai_result
                    )


            except Exception as e:

                st.error(
                    "AI detection failed."
                )

                st.exception(e)

            if image_result.get("valid"):

                st.success("✅ Image contains usable content.")

            else:

                st.error("❌ Image is invalid or blank.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Width",
                    image_result.get("width", "N/A")
                )

            with col2:
                st.metric(
                    "Height",
                    image_result.get("height", "N/A")
                )

            with col3:
                st.metric(
                    "Blur Score",
                    image_result.get("blur_score", "N/A")
                )

            st.write("### Detailed Analysis")

            st.json(image_result)

        # -----------------------------
        # VIDEO
        # -----------------------------




        elif file_result["media_type"] == "video":

            st.subheader("🎥 Video Preview")

            st.video(uploaded_file)

            st.subheader("📊 Video Analysis")

            with st.spinner("Analyzing video frames..."):

                video_result = analyze_video(
                    temp_path,
                    sample_count=30
                )

            status = video_result.get(
                "content_status",
                "UNKNOWN"
            )

            if status == "VALID":

                st.success(
                    "✅ VALID — Video contains sufficient clear, usable content."
                )

            elif status == "PARTIALLY_VALID":

                st.warning(
                    "⚠️ PARTIALLY VALID — Video contains some blank or blurry frames."
                )

            else:

                st.error(
                    "❌ INVALID — Video does not contain sufficient clear, usable content."
                )

            # -----------------------------
            # VIDEO METADATA
            # -----------------------------

            metadata = video_result.get("metadata", {})

            st.subheader("🎬 Video Metadata")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Width",
                    metadata.get("width", "N/A")
                )

            with col2:
                st.metric(
                    "Height",
                    metadata.get("height", "N/A")
                )

            with col3:
                st.metric(
                    "FPS",
                    metadata.get("fps", "N/A")
                )

            with col4:
                st.metric(
                    "Duration",
                    f'{metadata.get("duration_seconds", 0)} sec'
                )

            # -----------------------------
            # FRAME ANALYSIS
            # -----------------------------

            st.subheader("🖼️ Frame Analysis")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Frames Analyzed",
                    video_result.get("frames_analyzed", 0)
                )

            with col2:
                st.metric(
                    "Valid Frames",
                    video_result.get("valid_frames", 0)
                )

            with col3:
                st.metric(
                    "Blank Frames",
                    video_result.get("blank_frames", 0)
                )

            with col4:
                st.metric(
                    "Blurry Frames",
                    video_result.get("blurry_frames", 0)
                )

            st.subheader("📊 Content Distribution")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Usable Content",
                    f'{video_result.get("usable_percentage", 0):.2f}%'
                )

            with col2:
                st.metric(
                    "Blurry Content",
                    f'{video_result.get("blurry_percentage", 0):.2f}%'
                )

            with col3:
                st.metric(
                    "Blank Content",
                    f'{video_result.get("blank_percentage", 0):.2f}%'
                )


            st.subheader("⏱️ Content Duration")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Usable Duration",
                    f'{video_result.get("usable_duration_seconds", 0):.2f} sec'
                )

            with col2:
                st.metric(
                    "Blurry Duration",
                    f'{video_result.get("blurry_duration_seconds", 0):.2f} sec'
                )

            with col3:
                st.metric(
                    "Blank Duration",
                    f'{video_result.get("blank_duration_seconds", 0):.2f} sec'
                )

            # -----------------------------
            # CONTENT COVERAGE
            # -----------------------------

            coverage = video_result.get(
                "content_coverage",
                0
            )

            st.subheader("📈 Content Coverage")

            st.progress(coverage)

            st.write(
                f"Content Coverage: **{coverage * 100:.2f}%**"
            )

            content_status = video_result.get(
                "content_status",
                "UNKNOWN"
            )

            if content_status == "VALID":

                st.success(
                    "✅ VALID — Most sampled frames contain usable content."
                )

            elif content_status == "PARTIALLY_VALID":

                st.warning(
                    "⚠️ PARTIALLY VALID — Some frames contain blank/unusable content."
                )

            else:

                st.error(
                    "❌ INVALID — Most sampled frames are blank/unusable."
                )

            # -----------------------------
            # DETAILED RESULT
            # -----------------------------

            with st.expander("🔍 Detailed Video Analysis"):

                st.json(video_result)

    except Exception as e:

        st.error("Something went wrong while processing the file.")

        st.exception(e)


        

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)