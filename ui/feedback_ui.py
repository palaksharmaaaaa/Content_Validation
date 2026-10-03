"""
Feedback and Continuous Learning UI Components for Streamlit.
Provides:
1. Low-level media & pixel forensic specifications card.
2. User multi-criteria 1-10 rating widget for online continual learning.
3. Interactive Continuous Learning & Forensic Memory Bank Dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import streamlit as st

from audio_detector import AudioProfiler, AudioSelfImprover
from image_detector import ImageProfiler, ImageSelfImprover
from image_detector.explain import build_nine_dimensions_dossier, generate_newbie_explanation
from video_detector import VideoProfiler, VideoSelfImprover
from ui.profile_view import add_ui_profile_blocks
from ui.stages import (
    collect_findings,
    render_confidence_and_reliability,
    render_findings_stage,
    render_gate_banner,
)


def profile_media(file_path: str | Path, modality: str = "auto", source: str = "User Upload") -> Dict[str, Any]:
    p = Path(file_path)
    suffix = p.suffix.lower()
    if modality == "image" or (modality == "auto" and suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
        return add_ui_profile_blocks("image", ImageProfiler().profile_image(file_path, source=source))
    if modality == "video" or (modality == "auto" and suffix in (".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return add_ui_profile_blocks("video", VideoProfiler().profile_video(file_path))
    if modality == "audio" or (modality == "auto" and suffix in (".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a")):
        return add_ui_profile_blocks("audio", AudioProfiler().profile_audio(file_path))
    return {"success": False, "error": f"Unknown format: {suffix}"}


def render_scene_and_content_intelligence(content_data: Dict[str, Any], modality: str = "image") -> None:
    """
    Renders comprehensive scene, content, environment, daytime, tone,
    entities, items, and purpose analysis BEFORE presenting AI detection.
    """
    if not content_data or "error" in content_data:
        return

    st.subheader("🌐 Scene, Content & Environmental Intelligence")
    st.caption("Deep contextual analysis of what is depicted in the media before reviewing AI forensics.")

    if modality in ("image", "video"):
        entities = content_data.get("entities", {})
        humans = entities.get("humans", {})
        animals = entities.get("animals", {})
        vehicles = content_data.get("vehicles", {})
        items = content_data.get("contents_and_items", {})
        env = content_data.get("environment_and_surroundings", {})
        lighting = content_data.get("lighting_and_daytime", {})
        tone = content_data.get("tone_and_mood", {})
        purpose = content_data.get("purpose_and_depiction", {})

        # Row 1: Purpose & Setting Overview
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Purpose / Depiction", purpose.get("photographic_purpose", "General Depiction"))
        col2.metric("🏞️ Setting & Environment", f"{env.get('location_context', 'Indoor')} • {env.get('setting_type', 'General')}")
        col3.metric("☀️ Daytime & Lighting", lighting.get("estimated_daytime", "Daylight"))

        # Row 2: Living Entities & Items
        ecol1, ecol2, ecol3 = st.columns(3)
        human_str = f"{humans.get('persons_count', 0)} person(s), {humans.get('faces_count', 0)} face(s)" if humans.get("has_humans") else "None detected"
        ecol1.metric("🧍 Living Humans", human_str)

        anim_list = animals.get("animal_types", [])
        ecol2.metric("🐾 Animals Detected", ", ".join(anim_list) if anim_list else "None detected")

        veh_list = vehicles.get("vehicle_types", [])
        ecol3.metric("🚗 Vehicles Detected", ", ".join(veh_list) if veh_list else "None detected")

        # Row 3: Items, Tone, Lighting details in expandable container
        with st.expander("🔎 View Full Scene Breakdown (Items, Color Tone, Lighting Details)", expanded=True):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("##### 📦 Identified Items & Objects")
                item_list = items.get("identified_items", [])
                if item_list:
                    st.write("• " + ", ".join(f"`{item}`" for item in item_list))
                else:
                    st.write("• No prominent everyday objects isolated.")

                top_rec = items.get("top_recognitions", [])
                if top_rec:
                    st.caption("Top recognitions: " + ", ".join(f"{r['label']} ({r['confidence_pct']}%)" for r in top_rec[:4]))

                st.write(f"• **Text / Typography Regions:** `{items.get('text_regions_count', 0)}` block(s) detected")

            with sc2:
                st.markdown("##### 🎨 Atmosphere, Color Tone & Lighting Style")
                st.write(f"• **Color Tone:** `{tone.get('color_tone', 'Neutral')}`")
                st.write(f"• **Atmospheric Mood:** `{tone.get('atmospheric_mood', 'Balanced')}`")
                st.write(f"• **Lighting Style:** `{lighting.get('lighting_style', 'Ambient')}`")
                st.write(f"• **Color Temperature:** `{lighting.get('color_temperature', 'Neutral')}`")
                st.write(f"• **Vegetation Coverage:** `{env.get('vegetation_coverage_pct', 0)}%` | **Sky/Water:** `{env.get('sky_water_coverage_pct', 0)}%`")

    elif modality == "audio":
        col1, col2, col3 = st.columns(3)
        col1.metric("🎵 Audio Stream Type", content_data.get("dominant_audio_type", "Audio"))
        col2.metric("🎙️ Acoustic Environment", content_data.get("acoustic_environment", "Studio"))
        col3.metric("🎭 Vocal Delivery & Tone", content_data.get("vocal_tone_and_delivery", "Conversational"))

        with st.expander("🔎 View Acoustic Environment & Speech Details", expanded=True):
            st.write(f"• **Audio Purpose / Genre:** `{content_data.get('audio_purpose', 'N/A')}`")
            st.write(f"• **Estimated Speakers:** `{content_data.get('estimated_speakers', 0)}`")
            st.write(f"• **Silence Ratio:** `{content_data.get('silence_ratio_pct', 0)}%`")
            st.write(f"• **Dynamic Range Index:** `{content_data.get('dynamic_range_index', 0)}`")


def render_media_specs(profile_data: Dict[str, Any]) -> None:
    """Renders comprehensive low-level media, pixel, and signal forensics."""
    if not profile_data.get("success"):
        return

    st.subheader("🔬 Deep Media & Pixel Signal Specifications")

    file_id = profile_data.get("file_identity", {})
    pixel_spec = profile_data.get("pixel_specifications") or profile_data.get("stream_specifications") or {}
    audio_spec = profile_data.get("signal_specifications") or profile_data.get("demuxed_audio", {}).get("signal_specifications", {})
    provenance = profile_data.get("provenance_metadata", {})

    with st.expander("🔍 View Technical Attributes (Pixel Type, Entropy, Bit Depth, Hashes)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("File Format", file_id.get("container_format", "N/A"))
        c2.metric("File Size", f"{file_id.get('size_kb', 0):.1f} KB")
        c3.metric("MIME Type", file_id.get("mime_type", "N/A"))
        sha = file_id.get("sha256", "")
        c4.metric("SHA-256 (Prefix)", f"{sha[:10]}..." if sha else "N/A")

        if pixel_spec:
            st.markdown("##### 🎨 Visual & Pixel Architecture")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Pixel Data Type", pixel_spec.get("pixel_data_type", "uint8"))
            p2.metric("Data per Pixel", f"{pixel_spec.get('bits_per_pixel', 24)} bpp")
            p3.metric("Shannon Entropy", f"{pixel_spec.get('shannon_entropy_bpp', pixel_spec.get('keyframe_entropy_bpp', 0.0)):.3f} bits/px")
            p4.metric("Color Space", pixel_spec.get("color_space", "RGB"))

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Geometry", f"{pixel_spec.get('width', 'N/A')} x {pixel_spec.get('height', 'N/A')}")
            d2.metric("Aspect Ratio", pixel_spec.get("aspect_ratio", "N/A"))
            d3.metric("Total Megapixels", f"{pixel_spec.get('megapixels', 0.0):.2f} MP")
            d4.metric("Pixel Variance", f"{pixel_spec.get('pixel_variance', 0.0):.1f}")

        if audio_spec:
            st.markdown("##### 🎙️ Audio Signal & Acoustic Characteristics")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Sample Rate", f"{audio_spec.get('sample_rate_hz', 0)} Hz")
            a2.metric("Bit Depth", f"{audio_spec.get('bit_depth', 16)}-bit")
            a3.metric("Bitrate", f"{audio_spec.get('bitrate_kbps', 0.0):.1f} kbps")
            a4.metric("RMS Energy", f"{audio_spec.get('rms_energy', 0.0):.4f}")

        if provenance and provenance.get("has_exif"):
            st.markdown("##### 📷 Hardware Provenance & Proven Tags")
            cam_str = f"{provenance.get('camera_make', '')} {provenance.get('camera_model', '')}".strip() or "Standard Device"
            st.write(f"• Hardware: **{cam_str}** | Software: `{provenance.get('software') or 'None'}` | Date: `{provenance.get('date_time') or 'None'}`")

        st.json({
            "cryptographic_hashes": {"sha256": file_id.get("sha256"), "md5": file_id.get("md5")},
            "specifications": pixel_spec or audio_spec,
            "provenance": provenance,
        })


_GROUND_TRUTH_LABELS = ["AI-Generated", "Real / Camera Authentic", "Partially Forged / Deepfake", "Undetermined / Mixed"]
_GENERATOR_FAMILIES = [
    "Auto-Attributed / Unspecified",
    "Google Gemini / Imagen 3 (USA)",
    "OpenAI Sora (USA)",
    "OpenAI DALL-E 3 / ChatGPT (USA)",
    "Kuaishou Kling AI (China)",
    "ByteDance Seedance / Jimeng AI (China)",
    "MiniMax Hailuo AI (China)",
    "Black Forest Labs Flux.1 (Germany/EU)",
    "Midjourney v5 / v6 (USA)",
    "Runway Gen-2 / Gen-3 (USA)",
    "Stability AI SDXL / SD 3.5 (UK)",
    "Luma Dream Machine (USA)",
    "ElevenLabs Voice Engine (USA/Poland)",
    "Alibaba CosyVoice (China)",
    "Suno AI Music (USA)",
    "Udio AI Music (USA)",
    "Authentic Camera (Apple / Sony / Canon / Nikon)",
    "Other Custom Generator",
]
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _resolve_feedback_modality(modality: str, media_path: str | Path) -> str:
    """'image' | 'video' | 'audio' (auto-detected from the extension when modality == 'auto')."""
    name = str(media_path).lower()
    if modality == "image" or (modality == "auto" and name.endswith(_IMAGE_SUFFIXES)):
        return "image"
    if modality == "video" or (modality == "auto" and name.endswith(_VIDEO_SUFFIXES)):
        return "video"
    return "audio"


def _render_rating_sliders(unique_key: str) -> Dict[str, float]:
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        real_score = st.slider(
            "📷 Real / Authentic Score (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5,
            key=f"{unique_key}_real_score",
            help="1 = Completely Fake/Synthetic, 10 = 100% Genuine Camera/Microphone Capture",
        )
        synth_score = st.slider(
            "🤖 Synthetic / AI-Generated Score (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5,
            key=f"{unique_key}_synth_score",
            help="1 = Natural/Organic, 10 = Full Generative Synthesis (Midjourney, Gemini, ElevenLabs, Sora)",
        )
    with fcol2:
        forged_score = st.slider(
            "✂️ Forged / Edited / Deepfake Score (1-10)", min_value=1.0, max_value=10.0, value=3.0, step=0.5,
            key=f"{unique_key}_forged_score",
            help="1 = Unaltered, 10 = Heavily Doctored / Face-Swapped / Splice-Edited",
        )
        undetected_score = st.slider(
            "⚠️ Undetected / Missed Artifacts Score (0-10)", min_value=0.0, max_value=10.0, value=2.0, step=0.5,
            key=f"{unique_key}_undetected_score",
            help="How much did current automated detection miss or under-report? (0 = Caught Everything, 10 = Missed Crucial Artifacts)",
        )
    return {
        "real_score": real_score,
        "synthetic_score": synth_score,
        "forged_score": forged_score,
        "undetected_score": undetected_score,
    }


def _render_label_inputs(unique_key: str, notes_placeholder: str) -> Tuple[str, str, str]:
    """Ground-truth label, generator tag and free-text notes."""
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        ground_truth = st.selectbox("Confirmed Ground Truth Label", _GROUND_TRUTH_LABELS, key=f"{unique_key}_gt")
        generator_select = st.selectbox("Known / Suspected Generator Family", _GENERATOR_FAMILIES, key=f"{unique_key}_gen_select")
    with rcol2:
        custom_gen_tag = st.text_input(
            "Custom Generator / Camera Hardware Details (Optional)",
            placeholder="e.g. Gemini 1.5 Flash, Midjourney v6.1, iPhone 16 Pro Max",
            key=f"{unique_key}_custom_gen_tag",
        )
        generator_tag = custom_gen_tag.strip() or (generator_select if generator_select != "Auto-Attributed / Unspecified" else "")
        user_notes = st.text_area(
            "Forensic Observations & Commentary", placeholder=notes_placeholder, key=f"{unique_key}_notes",
        )
    return ground_truth, generator_tag, user_notes


def _submit_feedback(
    media_path: str | Path, modality: str, forensic_data: Any, ratings: Dict[str, float],
    ground_truth: str, generator_tag: str, user_notes: str,
) -> None:
    """Records the rating with the owning modality's learner (queued by reference; nothing is copied)."""
    user_label = "AI" if ("AI" in ground_truth or ratings["synthetic_score"] >= 5.5) else "REAL"
    metrics = forensic_data if isinstance(forensic_data, dict) else (forensic_data.to_dict() if hasattr(forensic_data, "to_dict") else {})
    notes_text = f"[{ground_truth}] {user_notes} | Gen: {generator_tag} | Ratings: {ratings}"
    resolved = _resolve_feedback_modality(modality, media_path)
    improver = {"image": ImageSelfImprover, "video": VideoSelfImprover, "audio": AudioSelfImprover}[resolved]()
    calib = improver.record_feedback(str(media_path), user_label, metrics, notes=notes_text)
    st.success(f"✅ **{resolved.capitalize()} Forensic Feedback Successfully Recorded!**")
    st.info(f"🧠 **Learned Calibration Updates:** Processed {calib.get('samples_processed', 1)} verified samples.")


def _render_feedback_form(
    media_path: str | Path, modality: str, forensic_data: Any, unique_key: str, notes_placeholder: str
) -> None:
    ratings = _render_rating_sliders(unique_key)
    ground_truth, generator_tag, user_notes = _render_label_inputs(unique_key, notes_placeholder)
    if st.button("💾 Submit Rating & Dynamically Retrain Algorithm", key=f"{unique_key}_submit_btn"):
        with st.spinner("Archiving fingerprint, auto-calibrating parameters, and updating models..."):
            _submit_feedback(media_path, modality, forensic_data, ratings, ground_truth, generator_tag, user_notes)


def render_feedback_box(
    media_path: str | Path,
    modality: str,
    forensic_data: Dict[str, Any],
    unique_key: str,
) -> None:
    """
    Renders an interactive rating and feedback widget for continuous learning.
    Allows user to rate 1-10 on Real, Synthetic, Forged, and Undetected metrics.
    """
    st.subheader("📝 Forensic Feedback & Dynamic Algorithm Training")
    st.caption(
        "Help the engine learn and continuously improve: rate the authenticity out of 10. "
        "The system will remember this file's pixel profile, metadata, and physics metrics, "
        "dynamically auto-updating its neural weights and detection thresholds."
    )
    with st.expander("⭐ Rate this media & auto-train engine", expanded=False):
        _render_feedback_form(
            media_path, modality, forensic_data, unique_key,
            "Describe specific telltales: waxy skin, warped fingers, abnormal room reflections, vocoder cutoff...",
        )


def render_retrain_panel() -> None:
    """Per-modality "Retrain Now": fine-tunes on files registered in place and promotes only if validation holds."""
    import importlib

    from audio_detector.config import RETRAIN_SUGGEST_THRESHOLD as _T  # same value in all three configs
    st.subheader("🔁 Retrain on Verified Feedback")
    st.caption(
        "Verified files are queued by reference (never copied). Retraining validates on held-out files and "
        "keeps the previous checkpoint if accuracy would drop."
    )
    cols = st.columns(3)
    for col, (label, mod) in zip(cols, (("Image", "image"), ("Video", "video"), ("Audio", "audio"))):
        with col:
            retrain = importlib.import_module(f"{mod}_detector.retrain")
            pending = retrain.count_pending_corrections()
            st.metric(f"{label}: queued", pending)
            if pending >= _T:
                st.info(f"{pending} new verified files — a retrain is suggested.")
            if st.button(f"Retrain Now ({label})", key=f"retrain_now_{mod}", disabled=pending == 0):
                with st.spinner(f"Retraining {label.lower()} model..."):
                    try:
                        result = retrain.run_retrain(min_new=1)
                    except Exception as exc:  # surface, don't crash the page
                        st.error(f"Retrain failed: {exc}")
                    else:
                        (st.success if result.promoted else st.warning)(result.message)


def render_learning_dashboard() -> None:
    """Renders the Forensic Memory Bank & Continuous Learning dashboard."""
    st.subheader("🧠 Forensic Memory Bank & Dynamic Learning Dashboard")
    st.write(
        "Real-time overview of learned media fingerprints, dynamically-calibrated feature weights, "
        "and user rating distributions across all inspected images, videos, and audio clips."
    )

    img_improver = ImageSelfImprover()
    vid_improver = VideoSelfImprover()
    aud_improver = AudioSelfImprover()

    img_records = img_improver.load_memory()
    vid_records = vid_improver.load_memory()
    aud_records = aud_improver.load_memory()

    img_calib = img_improver.load_calibration()
    vid_calib = vid_improver.load_calibration()
    aud_calib = aud_improver.load_calibration()

    all_records = []
    for r in img_records:
        rc = dict(r)
        rc["modality"] = "IMAGE"
        all_records.append(rc)
    for r in vid_records:
        rc = dict(r)
        rc["modality"] = "VIDEO"
        all_records.append(rc)
    for r in aud_records:
        rc = dict(r)
        rc["modality"] = "AUDIO"
        all_records.append(rc)

    # Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Remembered Media", len(all_records))
    m2.metric("Images / Videos / Audio", f"{len(img_records)} / {len(vid_records)} / {len(aud_records)}")

    ai_count = sum(1 for r in all_records if r.get("user_label") == "AI")
    real_count = sum(1 for r in all_records if r.get("user_label") == "REAL")
    m3.metric("Verified AI Samples", ai_count)
    m4.metric("Verified Real Samples", real_count)

    st.markdown("---")
    render_retrain_panel()
    st.markdown("---")

    # Dynamic Weights and Calibration Offsets
    st.subheader("⚙️ Dynamically-Learned Calibration Weights")
    st.caption("These weights automatically adjust whenever user feedback flags missed artifacts or false positives.")

    t_img, t_vid, t_aud = st.tabs(["🖼️ Image Forensics Weights", "🎬 Video Temporal Weights", "🎵 Audio Acoustic Weights"])
    with t_img:
        i_weights = img_calib.get("feature_weights", {})
        i_offsets = img_calib.get("sensitivity_offsets", {})
        iw1, iw2, iw3, iw4 = st.columns(4)
        iw1.metric("Noise Residual Weight", f"{i_weights.get('noise_residual', 0.35):.2f}")
        iw2.metric("Surface Smoothness Weight", f"{i_weights.get('surface_smoothness', 0.30):.2f}")
        iw3.metric("2D FFT Decay Weight", f"{i_weights.get('fft_decay', 0.20):.2f}")
        iw4.metric("Facial Shading Weight", f"{i_weights.get('facial_shading', 0.25):.2f}")
        io1, io2, io3 = st.columns(3)
        io1.metric("Noise Center Offset", f"{i_offsets.get('noise_center_offset', 0.0):+.2f}")
        io2.metric("Smooth Center Offset", f"{i_offsets.get('smooth_center_offset', 0.0):+.2f}")
        io3.metric("FFT Decay Offset", f"{i_offsets.get('fft_decay_offset', 0.0):+.2f}")

    with t_vid:
        v_weights = vid_calib.get("temporal_weights", {})
        v_thresh = vid_calib.get("motion_thresholds", {})
        vw1, vw2, vw3 = st.columns(3)
        vw1.metric("Frame AI Ratio Weight", f"{v_weights.get('frame_ai_ratio', 0.60):.2f}")
        vw2.metric("Motion Warping Weight", f"{v_weights.get('motion_warping', 0.25):.2f}")
        vw3.metric("Diffusion Flicker Weight", f"{v_weights.get('diffusion_flicker', 0.15):.2f}")
        vt1, vt2, vt3 = st.columns(3)
        vt1.metric("High Warping Var Thresh", f"{v_thresh.get('high_warping_var', 140.0):.1f}")
        vt2.metric("Suspicious Flicker Thresh", f"{v_thresh.get('suspicious_flicker_var', 75.0):.1f}")
        vt3.metric("Unnatural Freeze Thresh", f"{v_thresh.get('unnatural_freeze_var', 0.8):.2f}")

    with t_aud:
        a_weights = aud_calib.get("acoustic_weights", {})
        a_thresh = aud_calib.get("thresholds", {})
        aw1, aw2, aw3, aw4 = st.columns(4)
        aw1.metric("Vocoder Cutoff Weight", f"{a_weights.get('vocoder_cutoff', 0.40):.2f}")
        aw2.metric("Spectral Flatness Weight", f"{a_weights.get('spectral_flatness', 0.30):.2f}")
        aw3.metric("Silence Ratio Weight", f"{a_weights.get('silence_ratio', 0.20):.2f}")
        aw4.metric("High Freq Roll Weight", f"{a_weights.get('high_freq_roll', 0.10):.2f}")
        at1, at2, at3 = st.columns(3)
        at1.metric("Vocoder Min Cutoff", f"{a_thresh.get('vocoder_min_hz', 6500)} Hz")
        at2.metric("Vocoder Max Cutoff", f"{a_thresh.get('vocoder_max_hz', 8200)} Hz")
        at3.metric("Silence Synth Min", f"{a_thresh.get('silence_synthetic_min', 0.12):.2f}")

    st.markdown("---")

    # Remembered Media Records List
    st.subheader(f"📚 Remembered Media Library ({len(all_records)} Entries)")
    if not all_records:
        st.info("No media feedback registered yet. Analyze any image, video, or audio and submit ratings above to populate memory.")
        return

    for rec in reversed(all_records[-15:]):
        mod = rec.get("modality", "MEDIA")
        gt = rec.get("user_label") or rec.get("ground_truth", "Unknown")
        fpath = rec.get("image_path") or rec.get("video_path") or rec.get("audio_path") or rec.get("media_path") or "media"
        fname = Path(fpath).name
        ts = rec.get("timestamp", "")[:19].replace("T", " ")
        notes = rec.get("notes") or rec.get("user_notes", "")
        metrics = rec.get("metrics") or rec.get("forensic_metrics", {})

        with st.expander(f"[{mod}] {fname} — Label: {gt} — {ts}"):
            st.write(f"• **Label:** `{gt}`")
            if notes:
                st.write(f"• **Forensic Notes / Tags:** {notes}")
            if metrics:
                st.json(metrics)


def render_forensic_dossier(dossier: Dict[str, Any]) -> None:
    """Renders the comprehensive NIST-aligned Media Forensics Dossier Report."""
    st.subheader("📋 Official Media Forensics Dossier Report")

    # 1. Authenticity Probabilities (Big Metrics)
    probs = dossier.get("authenticity_probabilities", {})
    p_ai = probs.get("p_ai", 0.0)
    p_real = probs.get("p_real", 0.0)
    p_und = probs.get("p_undecided", 100.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("🤖 P(AI-Generated)", f"{p_ai:.1f}%")
    c2.metric("📷 P(Authentic Capture)", f"{p_real:.1f}%")
    c3.metric("❓ P(Undetermined / OOD)", f"{p_und:.1f}%")

    status = dossier.get("final_status", "UNDETERMINED")
    if status == "LIKELY_SYNTHETIC":
        st.error(f"🚨 **Verdict: {status}** — {dossier.get('reason')}")
    elif status == "LIKELY_AUTHENTIC":
        st.success(f"✅ **Verdict: {status}** — {dossier.get('reason')}")
    elif status == "PARTIALLY_SYNTHETIC_OR_EDITED":
        st.warning(f"⚠️ **Verdict: {status}** — {dossier.get('reason')}")
    else:
        st.info(f"ℹ️ **Verdict: {status}** — {dossier.get('reason')}")

    st.markdown("---")

    # 2. Content Inventory & 3. Localization
    col_inv, col_loc = st.columns(2)
    with col_inv:
        st.markdown("#### 📦 Content Inventory *(What is present)*")
        inv = dossier.get("content_inventory", {})
        st.write(f"• **Depiction / Purpose:** `{inv.get('photographic_purpose', 'General Scene')}`")
        st.write(f"• **Humans:** `{inv.get('persons_count', 0)}` person(s), `{inv.get('faces_count', 0)}` face(s)")

        if inv.get("animals_detected"):
            st.write(f"• **Animals:** {', '.join(inv.get('animal_types', []))}")
        if inv.get("vehicles_detected"):
            st.write(f"• **Vehicles:** {', '.join(inv.get('vehicle_types', []))}")

        items = inv.get("identified_items", [])
        if items:
            st.write(f"• **Identified Items:** {', '.join(f'`{i}`' for i in items[:4])}")

        st.write(f"• **Setting:** `{inv.get('location_context', 'N/A')}` • `{inv.get('setting_type', 'General')}`")
        st.write(f"• **Daytime & Lighting:** `{inv.get('estimated_daytime', 'Daylight')}` ({inv.get('lighting_style', 'Ambient')})")
        st.write(f"• **Tone & Atmosphere:** `{inv.get('color_tone', 'Neutral')}` • `{inv.get('atmospheric_mood', 'Balanced')}`")

        if inv.get("dominant_audio_type") and inv.get("dominant_audio_type") != "N/A":
            st.write(f"• **Audio Stream:** `{inv.get('dominant_audio_type')}` ({inv.get('acoustic_environment', 'Studio')})")

    with col_loc:
        st.markdown("#### 📍 Spatial & Temporal Localization *(Where is the edit)*")
        loc = dossier.get("localization", {})
        has_loc = False
        if loc.get("suspicious_image_area_pct", 0) > 0:
            st.write(f"• **Suspicious Image Region:** `{loc.get('suspicious_image_area_pct', 0):.1f}%` of frame")
            has_loc = True
        if loc.get("suspicious_video_duration_pct", 0) > 0:
            st.write(f"• **Suspicious Video Duration:** `{loc.get('suspicious_video_duration_pct', 0):.1f}%` of timeline")
            has_loc = True
        if loc.get("suspicious_audio_duration_pct", 0) > 0:
            st.write(f"• **Suspicious Audio Duration:** `{loc.get('suspicious_audio_duration_pct', 0):.1f}%` of speech")
            has_loc = True
        if not has_loc:
            st.write("• No localized manipulation clusters isolated.")

    # 4. Provenance & C2PA
    st.markdown("#### 🛡️ Provenance & Content Credentials (C2PA)")
    prov = dossier.get("provenance", {})
    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("C2PA Manifest", "PRESENT" if prov.get("c2pa_present") else "ABSENT")
    pcol2.metric("C2PA Signature", prov.get("c2pa_signature", "Absent"))
    pcol3.metric("AI Declared in Claim", "YES" if prov.get("ai_declared_in_c2pa") else "NO")

    # 5. Global Model Attribution
    attr = dossier.get("model_attribution", {})
    if attr:
        st.markdown("#### 🌐 Global Model Attribution & Fingerprint")
        acol1, acol2, acol3 = st.columns(3)
        acol1.metric("Attributed Generator", attr.get("attributed_model", "Unknown"))
        acol2.metric("Region / Country of Origin", attr.get("region_of_origin", "Global"))
        acol3.metric("Attribution Confidence", f"{int(attr.get('attribution_confidence', 0) * 100)}%")

        candidates = attr.get("top_candidates", [])
        if candidates:
            cand_str = " • ".join(f"**{c.get('model_name')}** ({c.get('region')}): `{c.get('attribution_probability')}%`" for c in candidates)
            st.caption(f"Top Candidate Fingerprints: {cand_str}")

        if attr.get("watermark_detected"):
            st.warning("⚠️ **Impressioned Watermark / Emblem Detected** in media corners.")

    # 6. Cross-Modal Engine
    cm = dossier.get("cross_modal", {})
    if cm and cm.get("is_multimodal"):
        st.markdown("#### 🔄 Cross-Modal Consistency (Audio-Visual Sync)")
        cm_col1, cm_col2 = st.columns(2)
        cm_col1.metric("Synchronization Status", cm.get("cross_modal_status"))
        cm_col2.metric("Tampering Risk", cm.get("tampering_risk"))

    # 7. Forensic Evidence Trail
    st.markdown("#### 🔍 Forensic Evidence Audit Trail")
    trail = dossier.get("evidence_trail", [])
    if trail:
        for t in trail:
            st.markdown(f"• {t}")
    else:
        st.write("No anomalous cues detected.")


# (taxonomy state, extra final_status values, st level, emoji, default label, which percentage to show, plain-English text)
_VERDICT_BANNERS = (
    ("AUTHENTIC_SCREENSHOT", ("AUTHENTIC SCREENSHOT",), "info", "📱", "Authentic Device Screenshot", None,
     "This is an **authentic digital screen capture** from a mobile phone, tablet, laptop, or desktop monitor. "
     "It contains native operating system / application UI rendering, standard display aspect ratios, and no generative synthesis detected."),
    ("AI_ENHANCED_SCREENSHOT", ("AI-ENHANCED SCREENSHOT",), "warning", "📱🧬", "AI-Enhanced / Composite Screenshot", ("p_ai", "AI Signals"),
     "This is a **device screen capture displaying media modified by AI tools** "
     "(e.g. neural face swaps, deepfake filters, generative inpainting, or AI upscaling)."),
    ("AI_GENERATED_SCREENSHOT", ("AI-GENERATED SCREENSHOT",), "error", "📱🤖", "AI-Generated Content Screenshot", ("p_ai", "AI Confidence"),
     "This is a **device screen capture displaying fully synthetic generative AI media** "
     "(e.g. outputs from Midjourney, ChatGPT/DALL-E, Gemini, or diffusion apps captured on a device screen)."),
    ("AUTHENTIC_EDITED", ("AUTHENTIC (CONVENTIONALLY EDITED)", "AUTHENTIC_EDITED"), "info", "🎨",
     "Authentic Created Photograph (Edited / Graphic Design)", None,
     "This is a **real photograph modified with standard editing software** "
     "(e.g. cropping, background removal, alpha transparency cutout, solid studio backdrop replacement, or Canva/Photoshop composition). "
     "The base subject preserves camera sensor noise and natural physical anatomy, with no generative AI synthesis detected."),
    ("AI_ENHANCED_COMPOSITE", ("PARTIALLY_SYNTHETIC_OR_EDITED", "AI-ENHANCED / COMPOSITE"), "warning", "🧬", "AI-Enhanced / Composite (Mix)",
     ("p_ai", "AI Signals"),
     "This is a **hybrid composite combining real capture with neural AI algorithms**. "
     "A real base image was augmented using deep learning models (such as neural face-swapping, generative inpainting, "
     "deep learning object addition/removal, or Topaz Photo AI / Remini neural upscaling and sharpening)."),
    ("FULLY_AI_GENERATED", ("LIKELY_SYNTHETIC", "LIKELY AI-GENERATED"), "error", "🤖", "Fully AI Generated", ("p_ai", "AI Confidence"),
     "This file shows the signature of **end-to-end synthesis by generative diffusion or neural models** "
     "(e.g. Google Gemini / Imagen, Midjourney, DALL-E, Flux, Stable Diffusion), indicated by synthetic surface smoothing, "
     "absence of physical Poisson sensor noise, and generative metadata markers."),
    ("AUTHENTIC_REAL_PHOTOGRAPH", ("LIKELY_AUTHENTIC", "LIKELY REAL"), "success", "🌿", "Authentic Real Photograph", ("p_real", "Confidence"),
     "This file is **consistent with a genuine real-world capture**. "
     "It exhibits camera-like sensor noise (PRNU), organic optical light distribution, and coherent physical geometry, "
     "and no generative AI manipulation was detected. This is a heuristic, uncalibrated estimate, not proof."),
)


def _render_verdict_banner(decision: Dict[str, Any], p_ai: float, p_real: float) -> None:
    tax_state = decision.get("taxonomy_state")
    status = decision.get("final_status", "UNDETERMINED")
    label = decision.get("taxonomy_label")
    pcts = {"p_ai": p_ai, "p_real": p_real}
    for state, statuses, level, emoji, default_label, pct, text in _VERDICT_BANNERS:
        if tax_state == state or status in (state, *statuses):
            suffix = f" ({pcts[pct[0]]:.1f}% {pct[1]})" if pct else ""
            getattr(st, level)(f"{emoji} **Verdict: {label or default_label}**{suffix}")
            st.markdown(f"📌 **Plain-English Verdict:** {text}")
            return
    st.info("❓ **Verdict: Undetermined / Mixed Signals**")
    st.markdown(
        "📌 **Plain-English Verdict:** Forensic indicators show balanced evidence or heavy platform compression. "
        "The system maintains formal scientific uncertainty rather than guessing."
    )


def _render_probability_gauges(p_ai: float, p_real: float, p_und: float) -> None:
    st.markdown("##### 📈 Authenticity Probabilities")
    st.caption("Heuristic, uncalibrated probabilities separating synthetic generation from genuine capture.")
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("🤖 AI-Generated", f"{p_ai:.1f}%")
    col_p2.metric("📷 Camera / Real", f"{p_real:.1f}%")
    col_p3.metric("❓ Undetermined", f"{p_und:.1f}%")
    st.progress(
        min(1.0, max(0.0, p_ai / 100.0)),
        text=f"Generative AI Probability: {p_ai:.1f}% AI vs {p_real:.1f}% Real"
    )


def _render_depiction(decision: Dict[str, Any]) -> None:
    st.markdown("##### 🖼️ What Does This File Depict & Represent?")
    inv = decision.get("content_inventory", {})
    st.markdown(f"**Summary:** *{decision.get('what_it_represents') or 'Visual media capture'}*")

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.write(f"• **Depiction / Purpose:** `{inv.get('photographic_purpose', 'General Scene')}`")
        st.write(f"• **Setting & Environment:** `{inv.get('location_context', 'General')}` • `{inv.get('setting_type', 'Ambient')}`")
        st.write(f"• **Daytime & Lighting:** `{inv.get('estimated_daytime', 'Daylight')}` ({inv.get('lighting_style', 'Ambient')})")
        st.write(f"• **Atmosphere & Tone:** `{inv.get('color_tone', 'Neutral')}` • `{inv.get('atmospheric_mood', 'Balanced')}`")
    with c_col2:
        st.write(f"• **Living Humans:** `{inv.get('persons_count', 0)}` person(s), `{inv.get('faces_count', 0)}` face(s)")
        if inv.get("animals_detected"):
            st.write(f"• **Animals:** {', '.join(inv.get('animal_types', []))}")
        if inv.get("vehicles_detected"):
            st.write(f"• **Vehicles:** {', '.join(inv.get('vehicle_types', []))}")
        items = inv.get("identified_items", [])
        if items:
            st.write(f"• **Identified Items:** {', '.join(f'`{i}`' for i in items[:4])}")
        if inv.get("dominant_audio_type") and inv.get("dominant_audio_type") != "N/A":
            st.write(f"• **Audio Stream:** `{inv.get('dominant_audio_type')}` ({inv.get('acoustic_environment', 'Studio')})")


def _render_optical_pillar(decision: Dict[str, Any], p_real: float, tax_state: Optional[str]) -> None:
    st.markdown("###### 🔬 Optical & Sensor Physics")
    prov = decision.get("provenance", {})
    if prov.get("hardware_make"):
        st.write(f"• **Camera Hardware (EXIF tags, unauthenticated):** `{prov.get('hardware_make')} {prov.get('hardware_model') or ''}`".strip())
    else:
        st.write("• **Camera Hardware:** No embedded hardware EXIF (common for stripped web/chat uploads)")

    if p_real >= 60.0 or tax_state == "AUTHENTIC_REAL_PHOTOGRAPH":
        st.write("• **PRNU Sensor Noise:** Preserved natural Poisson shot noise & physical optical grain.")
        st.write("• **Lens Optics:** Authentic depth of field, coherent lighting vectors & unmanipulated geometry.")
    elif tax_state == "AUTHENTIC_EDITED":
        st.write("• **PRNU Sensor Noise:** Base subject retains authentic camera sensor noise.")
        st.write("• **Software Composition:** Background boundary isolated or studio backdrop replaced.")
    else:
        st.write("• **PRNU Sensor Noise:** Latent diffusion denoising detected; physical shot noise absent.")
        st.write("• **Lens Optics:** Algorithmic blur/bokeh and synthetic illumination rendering.")


def _render_provenance_pillar(decision: Dict[str, Any]) -> None:
    st.markdown("###### 🏷️ Provenance & Generator Watermarks")
    prov = decision.get("provenance", {})
    if decision.get("model_attribution", {}).get("watermark_detected", False):
        st.write("• **AI Watermark:** ⚠️ Google Gemini 4-pointed sparkle emblem detected in corner quadrant.")
    else:
        st.write("• **AI Watermark:** No visible generator watermark or logo emblem detected.")

    if prov.get("c2pa_present"):
        st.write(f"• **C2PA Credentials:** Markers found ({prov.get('c2pa_signature', 'unverified')}); presence only, not cryptographically verified")
    else:
        st.write("• **C2PA Credentials:** Absent (neutral/stripped)")
    if prov.get("software"):
        st.write(f"• **Software Tag:** `{prov.get('software')}`")


def _render_attribution(decision: Dict[str, Any], tax_state: Optional[str]) -> None:
    st.markdown("##### 🌐 Global AI Generator Attribution")
    attr = decision.get("model_attribution", {})
    is_authentic_media = tax_state in ("AUTHENTIC_REAL_PHOTOGRAPH", "AUTHENTIC_EDITED") and not (
        attr.get("attributed_model") and "Topaz" in attr.get("attributed_model")
    )
    if is_authentic_media:
        st.success("✅ **Camera / Optical Origin:** No AI generator footprint detected; consistent with an authentic photographic capture.")
        return
    acol1, acol2, acol3 = st.columns(3)
    acol1.metric("Attributed Model", attr.get("attributed_model", "Unattributable"))
    acol2.metric("Region / Country", attr.get("region_of_origin", "Global"))
    acol3.metric("Attribution Match", f"{int(attr.get('attribution_confidence', 0) * 100)}%")
    candidates = attr.get("top_candidates", [])
    if candidates:
        cand_items = [f"**{c.get('model_name')}** ({c.get('region')}): `{c.get('attribution_probability')}%`" for c in candidates[:3]]
        st.caption("Top Candidates: " + " • ".join(cand_items))


def _render_localization(decision: Dict[str, Any]) -> None:
    loc = decision.get("localization", {})
    susp_area = float(loc.get("suspicious_image_area_pct", 0))
    susp_vid = float(loc.get("suspicious_video_duration_pct", 0))
    susp_aud = float(loc.get("suspicious_audio_duration_pct", 0))
    if not (susp_area > 0 or susp_vid > 0 or susp_aud > 0):
        return
    st.markdown("##### 📍 Manipulation Localization")
    if susp_area > 0:
        st.write(f"• **Localized Anomaly Region:** `{susp_area:.1f}%` of frame area exhibits neural smoothing or background replacement.")
    if susp_vid > 0:
        st.write(f"• **Manipulated Video Duration:** `{susp_vid:.1f}%` of timeline.")
    if susp_aud > 0:
        st.write(f"• **Manipulated Speech Duration:** `{susp_aud:.1f}%` of audio track.")


def render_analysis_right_panel(decision: Dict[str, Any], content_res: Dict[str, Any], modality: str = "image") -> None:
    """
    Renders the unified, simplified forensic analysis and media intelligence panel:
    verdict banner, probabilities, depiction, key evidence, per-pillar breakdown, attribution,
    localization and the expandable audit trail.
    """
    if not decision:
        return

    st.subheader("📊 Forensic Analysis & Media Intelligence")
    probs = decision.get("authenticity_probabilities", {})
    p_ai = float(probs.get("p_ai", 0.0))
    p_real = float(probs.get("p_real", 0.0))
    p_und = float(probs.get("p_undecided", 100.0))
    tax_state = decision.get("taxonomy_state")

    _render_verdict_banner(decision, p_ai, p_real)
    _render_probability_gauges(p_ai, p_real, p_und)
    st.markdown("---")
    _render_depiction(decision)
    st.markdown("---")

    st.markdown("##### 🔍 Why This Classification? (Key Forensic Evidence)")
    tax_reasons = decision.get("taxonomy_reasons", [])
    if tax_reasons:
        for r in tax_reasons:
            st.markdown(f"• {r}")
    else:
        st.write("• Forensic analysis completed across optical, biological, and metadata domains.")

    with st.expander("🔎 View Forensic Evidence Breakdown by Pillar", expanded=True):
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            _render_optical_pillar(decision, p_real, tax_state)
        with ecol2:
            _render_provenance_pillar(decision)

    _render_attribution(decision, tax_state)
    _render_localization(decision)

    with st.expander("🔍 Deep Forensic Audit Trail (For Professionals & Auditors)", expanded=False):
        trail = decision.get("evidence_trail", [])
        if trail:
            for t in trail:
                st.markdown(f"• {t}")
        else:
            st.write("No anomalous cues detected.")


def render_bottom_feedback_panel(
    media_path: str | Path,
    modality: str,
    forensic_data: Dict[str, Any],
    profile_data: Dict[str, Any],
    decision: Dict[str, Any],
    unique_key: str,
) -> None:
    """
    Renders the unified single wide bottom panel across the entire screen:
    1. Multi-criteria 1-10 rating sliders & online auto-retraining.
    2. Deep technical media, pixel & signal specifications.
    3. Complete JSON forensic dossier export and download.
    """
    st.markdown("---")
    st.subheader("📝 Forensic Feedback, Continual Retraining & Deep Technical Specifications")
    st.caption(
        "Help the engine learn and continuously improve: rate the authenticity out of 10. "
        "The system will remember this file's pixel profile, metadata, and physics metrics, "
        "dynamically auto-updating its neural weights and detection thresholds."
    )

    tab_fb, tab_specs, tab_json = st.tabs([
        "⭐ Rate Media & Dynamically Retrain Algorithm",
        "🔬 Deep Media, Pixel & Signal Specifications",
        "📥 Export Complete Forensic JSON Dossier",
    ])

    with tab_fb:
        _render_feedback_form(
            media_path, modality, forensic_data, unique_key,
            "Describe specific telltales: waxy skin, warped fingers, abnormal reflections, vocoder cutoff...",
        )

    with tab_specs:
        render_media_specs(profile_data)

    with tab_json:
        _render_dossier_export(media_path, modality, decision, profile_data, forensic_data, unique_key)


def _render_dossier_export(
    media_path: str | Path, modality: str, decision: Dict[str, Any], profile_data: Dict[str, Any],
    forensic_data: Dict[str, Any], unique_key: str,
) -> None:
    st.markdown("##### 📋 Complete Multi-Modal Forensic Dossier")
    full_dossier_payload = {
        "media_file": str(media_path),
        "modality": modality,
        "dossier": decision,
        "specifications": profile_data,
        "forensic_signals": forensic_data,
    }
    st.json(full_dossier_payload)
    st.download_button(
        label="💾 Download Official Forensic Dossier (JSON)",
        data=json.dumps(full_dossier_payload, indent=2, default=str),
        file_name=f"forensic_dossier_{Path(media_path).stem}.json",
        mime="application/json",
        key=f"{unique_key}_dl_btn",
    )


class _ProfileView(NamedTuple):
    """Lookups derived once from the profile dict and shared by the section renderers."""

    profile_data: Any
    geom: Any
    disp: Any
    pcol: Any
    exif: Any
    phys: Any
    file_id: Any
    fname: Any
    fmt: Any
    size_kb: Any
    size_mb: Any
    sha: Any
    src: Any


def _render_profile_section_1(v: _ProfileView) -> None:
    """1. Media & Container Origin"""
    st.markdown("##### 📌 File Identity & Provenance Container")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("File Name", v.fname[:20] + "..." if len(v.fname) > 23 else v.fname)
    f2.metric("File Size", f"{v.size_kb:.1f} KB ({v.size_mb:.2f} MB)")
    f3.metric("Format / MIME", f"{v.fmt} • {v.profile_data.get('mime_type', 'image/' + str(v.fmt).lower())}")
    f4.metric("Source Origin", v.src)
    f5.metric("SHA-256 (Prefix)", f"{v.sha[:10]}..." if v.sha else "N/A")

    st.markdown("---")


def _render_profile_section_2(v: _ProfileView) -> None:
    """2. Dimensions, Spatial Geometry & DPI"""
    st.markdown("##### 📐 Dimensions, Geometry & Display Resolution (DPI)")
    w = v.geom.get("width", v.profile_data.get("width", 0))
    h = v.geom.get("height", v.profile_data.get("height", 0))
    tot_pix = v.geom.get("total_pixels", int(w * h))
    mp = v.geom.get("megapixels", round(tot_pix / 1_000_000.0, 2))
    asp_str = v.geom.get("aspect_ratio_str", f"{v.geom.get('aspect_ratio', 0.0)}:1")
    orient = v.geom.get("orientation", "Landscape")

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Width x Height", f"{w} × {h} px")
    g2.metric("Megapixels (MP)", f"{mp:.2f} MP")
    g3.metric("Total Pixel Count", f"{tot_pix:,} px")
    g4.metric("Aspect Ratio", asp_str)
    g5.metric("Orientation", orient)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("DPI Resolution", v.disp.get("dpi_str", "72 x 72 DPI"))
    d2.metric("Bit Depth", v.disp.get("bit_depth", "24-bit (3x8-bit)"))
    d3.metric("Color Space", v.disp.get("color_space", "Standard sRGB"))
    d4.metric("Alpha Channel", "Present (RGBA)" if v.disp.get("has_alpha_channel") else "None (RGB)")

    st.markdown("---")


def _render_profile_section_3(v: _ProfileView) -> None:
    """3. Device & Acquisition EXIF Parameters"""
    st.markdown("##### 📷 Hardware Device & EXIF Acquisition Metadata")
    has_cam = bool(v.exif.get("camera_make") or v.exif.get("camera_model"))
    cam_make = v.exif.get("camera_make") or "Unspecified Hardware"
    cam_model = v.exif.get("camera_model") or ""
    lens = v.exif.get("lens_model") or "Standard Lens / Unspecified"
    shutter = v.exif.get("exposure_time") or "N/A"
    aperture = v.exif.get("aperture") or "N/A"
    iso_val = v.exif.get("iso") or "N/A"
    focal = v.exif.get("focal_length") or "N/A"
    flash = v.exif.get("flash", "Not Fired")
    wb = v.exif.get("white_balance", "Auto")
    gps_str = v.exif.get("gps_details", {}).get("coordinates_str", "Not Embedded")
    software = v.exif.get("software") or "None (Clean Exif)"
    date_str = v.exif.get("date_time") or "Unknown / Stripped"

    if has_cam:
        st.success(f"Camera hardware EXIF tags present (unauthenticated): **{cam_make} {cam_model}** | Lens: `{lens}`")
    else:
        st.info("ℹ️ No embedded hardware camera EXIF tags found (characteristic of stripped web/social uploads or AI synthesis).")

    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Shutter Speed", shutter)
    e2.metric("Aperture", aperture)
    e3.metric("ISO Sensitivity", str(iso_val))
    e4.metric("Focal Length", focal)
    e5.metric("Flash / White Balance", f"{flash} • {wb}")

    st.caption(f"• **Capture Date:** `{date_str}` | **Software Tag:** `{software}` | **GPS Coordinates:** `{gps_str}`")

    st.markdown("---")


def _render_profile_section_4(v: _ProfileView) -> None:
    """4. Pixel-by-Pixel Color & Photometric Statistics"""
    st.markdown("##### 🎨 Pixel-by-Pixel Photometric Statistics & Dynamic Range")
    entropy = v.pcol.get("shannon_entropy_bpp", v.profile_data.get("pixel_entropy", 0.0))
    lum_mean = v.pcol.get("luminance_mean", 0.0)
    lum_median = v.pcol.get("luminance_median", lum_mean)
    lum_range = v.pcol.get("dynamic_range", 0)
    lum_min = v.pcol.get("luminance_min", 0)
    lum_max = v.pcol.get("luminance_max", 255)
    clip_hi = v.pcol.get("highlight_clipped_pct", 0.0)
    clip_sh = v.pcol.get("shadow_crushed_pct", 0.0)

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Shannon Entropy", f"{entropy:.3f} bits/px")
    p2.metric("Mean / Median Lum", f"{lum_mean:.1f} / {lum_median:.1f}")
    p3.metric("Dynamic Range", f"{lum_range} ({lum_min}..{lum_max})")
    p4.metric("Clipped Highlights", f"{clip_hi:.2f}%")
    p5.metric("Crushed Shadows", f"{clip_sh:.2f}%")

    c_means = v.pcol.get("channel_means", {})
    c_stds = v.pcol.get("channel_stds", {})
    if "red" in c_means:
        st.caption(
            f"• **Channel Distributions (Mean ± Std):** "
            f"Red: `{c_means.get('red', 0.0):.1f} ± {c_stds.get('red', 0.0):.1f}` | "
            f"Green: `{c_means.get('green', 0.0):.1f} ± {c_stds.get('green', 0.0):.1f}` | "
            f"Blue: `{c_means.get('blue', 0.0):.1f} ± {c_stds.get('blue', 0.0):.1f}` | "
            f"Unique Quantized Colors: `{v.pcol.get('unique_quantized_colors', 0):,}`"
        )


def _render_profile_section_5(v: _ProfileView) -> None:
    """5. Dominant Color Palette Swatches"""
    st.markdown("##### 🌈 Dominant Color Palette & Canvas Coverage")
    palette = v.pcol.get("dominant_palette", [])
    if palette:
        palette_html = '<div style="display: flex; flex-wrap: wrap; gap: 14px; margin-top: 6px; margin-bottom: 12px;">'
        for c in palette:
            palette_html += (
                f'<div style="background: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 8px; border: 1px solid #444; text-align: center; min-width: 100px;">'
                f'<div style="width: 100%; height: 32px; background-color: {c["hex"]}; border-radius: 4px; border: 1px solid #222; margin-bottom: 6px;"></div>'
                f'<div style="font-size: 12px; font-weight: bold;">{c.get("color_name", c["hex"])}</div>'
                f'<div style="font-size: 11px; color: #aaa;"><code>{c["hex"]}</code> • <b>{c["coverage_pct"]}%</b></div>'
                f'</div>'
            )
        palette_html += "</div>"
        st.markdown(palette_html, unsafe_allow_html=True)
    else:
        st.write("• Color palette extracted.")

    st.markdown("---")


def _render_profile_section_6(v: _ProfileView) -> None:
    """6. Raw Physical Noise & Forensic Signals"""
    st.markdown("##### 🔬 Raw Physical Noise & Frequency Spectrum Properties")
    prnu_noise = v.phys.get("prnu_noise_mean", 0.0)
    flat_noise = v.phys.get("flat_region_noise_mean", prnu_noise)
    smoothness = v.phys.get("surface_smoothness_index", 0.0)
    fft_alpha = v.phys.get("fft_decay_alpha", 2.05)
    edges_pct = v.phys.get("canny_edge_pct", 0.0)
    dark_lines = v.phys.get("dark_line_art_pct", 0.0)
    sharpness = v.phys.get("laplacian_sharpness_var", 0.0)

    n1, n2, n3, n4, n5 = st.columns(5)
    n1.metric("PRNU Sensor Noise", f"{prnu_noise:.3f}")
    n2.metric("Flat-Region Noise", f"{flat_noise:.3f}")
    n3.metric("Surface Smoothness", f"{smoothness:.3f}")
    n4.metric("Fourier Decay Alpha", f"{fft_alpha:.2f}")
    n5.metric("Focus / Sharpness Var", f"{sharpness:.1f}")

    st.caption(
        f"• **Edge Density:** `{edges_pct:.2f}%` Canny edges | "
        f"• **Contour Density:** `{dark_lines:.2f}%` dark line-art strokes | "
        f"• **Optical Shot Noise Baseline:** `{'Preserved (>= 1.20)' if flat_noise >= 1.20 else 'Synthetic / Denoiser Absence (< 1.20)'}`"
    )


_PROFILE_SECTIONS = (_render_profile_section_1, _render_profile_section_2, _render_profile_section_3, _render_profile_section_4, _render_profile_section_5, _render_profile_section_6)


def render_pre_analysis_specifications(
    profile_data: Dict[str, Any], expanded: bool = True, source: str = "User Upload"
) -> None:
    """
    Renders Stage 1: Pre-Analysis Feature & Metadata Extraction.
    Shows dimensions, DPI, pixel-by-pixel information, EXIF details, noise, and colors
    before running prediction/detection.
    """
    if not profile_data:
        return

    st.markdown("### 📋 Stage 1: Pre-Analysis Feature & Metadata Extraction")
    st.caption(
        "Instant extraction of image dimensions, DPI, pixel-by-pixel statistics, EXIF hardware parameters, "
        "color palette, and raw physical noise prior to running AI detection models."
    )

    geom = profile_data.get("spatial_geometry") or profile_data.get("pixel_specifications") or {}
    disp = profile_data.get("display_attributes") or {}
    pcol = profile_data.get("pixel_color_profile") or {}
    exif = profile_data.get("exif_device_details") or profile_data.get("provenance_metadata") or {}
    phys = profile_data.get("raw_physical_signals") or {}
    file_id = profile_data.get("file_identity") or {}

    fname = profile_data.get("filename") or file_id.get("filename", "Uploaded File")
    fmt = profile_data.get("format") or file_id.get("container_format", "N/A")
    size_kb = profile_data.get("file_size_kb") or file_id.get("size_kb", 0.0)
    size_mb = profile_data.get("file_size_mb", size_kb / 1024.0)
    sha = profile_data.get("sha256") or file_id.get("sha256", "")
    src = profile_data.get("source", source)

    v = _ProfileView(profile_data=profile_data, geom=geom, disp=disp, pcol=pcol, exif=exif, phys=phys, file_id=file_id, fname=fname, fmt=fmt, size_kb=size_kb, size_mb=size_mb, sha=sha, src=src)

    with st.expander("🔍 View Complete Pre-Analysis Image Profile & Pixel Telemetry", expanded=expanded):
        for render_section in _PROFILE_SECTIONS:
            render_section(v)


_DIMENSION_TABS = (
    ("1. Provenance", "dimension_1_hardware_provenance"),
    ("2. Pixel Specs", "dimension_2_pixel_architecture"),
    ("3. PRNU Noise", "dimension_3_prnu_sensor_noise"),
    ("4. Smoothness", "dimension_4_surface_smoothness"),
    ("5. Fourier FFT", "dimension_5_fourier_fft_decay"),
    ("6. Genre & Scene", "dimension_6_subject_genre"),
    ("7. Visual Medium", "dimension_7_visual_medium"),
    ("8. Sensor Spectrum", "dimension_8_sensor_spectrum"),
    ("9. AI Attribution", "dimension_9_generative_attribution"),
)

def _render_dimension_1(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 1')}")
    if d.get("container_atoms"):
        st.write(f"• **Container Atoms:** `{', '.join(str(a) for a in d.get('container_atoms', []))}`")
    if d.get("container_format"):
        st.write(f"• **Audio Stream Format:** `{d.get('container_format')}`")
    if d.get("camera_hardware") or d.get("hardware_origin"):
        cam_name = d.get("camera_hardware") or d.get("hardware_origin")
        lens = f" | Lens: `{d.get('lens_model')}`" if d.get("lens_model") else ""
        st.write(f"• **Hardware Identity:** `{cam_name}`{lens}")
    if d.get("iptc_digital_source_type"):
        st.write(f"• **IPTC Digital Source Type:** `{d.get('iptc_digital_source_type')}`")
    st.write(f"• **C2PA Content Credentials:** `{d.get('c2pa_status')}`")
    if d.get("software_signature"):
        st.write(f"• **Software Fingerprint:** `{d.get('software_signature')}` | Date: `{d.get('date_taken')}`")
    if d.get("gps_coordinates"):
        st.write(f"• **GPS Location:** `{d.get('gps_coordinates')}`")
    st.info(f"Verdict: **{d.get('provenance_verdict')}**")


def _render_dimension_2(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 2')}")
    if d.get("geometry"):
        st.write(f"• **Geometry:** `{d.get('geometry')}` ({d.get('aspect_ratio')}, {d.get('orientation', 'Standard')})")
    if d.get("frame_rate_fps"):
        st.write(f"• **Frame Rate & Duration:** `{d.get('frame_rate_fps')}` • `{d.get('duration')}` ({d.get('total_frames')} frames)")
    if d.get("sample_rate"):
        st.write(f"• **Acoustic Geometry:** `{d.get('sample_rate')}` • `{d.get('duration')}` • `{d.get('channels')}` • `{d.get('bit_depth')}`")
    if d.get("dpi"):
        st.write(f"• **DPI & Bit Depth:** `{d.get('dpi')}` • `{d.get('bit_depth')}` ({d.get('color_space')})")
    if d.get("shannon_entropy"):
        st.write(f"• **Shannon Entropy:** `{d.get('shannon_entropy')}`")
    if d.get("luminance_dynamic_range"):
        st.write(f"• **Luminance Range:** `{d.get('luminance_dynamic_range')}`")
    if d.get("clipping_profile"):
        st.write(f"• **Clipping Profile:** `{d.get('clipping_profile')}`")
    if d.get("unique_colors_quantized"):
        st.write(f"• **Unique Quantized Colors:** `{d.get('unique_colors_quantized', 0):,}`")


def _render_dimension_3(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 3')}")
    if d.get("temporal_warping_variance") is not None:
        st.write(f"• **Inter-Frame Motion Variance:** `{d.get('temporal_warping_variance', 0.0):.2f}`")
    if d.get("vocoder_cutoff_hz") is not None:
        st.write(f"• **Vocoder Brickwall Cutoff:** `{d.get('vocoder_cutoff_hz', 0):,} Hz`")
    if d.get("prnu_residual_mean") is not None:
        st.write(f"• **PRNU Residual Mean:** `{d.get('prnu_residual_mean', 0.0):.3f}` | Flat Region Noise: `{d.get('flat_region_noise', 0.0):.3f}`")
    if d.get("mathematical_physics"):
        st.write(f"• **Mathematical Formulation:** `{d.get('mathematical_physics')}`")
    if d.get("diagnosis"):
        if d.get("is_natural_shot_noise") or d.get("temporal_warping_variance", 0.0) < 140.0:
            st.success(f"Diagnosis: {d.get('diagnosis')}")
        else:
            st.error(f"Diagnosis: {d.get('diagnosis')}")


def _render_dimension_4(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 4')}")
    if d.get("diffusion_flickering_ratio") is not None:
        st.write(f"• **Diffusion Flickering Ratio:** `{d.get('diffusion_flickering_ratio', 0.0):.3f}`")
    if d.get("spectral_flatness_wiener") is not None:
        st.write(f"• **Wiener Spectral Flatness:** `{d.get('spectral_flatness_wiener', 0.0):.4f}`")
    if d.get("smoothness_index") is not None:
        st.write(f"• **Surface Smoothness Index:** `{d.get('smoothness_index', 0.0):.3f}`")
    st.write("• **Physical Principle:** Organic micro-textures maintain high local bilateral variance; neural generative models over-smooth.")
    if d.get("diagnosis"):
        if d.get("is_diffusion_smoothed") or d.get("is_flickering_detected") or d.get("is_synthetic_smoothness"):
            st.warning(f"⚠️ {d.get('diagnosis')}")
        else:
            st.success(f"✅ {d.get('diagnosis')}")


def _render_dimension_5(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 5')}")
    if d.get("digital_silence_ratio") is not None:
        st.write(f"• **Digital Zero Silence Ratio:** `{d.get('digital_silence_ratio')}`")
    if d.get("keyframe_fft_alpha") is not None:
        st.write(f"• **Keyframe Spectral Alpha:** `{d.get('keyframe_fft_alpha', 2.05):.3f}`")
    if d.get("spectral_decay_alpha") is not None:
        st.write(f"• **Spectral Decay Alpha:** `{d.get('spectral_decay_alpha', 2.05):.3f}`")
    if d.get("mathematical_physics"):
        st.write(f"• **Field Law:** `{d.get('mathematical_physics')}`")
    if d.get("diagnosis"):
        if d.get("is_anomalous_decay") or d.get("has_digital_zero_silence"):
            st.warning(f"⚠️ {d.get('diagnosis')}")
        else:
            st.success(f"✅ {d.get('diagnosis')}")


def _render_dimension_6(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 6')}")
    if d.get("audio_type"):
        st.write(f"• **Dominant Audio Type:** `{d.get('audio_type')}` • `{d.get('estimated_speakers')} speaker(s)`")
        st.write(f"• **Acoustic Environment:** `{d.get('acoustic_environment')}`")
    else:
        st.write(f"• **Primary Visual Genre:** `{d.get('primary_genre')}`")
        st.write(f"• **Living Entities:** `{d.get('persons_count', 0)}` person(s), `{d.get('faces_count', 0)}` face(s) (Stylized Character: `{d.get('is_stylized_character')}`)")
        st.write(f"• **Setting & Atmosphere:** `{d.get('setting_and_environment')}` • `{d.get('atmospheric_mood')}`")
        st.write(f"• **Daytime & Lighting Quality:** `{d.get('lighting_and_daytime')}`")


def _render_dimension_7(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 7')}")
    if d.get("acoustic_delivery_tone"):
        st.write(f"• **Acoustic Delivery Tone:** `{d.get('acoustic_delivery_tone')}`")
    if d.get("visual_medium"):
        st.write(f"• **Visual Medium:** `{d.get('visual_medium')}`")
    if d.get("synthesis_type"):
        st.write(f"• **Temporal Synthesis Type:** `{d.get('synthesis_type')}`")
    if d.get("is_digital_art") is not None:
        st.write(f"• **Digital Art / Painting Detected:** `{d.get('is_digital_art')}`")
    if d.get("canny_edge_density_pct") is not None:
        st.write(f"• **Canny Edge Density:** `{d.get('canny_edge_density_pct', 0.0):.2f}%` | Dark Line Art: `{d.get('dark_line_contours_pct', 0.0):.2f}%`")
    if d.get("laplacian_focus_sharpness") is not None:
        st.write(f"• **Focus Sharpness:** `{d.get('laplacian_focus_sharpness', 0.0):.1f}`")


def _render_dimension_8(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 8')}")
    if d.get("bandwidth_class"):
        st.write(f"• **Acoustic Bandwidth:** `{d.get('bandwidth_class')}` | Nyquist Ceiling: `{d.get('nyquist_ceiling')}`")
    if d.get("sensor_spectrum") or d.get("sensor_modality"):
        st.write(f"• **Acquisition Modality:** `{d.get('sensor_spectrum') or d.get('sensor_modality')}`")
    if d.get("color_channels") is not None:
        st.write(f"• **Color Channels:** `{d.get('color_channels', 3)}` channels")
    if d.get("diagnosis"):
        st.write(f"• **Diagnostic:** {d.get('diagnosis')}")


def _render_dimension_9(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 9')}")
    reg = f" ({d.get('region_of_origin')})" if d.get("region_of_origin") else ""
    st.write(f"• **Attributed Generator:** `{d.get('attributed_model')}`{reg}")
    st.write(f"• **Attribution Confidence:** `{d.get('confidence', '0%')}`")
    st.write(f"• **Watermark Detected:** `{'YES' if d.get('watermark_detected') else 'None'}`")
    if d.get("suspicious_duration_pct") is not None:
        st.write(f"• **Suspicious Duration Fraction:** `{d.get('suspicious_duration_pct')}`")
    if d.get("spatial_manipulated_area_pct") is not None:
        st.write(f"• **Spatial Manipulation Area:** `{d.get('spatial_manipulated_area_pct', 0.0):.1f}%` of frame")


_DIMENSION_RENDERERS = (_render_dimension_1, _render_dimension_2, _render_dimension_3, _render_dimension_4, _render_dimension_5, _render_dimension_6, _render_dimension_7, _render_dimension_8, _render_dimension_9)


def render_nine_dimensions_breakdown(nine_dims: Dict[str, Any], expanded: bool = True) -> None:
    """
    Renders Stage 2: 9-Dimensions Forensic Taxonomy Analyzer.
    Exhaustively covers all 9 analytical dimensions formalized in GLOBAL_IMAGE_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md.
    """
    if not nine_dims:
        return

    st.markdown("### 🌐 Stage 2: 9-Dimensions Forensic Taxonomy Analyzer")
    st.caption("Evaluation across all 9 forensic dimensions (referencing NIST OpenMFC, C2PA and IPTC concepts; informational, not a certified compliance check):")

    with st.expander("🔬 View All 9 Analytical Dimensions Audit", expanded=expanded):
        tabs = st.tabs([title for title, _ in _DIMENSION_TABS])
        for i, (tab, (_title, key)) in enumerate(zip(tabs, _DIMENSION_TABS), start=1):
            with tab:
                _DIMENSION_RENDERERS[i - 1](nine_dims.get(key, {}))


def render_image_type_and_category(
    decision: Dict[str, Any],
    content_res: Dict[str, Any],
    ai_result: Dict[str, Any],
    nine_dims: Optional[Dict[str, Any]] = None,
) -> None:
    """Renders Stage 3: Image Type and Category Identification."""
    st.markdown("### 🏷️ Stage 3: Image Type & Category Identification")
    st.caption("Classified taxonomy ontology, IPTC digital source type, visual genre, medium, and sensor spectrum.")

    tax_label = decision.get("taxonomy_label") or ai_result.get("taxonomy_label", "Analyzed Media")
    tax_state = decision.get("taxonomy_state") or ai_result.get("taxonomy_state", "UNDECIDED")
    iptc_code = (
        nine_dims.get("dimension_1_hardware_provenance", {}).get("iptc_digital_source_type")
        if nine_dims
        else IPTC_SOURCE_TYPE_MAPPING.get(tax_state, "digitalsourcetype:digitalCapture")
    )
    genre = content_res.get("purpose_and_depiction", {}).get("primary_genre") or ai_result.get("subject_genre", "General Scene")
    medium = ai_result.get("visual_medium", "Photographic Capture")
    spectrum = ai_result.get("sensor_spectrum", "Visible Spectrum (Bayer RGB)")
    layout = content_res.get("purpose_and_depiction", {}).get("document_layout", "None (Standard Visual Content)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Taxonomy State", tax_label)
    col2.metric("Visual Genre", genre)
    col3.metric("Visual Medium", medium)

    col4, col5, col6 = st.columns(3)
    col4.metric("IPTC Digital Source", iptc_code.split("(")[0].strip())
    col5.metric("Sensor Modality", spectrum.split("(")[0].strip())
    col6.metric("Document / Information Layout", layout)


def render_quantified_detections_and_inventory(
    decision: Dict[str, Any],
    content_res: Dict[str, Any],
    ai_result: Dict[str, Any],
    profile_data: Dict[str, Any],
) -> None:
    """Renders Stage 4: Algorithmic Detection & Quantified Inventory (Percentages & Counts)."""
    st.markdown("### 🔢 Stage 4: Algorithmic Detection & Quantified Inventory")
    st.caption("Precise percentages and counts for every entity, item, pixel property, and forensic probe identified in the image.")

    probs = decision.get("authenticity_probabilities", {})
    p_ai = float(probs.get("p_ai", ai_result.get("ai_percentage", 0.0)))
    p_real = float(probs.get("p_real", ai_result.get("real_percentage", 0.0)))
    p_und = float(probs.get("p_undecided", ai_result.get("undecided_percentage", 100.0)))
    ai_area = float(ai_result.get("ai_spatial_area_pct", 0.0))

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("🤖 P(AI-Generated)", f"{p_ai:.1f}%")
    q2.metric("📷 P(Authentic Camera)", f"{p_real:.1f}%")
    q3.metric("❓ P(Undetermined)", f"{p_und:.1f}%")
    q4.metric("🔴 Manipulated Area", f"{ai_area:.1f}%")

    st.progress(min(1.0, max(0.0, p_ai / 100.0)), text=f"AI Synthesis Ratio: {p_ai:.1f}% AI vs {p_real:.1f}% Real")

    # Counts & Quantified Elements
    inv = content_res or {}
    humans = inv.get("living_entities", {}).get("humans") or inv.get("entities", {}).get("humans", {})
    persons_cnt = humans.get("persons_count", inv.get("persons_count", 0))
    faces_cnt = humans.get("faces_count", inv.get("faces_count", 0))
    is_char = humans.get("is_stylized_character", False)

    items = inv.get("contents_and_items", {}).get("identified_items", [])
    text_cnt = inv.get("contents_and_items", {}).get("text_regions_count", 0)
    animals = inv.get("living_entities", {}).get("animals", {}).get("animal_types", [])
    vehicles = inv.get("vehicles", {}).get("vehicle_types", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Living Humans", f"{persons_cnt} person(s), {faces_cnt} face(s)" + (" (Stylized)" if is_char else ""))
    c2.metric("Identified Items Count", f"{len(items)} item(s)")
    c3.metric("Text / Typography Blocks", f"{text_cnt} region(s)")
    c4.metric("Animals / Vehicles", f"{len(animals)} animal(s), {len(vehicles)} vehicle(s)")

    from collections import Counter
    if items:
        st.write(f"• **Isolated Objects:** {', '.join(f'`{it}`' for it in items)}")
    if animals:
        anim_summary = ", ".join(f"{cnt}x `{name}`" if cnt > 1 else f"`{name}`" for name, cnt in Counter(animals).items())
        st.write(f"• **Animals Detected ({len(animals)} total):** {anim_summary}")
    if vehicles:
        veh_summary = ", ".join(f"{cnt}x `{name}`" if cnt > 1 else f"`{name}`" for name, cnt in Counter(vehicles).items())
        st.write(f"• **Vehicles Detected ({len(vehicles)} total):** {veh_summary}")


def render_newbie_narrative_result(newbie_explanation: str, filename: str) -> None:
    """Renders Stage 5: Final Result and Plain-English Newbie Narrative Explanation."""
    st.markdown("---")
    st.markdown("### 📖 Stage 5: Forensic Result & Beginner-Friendly Explanation")
    st.caption("Human-readable narrative describing what was identified in the image and explaining the authenticity verdict in simple everyday terms.")

    if newbie_explanation:
        st.markdown(newbie_explanation)
    else:
        st.info("Generating narrative explanation...")


def _render_video_stream_specs(profile_data: Dict[str, Any]) -> None:
    st.markdown("### 🔬 Stage 1: Pre-Analysis Video Stream Specifications")
    st.caption("Low-level container headers, stream geometry, frame rates, codecs, and cryptographic hashes extracted before running detection.")
    geom = profile_data.get("geometry", {})
    codec = profile_data.get("codec_and_container", {})
    hashes = profile_data.get("cryptographic_hashes", {})

    c1, c2, c3, c4 = st.columns(4)
    w_val = geom.get("width", 0)
    h_val = geom.get("height", 0)
    c1.metric("Dimensions", f"{w_val} × {h_val} px" if w_val else "N/A")
    c2.metric("Frame Rate", f"{geom.get('fps', 0.0):.1f} fps")
    c3.metric("Duration", f"{geom.get('duration_seconds', 0.0):.2f} s")
    c4.metric("Total Frames", f"{geom.get('total_frames', 0):,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Container / Codec", f"{codec.get('container', 'MP4')} • {codec.get('codec', 'AVC')}")
    c6.metric("Aspect Ratio", geom.get("aspect_ratio", "N/A"))
    c7.metric("Bitrate", f"{codec.get('bitrate_kbps', 0.0):.0f} kbps" if codec.get('bitrate_kbps') else "N/A")
    sha_str = hashes.get("sha256", "")
    c8.metric("SHA-256", f"{sha_str[:12]}..." if sha_str else "N/A")

    st.markdown("---")


def _render_video_detection(decision: Dict[str, Any], video_res: Dict[str, Any]) -> None:
    st.markdown("### 🔢 Stage 4: Algorithmic Detection & Temporal Timeline Attribution")
    probs = decision.get("authenticity_probabilities", {})
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("🤖 AI Synthesis Probability", f"{probs.get('p_ai', 0.0):.1f}%")
    col_p2.metric("📷 Authentic Capture Probability", f"{probs.get('p_real', 0.0):.1f}%")
    col_p3.metric("❓ Undetermined / Epistemic Margin", f"{probs.get('p_undecided', 0.0):.1f}%")

    segments = video_res.get("temporal_segments", [])
    if segments:
        st.markdown("##### ⏱️ Video Temporal Timeline Segments")
        for s in segments:
            badge = "🚨 AI GENERATED" if "AI" in s.get("label", "") else ("✅ REAL" if "REAL" in s.get("label", "") else "❓ UNCERTAIN")
            st.write(f"• `{s.get('start_seconds', 0.0)}s ── {s.get('end_seconds', 0.0)}s` ({s.get('duration_seconds', 0.0)}s) : **{badge}**")


def _render_video_inspection(item: Dict[str, Any]) -> None:
    st.markdown("### 🎥 Visual Inspection & Temporal Media Player")
    col_vid, col_kf = st.columns([1.2, 1], gap="medium")
    with col_vid:
        st.video(item["path"])
    with col_kf:
        tmp_kf_path = item.get("tmp_kf_path")
        kf_ai = item.get("kf_ai")
        if tmp_kf_path and Path(tmp_kf_path).is_file():
            if kf_ai and kf_ai.get("heatmap_rgb") is not None:
                st.image(kf_ai["heatmap_rgb"], caption=f"Keyframe Heatmap ({kf_ai.get('ai_spatial_area_pct', 0)}% AI area)", width="stretch")
            else:
                st.image(tmp_kf_path, caption="Sampled Video Keyframe", width="stretch")
        else:
            st.info("Keyframe extraction completed.")


def _render_audio_signal_specs(item: Dict[str, Any], profile_data: Dict[str, Any]) -> None:
    st.markdown("### 🔬 Stage 1: Pre-Analysis Audio Signal Specifications")
    st.caption("Low-level container headers, sampling rates, bit depths, dynamic ranges, and cryptographic hashes extracted before running detection.")
    sr_val = item.get("sr") or profile_data.get("sample_rate", 44100)
    dur_val = item.get("duration") or profile_data.get("duration", 0.0)
    ch_val = profile_data.get("channels", 1)
    fmt_val = profile_data.get("format", "WAV")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampling Rate", f"{sr_val:,} Hz")
    c2.metric("Duration", f"{dur_val:.2f} s")
    c3.metric("Channels", "Stereo (2 Ch)" if ch_val == 2 else "Mono (1 Ch)")
    c4.metric("Format / Container", str(fmt_val).upper())

    hashes = profile_data.get("cryptographic_hashes", {})
    sha_str = hashes.get("sha256", "")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Bit Depth", profile_data.get("bit_depth", "16-bit PCM"))
    c6.metric("RMS Energy", f"{profile_data.get('rms_energy', 0.0):.4f}" if profile_data.get('rms_energy') else "Normal")
    c7.metric("Dynamic Range", f"{profile_data.get('dynamic_range_db', 0.0):.1f} dB" if profile_data.get('dynamic_range_db') else "Standard")
    c8.metric("SHA-256", f"{sha_str[:12]}..." if sha_str else "N/A")

    st.markdown("---")


def _render_audio_detection(decision: Dict[str, Any], audio_res: Dict[str, Any]) -> None:
    st.markdown("### 🔢 Stage 4: Algorithmic Detection & Speech Timeline Verification")
    probs = decision.get("authenticity_probabilities", {})
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("🤖 AI Voice Synthesis Probability", f"{probs.get('p_ai', 0.0):.1f}%")
    col_p2.metric("🎙️ Authentic Voice Probability", f"{probs.get('p_real', 0.0):.1f}%")
    col_p3.metric("❓ Undetermined / Epistemic Margin", f"{probs.get('p_undecided', 0.0):.1f}%")

    audio_segs = audio_res.get("temporal_segments", [])
    if audio_segs:
        st.markdown("##### ⏱️ Speech Timeline Segments")
        for a_seg in audio_segs:
            badge = "🚨 AI VOICE" if "AI" in a_seg.get("label", "") else ("✅ NATURAL SPEECH" if "REAL" in a_seg.get("label", "") else "❓ UNCERTAIN")
            st.write(f"• `{a_seg.get('start_seconds', 0.0)}s ── {a_seg.get('end_seconds', 0.0)}s` ({a_seg.get('duration_seconds', 0.0)}s) : **{badge}**")


def _render_audio_inspection(item: Dict[str, Any]) -> None:
    st.markdown("### 🎙️ Audio Player & Spectral Inspection")
    col_player, col_spec = st.columns([1, 1.2], gap="medium")
    with col_player:
        st.audio(item["path"])
        st.caption("Acoustic analysis inspects for brick-wall vocoder cutoffs (e.g. 7.5kHz/16kHz in ElevenLabs/Suno/CosyVoice), Wiener spectral entropy, and synthetic silence dropouts.")
    with col_spec:
        spec_img = item.get("spec_img")
        if spec_img is not None:
            st.image(
                spec_img,
                caption="Spectral Heatmap (Frequency vs Time) — Exposing Vocoder Cutoff Lines & Harmonic Smoothing",
                width="stretch",
            )
        else:
            st.info("Spectrogram generated for this audio track.")


def _render_image_inspection(item: Dict[str, Any], profile_data: Dict[str, Any], ai_result: Dict[str, Any]) -> None:
    st.markdown("### 🖼️ Visual Inspection & Heatmap Localization")
    col_orig, col_heat = st.columns(2)
    with col_orig:
        w_val = profile_data.get("width", 0)
        h_val = profile_data.get("height", 0)
        st.image(item["path"], caption=f"Original Media ({w_val} × {h_val} px)", width="stretch")
    with col_heat:
        heatmap_rgb = ai_result.get("heatmap_rgb")
        if heatmap_rgb is not None:
            ai_area = ai_result.get("ai_spatial_area_pct", 0.0)
            st.image(heatmap_rgb, caption=f"Spatial Anomaly Map (Estimated AI Area: {ai_area}%)", width="stretch")
        else:
            st.info("Heatmap not generated for this format.")


def _image_nine_dimensions(item: Dict[str, Any], profile_data: Dict[str, Any], ai_result: Dict[str, Any], content_res: Dict[str, Any]) -> Any:
    """The item's precomputed nine-dimension dossier, or one built on demand from its parts."""
    nine_dims = item.get("nine_dimensions_dossier")
    if not nine_dims and profile_data and ai_result:
        nine_dims = build_nine_dimensions_dossier(
            profile_data=profile_data,
            ai_result=ai_result,
            content_inventory=content_res,
            provenance_result=item.get("provenance_res"),
            attribution_result=item.get("attribution_res"),
        )
    return nine_dims


def _image_newbie_text(
    item: Dict[str, Any], profile_data: Dict[str, Any], ai_result: Dict[str, Any],
    content_res: Dict[str, Any], decision: Dict[str, Any], filename: str,
) -> Any:
    """The item's precomputed plain-English narrative, or one generated on demand."""
    text = item.get("newbie_explanation")
    if not text and profile_data and ai_result:
        text = generate_newbie_explanation(
            filename=filename, profile_data=profile_data, content_inventory=content_res,
            ai_result=ai_result, decision=decision,
        )
    return text


def render_linear_image_pipeline_results(item: Dict[str, Any]) -> None:
    """
    Renders the completely linear end-to-end forensic flow requested by the user:
    1. Pre-Analysis Feature & Metadata Extraction (Dimensions, DPI, Pixels, EXIF, Colors, Noise).
    2. 9-Dimensions Forensic Taxonomy Analyzer.
    3. Image Type & Category Identification.
    4. Algorithmic Detection & Quantified Inventory.
    5. Visual Inspection & Spatial Anomaly Heatmap.
    6. Final Result & Beginner-Friendly Newbie Narrative Explanation.
    7. Continuous Learning Feedback.
    """
    if not item or not item.get("success"):
        st.error(f"❌ Failed to process `{item.get('filename', 'media')}`: {item.get('error', 'Unknown error')}")
        return

    profile_data = item.get("img_profile") or item.get("file_profile") or {}
    ai_result = item.get("ai_result") or item.get("ai_detection") or {}
    content_res = item.get("content_res") or item.get("content_inventory") or {}
    decision = item.get("decision") or {}
    filename = item.get("filename", "image")
    dim_report = item.get("dimension_report") or {}

    # 0. Pre-analysis gate banner (hard-block / recognized out-of-scope format): stops the flow
    if render_gate_banner(item.get("gate")) and item.get("gate_blocked"):
        return

    # 1. Stage 1: Pre-Analysis Feature & Metadata Extraction
    render_pre_analysis_specifications(profile_data, expanded=True, source=item.get("source", "User Upload"))

    st.markdown("---")

    # 1b. Stage 1b: File Integrity & Security (advisory; never changes the score)
    render_findings_stage(
        "🛡️ Stage 1b: File Integrity & Security",
        "True format, trailing data, embedded executables/archives/scripts, SVG active content, and instruction-like metadata.",
        collect_findings(dim_report, ["file_integrity"]),
        advisory_note="Security findings are advisory — they never change the authenticity score.",
    )

    st.markdown("---")

    # 2. Stage 2: 9-Dimensions Forensic Taxonomy Analyzer
    nine_dims = _image_nine_dimensions(item, profile_data, ai_result, content_res)
    if nine_dims:
        render_nine_dimensions_breakdown(nine_dims, expanded=False)

    st.markdown("---")

    # 3. Stage 3: Image Type & Category Identification
    render_image_type_and_category(decision, content_res, ai_result, nine_dims=nine_dims)

    st.markdown("---")

    # 3b. Stage 3b: Metadata & Container Forensics
    render_findings_stage(
        "🗂️ Stage 3b: Metadata & Container Forensics",
        "JPEG quantization tables, PNG chunks, EXIF coherence, timestamp plausibility, embedded thumbnail, ICC profile.",
        collect_findings(dim_report, ["formats", "metadata"]),
        show_weight=True,
        advisory_note=(
            "Metadata is forgeable and often stripped by platforms. Missing metadata is never treated as evidence; "
            "contributions are small, capped log-odds."
        ),
    )

    st.markdown("---")

    # 4. Stage 4: Algorithmic Detection & Quantified Inventory
    render_quantified_detections_and_inventory(decision, content_res, ai_result, profile_data)

    # 4b. Stage 4b: Confidence Band & Reliability
    render_confidence_and_reliability(
        item.get("confidence_band"),
        item.get("ood"),
        dim_report.get("reliability"),
        item.get("attribution_open_set"),
    )

    st.markdown("---")

    _render_image_inspection(item, profile_data, ai_result)

    # 6. Stage 5: Forensic Result & Beginner-Friendly Newbie Narrative Explanation
    render_newbie_narrative_result(_image_newbie_text(item, profile_data, ai_result, content_res, decision, filename), filename)

    st.markdown("---")

    # 6b. Stage 6b: Context, Rights & Lifecycle (advisory)
    render_findings_stage(
        "⚖️ Stage 6b: Context, Rights & Lifecycle",
        "Perceptual fingerprint for re-use checks, rights/privacy/biometric flags, AI-disclosure labelling, and platform re-encode likelihood.",
        collect_findings(dim_report, ["context", "legal", "lifecycle"]),
        advisory_note="Advisory only — these flags do not change the authenticity score and are not legal advice.",
    )

    st.markdown("---")

    # 7. Forensic Feedback & Retraining
    render_bottom_feedback_panel(
        media_path=item["path"],
        modality="image",
        forensic_data=ai_result,
        profile_data=profile_data,
        decision=decision,
        unique_key=f"linear_img_{filename}",
    )


def render_linear_video_pipeline_results(item: Dict[str, Any]) -> None:
    """
    Renders the completely linear end-to-end forensic flow for video media:
    1. Pre-Analysis Feature & Stream Specifications (Resolution, FPS, Duration, Codec, Atoms, Hashes).
    2. 9-Dimensions Video Forensic Taxonomy Dossier.
    3. Video Scene, Living Entities & Content Intelligence.
    4. Algorithmic Detection & Quantified Temporal Inventory.
    5. Video Player, Sampled Keyframe & Spatial Heatmap Inspection.
    6. Final Result & Beginner-Friendly Newbie Narrative Explanation.
    7. Continuous Learning Feedback.
    """
    if not item or not item.get("success"):
        st.error(f"❌ Failed to process `{item.get('filename', 'video')}`: {item.get('error', 'Unknown error')}")
        return

    profile_data = item.get("vid_profile") or item.get("file_profile") or {}
    video_res = item.get("video_result") or item.get("temporal_forensics") or {}
    content_res = item.get("content_res") or item.get("content_inventory") or {}
    decision = item.get("decision") or {}
    filename = item.get("filename", "video")
    dim_report = item.get("dimension_report") or {}

    # 0. Pre-analysis gate banner (hard-block / recognized out-of-scope format): stops the flow
    if render_gate_banner(item.get("gate")) and item.get("gate_blocked"):
        return

    _render_video_stream_specs(profile_data)

    # 1b. Stage 1b: File Integrity & Security (advisory; never changes the score)
    render_findings_stage(
        "🛡️ Stage 1b: File Integrity & Security",
        "True container format, bytes beyond the declared end of the container, embedded executables/archives/scripts, and instruction-like metadata text.",
        collect_findings(dim_report, ["file_integrity"]),
        advisory_note="Security findings are advisory — they never change the authenticity score.",
    )

    st.markdown("---")

    # 2. Stage 2: 9-Dimensions Forensic Taxonomy Analyzer
    nine_dims = item.get("nine_dimensions_dossier")
    if nine_dims:
        render_nine_dimensions_breakdown(nine_dims, expanded=False)

    st.markdown("---")

    # 3. Stage 3: Scene, Living Entities & Content Intelligence
    render_scene_and_content_intelligence(content_res, modality="video")

    st.markdown("---")

    # 3b. Stage 3b: Container, Metadata & Signal Forensics
    render_findings_stage(
        "🗂️ Stage 3b: Container, Metadata & Signal Forensics",
        "MP4/MOV box layout, sample-table consistency, creation times and tool strings, hardware telemetry tracks, encoder strings, "
        "interlacing/combing, and frame cadence (duplicates, 3:2 pulldown).",
        collect_findings(dim_report, ["container", "signal"]),
        show_weight=True,
        advisory_note=(
            "Container fields are forgeable and are rewritten by editors and platforms. Interlace and cadence findings describe how the clip "
            "was captured or converted, not whether it is AI-generated. Contributions to the score are small, capped log-odds."
        ),
    )

    st.markdown("---")

    _render_video_detection(decision, video_res)

    # 4b. Stage 4b: Confidence Band & Reliability
    render_confidence_and_reliability(
        item.get("confidence_band"),
        item.get("ood"),
        dim_report.get("reliability"),
        item.get("attribution_open_set"),
    )

    st.markdown("---")

    _render_video_inspection(item)

    st.markdown("---")

    # 6. Stage 6: Forensic Result & Beginner-Friendly Newbie Narrative Explanation
    newbie_text = item.get("newbie_explanation")
    render_newbie_narrative_result(newbie_text, filename)

    st.markdown("---")

    # 6b. Stage 6b: Context, Rights & Lifecycle (advisory)
    render_findings_stage(
        "⚖️ Stage 6b: Context, Rights & Lifecycle",
        "Frame-hash fingerprint for re-use checks, rights/location/biometric/AI-disclosure flags, and re-encoding likelihood.",
        collect_findings(dim_report, ["context", "legal", "lifecycle"]),
        advisory_note="Advisory only — these flags do not change the authenticity score and are not legal advice.",
    )

    st.markdown("---")

    # 7. Stage 7: Forensic Feedback & Retraining
    render_bottom_feedback_panel(
        media_path=item["path"],
        modality="video",
        forensic_data=video_res,
        profile_data=profile_data,
        decision=decision,
        unique_key=f"linear_vid_{filename}",
    )


def render_linear_audio_pipeline_results(item: Dict[str, Any]) -> None:
    """
    Renders the completely linear end-to-end forensic flow for audio media:
    1. Pre-Analysis Feature & Acoustic Specifications (Sample Rate, Duration, Channels, Bit Depth, Hashes).
    2. 9-Dimensions Audio Forensic Taxonomy Dossier.
    3. Acoustic Scene, Environment & Vocal Delivery Intelligence.
    4. Algorithmic Detection & Speech Timeline Verification.
    5. Audio Player & High-Resolution Spectrogram Inspection.
    6. Final Result & Beginner-Friendly Newbie Narrative Explanation.
    7. Continuous Learning Feedback.
    """
    if not item or not item.get("success"):
        st.error(f"❌ Failed to process `{item.get('filename', 'audio')}`: {item.get('error', 'Unknown error')}")
        return

    profile_data = item.get("aud_profile") or item.get("file_profile") or {}
    audio_res = item.get("audio_result") or item.get("acoustic_forensics") or {}
    content_res = item.get("content_res") or item.get("scene_and_tone") or {}
    decision = item.get("decision") or {}
    filename = item.get("filename", "audio")
    dim_report = item.get("dimension_report") or {}

    # 0. Pre-analysis gate banner (hard-block / recognized symbolic music): stops the flow
    if render_gate_banner(item.get("gate")) and item.get("gate_blocked"):
        return

    _render_audio_signal_specs(item, profile_data)

    # 1b. Stage 1b: File Integrity & Security (advisory; never changes the score)
    render_findings_stage(
        "🛡️ Stage 1b: File Integrity & Security",
        "True container format, bytes beyond the declared end, embedded executables/archives/scripts, and instruction-like metadata text.",
        collect_findings(dim_report, ["file_integrity"]),
        advisory_note="Security findings are advisory — they never change the authenticity score.",
    )

    st.markdown("---")

    # 2. Stage 2: 9-Dimensions Forensic Taxonomy Analyzer
    nine_dims = item.get("nine_dimensions_dossier")
    if nine_dims:
        render_nine_dimensions_breakdown(nine_dims, expanded=False)

    st.markdown("---")

    # 3. Stage 3: Acoustic Scene, Environment & Vocal Delivery Intelligence
    render_scene_and_content_intelligence(content_res, modality="audio")

    st.markdown("---")

    # 3b. Stage 3b: Container, Metadata & Signal Forensics
    render_findings_stage(
        "🗂️ Stage 3b: Container, Metadata & Signal Forensics",
        "ID3 / RIFF-bext / FLAC-MD5 / MP3-LAME / Ogg structure, mains-hum (ENF) trace, fake hi-res and bit-depth padding, "
        "telephony band-limiting, and loudness/dynamics.",
        collect_findings(dim_report, ["container", "signal"]),
        show_weight=True,
        advisory_note=(
            "Tags are forgeable and often stripped by platforms. A missing ENF trace is never treated as evidence of synthesis "
            "(battery power, telephony and noise reduction remove mains hum). Contributions are small, capped log-odds."
        ),
    )

    st.markdown("---")

    _render_audio_detection(decision, audio_res)

    # 4b. Stage 4b: Confidence Band & Reliability
    render_confidence_and_reliability(
        item.get("confidence_band"),
        item.get("ood"),
        dim_report.get("reliability"),
        item.get("attribution_open_set"),
    )

    st.markdown("---")

    _render_audio_inspection(item)

    st.markdown("---")

    # 6. Stage 6: Forensic Result & Beginner-Friendly Newbie Narrative Explanation
    newbie_text = item.get("newbie_explanation")
    render_newbie_narrative_result(newbie_text, filename)

    st.markdown("---")

    # 6b. Stage 6b: Context, Rights & Lifecycle (advisory)
    render_findings_stage(
        "⚖️ Stage 6b: Context, Rights & Lifecycle",
        "Acoustic fingerprint for re-use checks, rights/voice-biometric/AI-disclosure flags, and transcoding-cascade likelihood.",
        collect_findings(dim_report, ["context", "legal", "lifecycle"]),
        advisory_note="Advisory only — these flags do not change the authenticity score and are not legal advice.",
    )

    st.markdown("---")

    # 7. Stage 7: Forensic Feedback & Retraining
    render_bottom_feedback_panel(
        media_path=item["path"],
        modality="audio",
        forensic_data=audio_res,
        profile_data=profile_data,
        decision=decision,
        unique_key=f"linear_aud_{filename}",
    )




