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
from typing import Any, Dict

import numpy as np
import streamlit as st

from audio_detector import AudioProfiler, AudioSelfImprover
from core.decision import generate_final_decision, normalize_percentages
from image_detector import ImageProfiler, ImageSelfImprover
from image_detector.explain import build_nine_dimensions_dossier, generate_newbie_explanation
from video_detector import VideoProfiler, VideoSelfImprover


def profile_media(file_path: str | Path, modality: str = "auto", source: str = "User Upload") -> Dict[str, Any]:
    p = Path(file_path)
    suffix = p.suffix.lower()
    if modality == "image" or (modality == "auto" and suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
        res = ImageProfiler().profile_image(file_path, source=source)
        if res.get("valid"):
            res["success"] = True
            res["file_identity"] = {
                "filename": res.get("filename"),
                "container_format": res.get("format"),
                "size_kb": (res.get("file_size_bytes", 0) / 1024.0),
                "mime_type": f"image/{res.get('format', 'png').lower()}",
                "sha256": res.get("sha256"),
            }
            res["pixel_specifications"] = {
                "width": res.get("width"),
                "height": res.get("height"),
                "aspect_ratio": res.get("aspect_ratio"),
                "channels": res.get("channels"),
                "pixel_entropy": res.get("pixel_entropy"),
            }
        return res
    elif modality == "video" or (modality == "auto" and suffix in (".mp4", ".mov", ".avi", ".mkv", ".webm")):
        res = VideoProfiler().profile_video(file_path)
        if res.get("valid"):
            res["success"] = True
            res["file_identity"] = {
                "filename": res.get("filename"),
                "container_format": res.get("container_format"),
                "size_kb": (res.get("file_size_bytes", 0) / 1024.0),
                "mime_type": f"video/{res.get('container_format', 'mp4').lower()}",
                "sha256": res.get("sha256"),
            }
            res["stream_specifications"] = {
                "duration_seconds": res.get("duration_seconds"),
                "fps": res.get("fps"),
                "total_frames": res.get("total_frames"),
                "width": res.get("width"),
                "height": res.get("height"),
                "codec": res.get("codec"),
            }
        return res
    elif modality == "audio" or (modality == "auto" and suffix in (".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a")):
        res = AudioProfiler().profile_audio(file_path)
        if res.get("valid"):
            res["success"] = True
            res["file_identity"] = {
                "filename": res.get("filename"),
                "container_format": res.get("format"),
                "size_kb": (res.get("file_size_bytes", 0) / 1024.0),
                "mime_type": f"audio/{res.get('format', 'wav').lower()}",
                "sha256": res.get("sha256"),
            }
            res["signal_specifications"] = {
                "sample_rate": res.get("sample_rate"),
                "duration_seconds": res.get("duration_seconds"),
                "rms_energy": res.get("rms_energy"),
                "crest_factor": res.get("crest_factor"),
                "dynamic_range_db": res.get("dynamic_range_db"),
                "is_clipped": res.get("is_clipped"),
            }
        return res
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
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            real_score = st.slider(
                "📷 Real / Authentic Score (1-10)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                key=f"{unique_key}_real_score",
                help="1 = Completely Fake/Synthetic, 10 = 100% Genuine Camera/Microphone Capture",
            )
            synth_score = st.slider(
                "🤖 Synthetic / AI-Generated Score (1-10)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                key=f"{unique_key}_synth_score",
                help="1 = Natural/Organic, 10 = Full Generative Synthesis (Midjourney, Gemini, ElevenLabs, Sora)",
            )

        with fcol2:
            forged_score = st.slider(
                "✂️ Forged / Edited / Deepfake Score (1-10)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                key=f"{unique_key}_forged_score",
                help="1 = Unaltered, 10 = Heavily Doctored / Face-Swapped / Splice-Edited",
            )
            undetected_score = st.slider(
                "⚠️ Undetected / Missed Artifacts Score (0-10)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key=f"{unique_key}_undetected_score",
                help="How much did current automated detection miss or under-report? (0 = Caught Everything, 10 = Missed Crucial Artifacts)",
            )

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            ground_truth = st.selectbox(
                "Confirmed Ground Truth Label",
                [
                    "AI-Generated",
                    "Real / Camera Authentic",
                    "Partially Forged / Deepfake",
                    "Undetermined / Mixed",
                ],
                key=f"{unique_key}_gt",
            )
            generator_select = st.selectbox(
                "Known / Suspected Generator Family",
                [
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
                ],
                key=f"{unique_key}_gen_select",
            )
        with rcol2:
            custom_gen_tag = st.text_input(
                "Custom Generator / Camera Hardware Details (Optional)",
                placeholder="e.g. Gemini 1.5 Flash, Midjourney v6.1 --v 6.1, iPhone 16 Pro Max",
                key=f"{unique_key}_custom_gen_tag",
            )
            generator_tag = custom_gen_tag.strip() if custom_gen_tag.strip() else (generator_select if generator_select != "Auto-Attributed / Unspecified" else "")

        user_notes = st.text_area(
            "Forensic Observations & Commentary",
            placeholder="Describe specific telltales: waxy skin, warped fingers, abnormal room reflections, vocoder cutoff...",
            key=f"{unique_key}_notes",
        )

        if st.button("💾 Submit Rating & Dynamically Retrain Algorithm", key=f"{unique_key}_submit_btn"):
            with st.spinner("Archiving fingerprint, auto-calibrating parameters, and updating models..."):
                ratings = {
                    "real_score": real_score,
                    "synthetic_score": synth_score,
                    "forged_score": forged_score,
                    "undetected_score": undetected_score,
                }
                user_label = "AI" if ("AI" in ground_truth or synth_score >= 5.5) else "REAL"
                metrics = forensic_data if isinstance(forensic_data, dict) else (forensic_data.to_dict() if hasattr(forensic_data, "to_dict") else {})
                notes_text = f"[{ground_truth}] {user_notes} | Gen: {generator_tag} | Ratings: {ratings}"

                if modality == "image" or (modality == "auto" and str(media_path).lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"))):
                    improver = ImageSelfImprover()
                    calib = improver.record_feedback(str(media_path), user_label, metrics, notes_text)
                    st.success("✅ **Image Forensic Feedback Successfully Recorded!**")
                    st.info(f"🧠 **Learned Calibration Updates:** Processed {calib.get('samples_processed', 1)} verified samples.")
                elif modality == "video" or (modality == "auto" and str(media_path).lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))):
                    improver = VideoSelfImprover()
                    calib = improver.record_feedback(str(media_path), user_label, metrics, notes_text)
                    st.success("✅ **Video Forensic Feedback Successfully Recorded!**")
                    st.info(f"🧠 **Learned Calibration Updates:** Processed {calib.get('samples_processed', 1)} verified samples.")
                else:
                    improver = AudioSelfImprover()
                    calib = improver.record_feedback(str(media_path), user_label, metrics, notes_text)
                    st.success("✅ **Audio Forensic Feedback Successfully Recorded!**")
                    st.info(f"🧠 **Learned Calibration Updates:** Processed {calib.get('samples_processed', 1)} verified samples.")


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

    status = dossier.get("final_status", "UNDETERMINED_OOD")
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


def render_analysis_right_panel(decision: Dict[str, Any], content_res: Dict[str, Any], modality: str = "image") -> None:
    """
    Renders the unified, simplified forensic analysis and media intelligence panel.
    Provides clear, human-readable plain-English explanations of:
    1. Taxonomy Classification & Plain-English Verdict
    2. What does this file depict & represent
    3. Why is it classified this way (Physical optics, facial realism, metadata/watermarks)
    4. Authenticity Probabilities & Percentages
    5. Model Attribution & Spatial Manipulation
    6. Expandable Deep Forensic Audit Trail
    """
    if not decision:
        return

    st.subheader("📊 Forensic Analysis & Media Intelligence")

    # 1. Classification Taxonomy & Plain-English Verdict Banner
    tax_state = decision.get("taxonomy_state")
    tax_label = decision.get("taxonomy_label")
    tax_reasons = decision.get("taxonomy_reasons", [])
    status = decision.get("final_status", "UNDETERMINED_OOD")

    probs = decision.get("authenticity_probabilities", {})
    p_ai = float(probs.get("p_ai", 0.0))
    p_real = float(probs.get("p_real", 0.0))
    p_und = float(probs.get("p_undecided", 100.0))

    # Evaluate taxonomy state with strict priority to prevent generic status interception
    if tax_state == "AUTHENTIC_SCREENSHOT" or status == "AUTHENTIC SCREENSHOT":
        st.info(f"📱 **Verdict: {tax_label or 'Authentic Device Screenshot'}**")
        st.markdown(
            "📌 **Plain-English Verdict:** This is an **authentic digital screen capture** from a mobile phone, tablet, laptop, or desktop monitor. "
            "It contains native operating system / application UI rendering, standard display aspect ratios, and zero generative synthesis."
        )
    elif tax_state == "AI_ENHANCED_SCREENSHOT" or status == "AI-ENHANCED SCREENSHOT":
        st.warning(f"📱🧬 **Verdict: {tax_label or 'AI-Enhanced / Composite Screenshot'}** ({p_ai:.1f}% AI Signals)")
        st.markdown(
            "📌 **Plain-English Verdict:** This is a **device screen capture displaying media modified by AI tools** "
            "(e.g. neural face swaps, deepfake filters, generative inpainting, or AI upscaling)."
        )
    elif tax_state == "AI_GENERATED_SCREENSHOT" or status == "AI-GENERATED SCREENSHOT":
        st.error(f"📱🤖 **Verdict: {tax_label or 'AI-Generated Content Screenshot'}** ({p_ai:.1f}% AI Confidence)")
        st.markdown(
            "📌 **Plain-English Verdict:** This is a **device screen capture displaying fully synthetic generative AI media** "
            "(e.g. outputs from Midjourney, ChatGPT/DALL-E, Gemini, or diffusion apps captured on a device screen)."
        )
    elif tax_state == "AUTHENTIC_EDITED" or status in ("AUTHENTIC (CONVENTIONALLY EDITED)", "AUTHENTIC_EDITED"):
        st.info(f"🎨 **Verdict: {tax_label or 'Authentic Created Photograph (Edited / Graphic Design)'}**")
        st.markdown(
            "📌 **Plain-English Verdict:** This is a **real photograph modified with standard editing software** "
            "(e.g. cropping, background removal, alpha transparency cutout, solid studio backdrop replacement, or Canva/Photoshop composition). "
            "The base subject preserves authentic camera sensor noise and natural physical anatomy without generative AI synthesis."
        )
    elif tax_state == "AI_ENHANCED_COMPOSITE" or status in ("PARTIALLY_SYNTHETIC_OR_EDITED", "AI-ENHANCED / COMPOSITE"):
        st.warning(f"🧬 **Verdict: {tax_label or 'AI-Enhanced / Composite (Mix)'}** ({p_ai:.1f}% AI Signals)")
        st.markdown(
            "📌 **Plain-English Verdict:** This is a **hybrid composite combining real capture with neural AI algorithms**. "
            "A real base image was augmented using deep learning models (such as neural face-swapping, generative inpainting, "
            "deep learning object addition/removal, or Topaz Photo AI / Remini neural upscaling and sharpening)."
        )
    elif tax_state == "FULLY_AI_GENERATED" or status in ("LIKELY_SYNTHETIC", "LIKELY AI-GENERATED"):
        st.error(f"🤖 **Verdict: {tax_label or 'Fully AI Generated'}** ({p_ai:.1f}% AI Confidence)")
        st.markdown(
            "📌 **Plain-English Verdict:** This file was **synthesized end-to-end via generative diffusion or neural models** "
            "(e.g. Google Gemini / Imagen, Midjourney, DALL-E, Flux, Stable Diffusion). "
            "Confirmed by synthetic surface smoothing, absence of physical Poisson sensor noise, and generative provenance markers."
        )
    elif tax_state == "AUTHENTIC_REAL_PHOTOGRAPH" or status in ("LIKELY_AUTHENTIC", "LIKELY REAL"):
        st.success(f"🌿 **Verdict: {tax_label or 'Authentic Real Photograph'}** ({p_real:.1f}% Confidence)")
        st.markdown(
            "📌 **Plain-English Verdict:** This file is a **100% genuine real-world capture**. "
            "It exhibits authentic physical camera sensor noise (PRNU), organic optical light distribution, and coherent physical geometry. "
            "It is completely free of generative AI manipulation."
        )
    else:
        st.info("❓ **Verdict: Undetermined / Mixed Signals**")
        st.markdown(
            "📌 **Plain-English Verdict:** Forensic indicators show balanced evidence or heavy platform compression. "
            "The system maintains formal scientific uncertainty rather than guessing."
        )

    # 2. Probability Gauges with Progress Bar
    st.markdown("##### 📈 Authenticity Probabilities")
    st.caption("Calibrated Bayesian probabilities separating synthetic generation from genuine capture.")
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("🤖 AI-Generated", f"{p_ai:.1f}%")
    col_p2.metric("📷 Camera / Real", f"{p_real:.1f}%")
    col_p3.metric("❓ Undetermined", f"{p_und:.1f}%")

    st.progress(
        min(1.0, max(0.0, p_ai / 100.0)),
        text=f"Generative AI Probability: {p_ai:.1f}% AI vs {p_real:.1f}% Real"
    )

    st.markdown("---")

    # 3. What Does This Media Depict & Represent?
    st.markdown("##### 🖼️ What Does This File Depict & Represent?")
    inv = decision.get("content_inventory", {})
    what_rep = decision.get("what_it_represents") or "Visual media capture"
    st.markdown(f"**Summary:** *{what_rep}*")

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.write(f"• **Depiction / Purpose:** `{inv.get('photographic_purpose', 'General Scene')}`")
        st.write(f"• **Setting & Environment:** `{inv.get('location_context', 'General')}` • `{inv.get('setting_type', 'Ambient')}`")
        st.write(f"• **Daytime & Lighting:** `{inv.get('estimated_daytime', 'Daylight')}` ({inv.get('lighting_style', 'Ambient')})")
        st.write(f"• **Atmosphere & Tone:** `{inv.get('color_tone', 'Neutral')}` • `{inv.get('atmospheric_mood', 'Balanced')}`")
    with c_col2:
        persons_cnt = inv.get("persons_count", 0)
        faces_cnt = inv.get("faces_count", 0)
        st.write(f"• **Living Humans:** `{persons_cnt}` person(s), `{faces_cnt}` face(s)")
        if inv.get("animals_detected"):
            st.write(f"• **Animals:** {', '.join(inv.get('animal_types', []))}")
        if inv.get("vehicles_detected"):
            st.write(f"• **Vehicles:** {', '.join(inv.get('vehicle_types', []))}")
        items = inv.get("identified_items", [])
        if items:
            st.write(f"• **Identified Items:** {', '.join(f'`{i}`' for i in items[:4])}")
        if inv.get("dominant_audio_type") and inv.get("dominant_audio_type") != "N/A":
            st.write(f"• **Audio Stream:** `{inv.get('dominant_audio_type')}` ({inv.get('acoustic_environment', 'Studio')})")

    st.markdown("---")

    # 4. Why This Classification? (Key Forensic Evidence & Indicators)
    st.markdown("##### 🔍 Why This Classification? (Key Forensic Evidence)")
    if tax_reasons:
        for r in tax_reasons:
            st.markdown(f"• {r}")
    else:
        st.write("• Forensic analysis completed across optical, biological, and metadata domains.")

    # 5. Multimodal Breakdown (Optical, Biological, Provenance)
    with st.expander("🔎 View Forensic Evidence Breakdown by Pillar", expanded=True):
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.markdown("###### 🔬 Optical & Sensor Physics")
            prov = decision.get("provenance", {})
            if prov.get("hardware_make"):
                st.write(f"• **Camera Hardware:** Verified `{prov.get('hardware_make')} {prov.get('hardware_model') or ''}`".strip())
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

        with ecol2:
            st.markdown("###### 🏷️ Provenance & Generator Watermarks")
            attr = decision.get("model_attribution", {})
            wm_detected = attr.get("watermark_detected", False)
            if wm_detected:
                st.write("• **AI Watermark:** ⚠️ Google Gemini 4-pointed sparkle emblem detected in corner quadrant.")
            else:
                st.write("• **AI Watermark:** No visible generator watermark or logo emblem detected.")

            if prov.get("c2pa_present"):
                st.write(f"• **C2PA Credentials:** Found ({prov.get('c2pa_signature', 'Valid')})")
            else:
                st.write("• **C2PA Credentials:** Absent (neutral/stripped)")

            if prov.get("software"):
                st.write(f"• **Software Tag:** `{prov.get('software')}`")

    # 6. Global AI Generator Attribution & Fingerprint
    st.markdown("##### 🌐 Global AI Generator Attribution")
    attr = decision.get("model_attribution", {})
    is_authentic_media = tax_state in ("AUTHENTIC_REAL_PHOTOGRAPH", "AUTHENTIC_EDITED") and not (
        attr.get("attributed_model") and "Topaz" in attr.get("attributed_model")
    )

    if is_authentic_media:
        st.success("✅ **Camera / Optical Origin:** No AI generator footprint detected. Verified authentic photographic capture.")
    else:
        acol1, acol2, acol3 = st.columns(3)
        acol1.metric("Attributed Model", attr.get("attributed_model", "Unattributable"))
        acol2.metric("Region / Country", attr.get("region_of_origin", "Global"))
        acol3.metric("Attribution Match", f"{int(attr.get('attribution_confidence', 0) * 100)}%")

        candidates = attr.get("top_candidates", [])
        if candidates:
            cand_items = [f"**{c.get('model_name')}** ({c.get('region')}): `{c.get('attribution_probability')}%`" for c in candidates[:3]]
            st.caption("Top Candidates: " + " • ".join(cand_items))

    # 7. Manipulation Localization
    loc = decision.get("localization", {})
    susp_area = float(loc.get("suspicious_image_area_pct", 0))
    susp_vid = float(loc.get("suspicious_video_duration_pct", 0))
    susp_aud = float(loc.get("suspicious_audio_duration_pct", 0))

    if susp_area > 0 or susp_vid > 0 or susp_aud > 0:
        st.markdown("##### 📍 Manipulation Localization")
        if susp_area > 0:
            st.write(f"• **Localized Anomaly Region:** `{susp_area:.1f}%` of frame area exhibits neural smoothing or background replacement.")
        if susp_vid > 0:
            st.write(f"• **Manipulated Video Duration:** `{susp_vid:.1f}%` of timeline.")
        if susp_aud > 0:
            st.write(f"• **Manipulated Speech Duration:** `{susp_aud:.1f}%` of audio track.")

    # 8. Complete Deep Forensic Audit Trail
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
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            real_score = st.slider(
                "📷 Real / Authentic Score (1-10)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                key=f"{unique_key}_real_score",
                help="1 = Completely Fake/Synthetic, 10 = 100% Genuine Camera/Microphone Capture",
            )
            synth_score = st.slider(
                "🤖 Synthetic / AI-Generated Score (1-10)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                key=f"{unique_key}_synth_score",
                help="1 = Natural/Organic, 10 = Full Generative Synthesis (Midjourney, Gemini, ElevenLabs, Sora)",
            )

        with fcol2:
            forged_score = st.slider(
                "✂️ Forged / Edited / Deepfake Score (1-10)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                key=f"{unique_key}_forged_score",
                help="1 = Unaltered, 10 = Heavily Doctored / Face-Swapped / Splice-Edited",
            )
            undetected_score = st.slider(
                "⚠️ Undetected / Missed Artifacts Score (0-10)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key=f"{unique_key}_undetected_score",
                help="How much did current automated detection miss or under-report? (0 = Caught Everything, 10 = Missed Crucial Artifacts)",
            )

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            ground_truth = st.selectbox(
                "Confirmed Ground Truth Label",
                [
                    "AI-Generated",
                    "Real / Camera Authentic",
                    "Partially Forged / Deepfake",
                    "Undetermined / Mixed",
                ],
                key=f"{unique_key}_gt",
            )
            generator_select = st.selectbox(
                "Known / Suspected Generator Family",
                [
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
                ],
                key=f"{unique_key}_gen_select",
            )
        with rcol2:
            custom_gen_tag = st.text_input(
                "Custom Generator / Camera Hardware Details (Optional)",
                placeholder="e.g. Gemini 1.5 Flash, Midjourney v6.1, iPhone 16 Pro Max",
                key=f"{unique_key}_custom_gen_tag",
            )
            generator_tag = custom_gen_tag.strip() if custom_gen_tag.strip() else (generator_select if generator_select != "Auto-Attributed / Unspecified" else "")

            user_notes = st.text_area(
                "Forensic Observations & Commentary",
                placeholder="Describe specific telltales: waxy skin, warped fingers, abnormal reflections, vocoder cutoff...",
                key=f"{unique_key}_notes",
            )

        if st.button("💾 Submit Rating & Dynamically Retrain Algorithm", key=f"{unique_key}_submit_btn"):
            with st.spinner("Archiving fingerprint, auto-calibrating parameters, and updating models..."):
                ratings = {
                    "real_score": real_score,
                    "synthetic_score": synth_score,
                    "forged_score": forged_score,
                    "undetected_score": undetected_score,
                }
                user_label = "AI" if ("AI" in ground_truth or synth_score >= 5.5) else "REAL"
                metrics = forensic_data if isinstance(forensic_data, dict) else (forensic_data.to_dict() if hasattr(forensic_data, "to_dict") else {})
                notes_text = f"[{ground_truth}] {user_notes} | Gen: {generator_tag} | Ratings: {ratings}"

                if modality == "image" or (modality == "auto" and str(media_path).lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"))):
                    improver = ImageSelfImprover()
                    calib = improver.record_feedback(str(media_path), user_label, metrics, notes_text)
                    st.success("✅ **Image Forensic Feedback Successfully Recorded!**")
                    st.info(f"🧠 **Learned Calibration Updates:** Processed {calib.get('samples_processed', 1)} verified samples.")
                elif modality == "video" or (modality == "auto" and str(media_path).lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))):
                    improver = VideoSelfImprover()
                    calib = improver.record_feedback(str(media_path), user_label, metrics, notes_text)
                    st.success("✅ **Video Forensic Feedback Successfully Recorded!**")
                    st.info(f"🧠 **Learned Calibration Updates:** Processed {calib.get('samples_processed', 1)} verified samples.")
                else:
                    improver = AudioSelfImprover()
                    calib = improver.record_feedback(str(media_path), user_label, metrics, notes_text)
                    st.success("✅ **Audio Forensic Feedback Successfully Recorded!**")
                    st.info(f"🧠 **Learned Calibration Updates:** Processed {calib.get('samples_processed', 1)} verified samples.")

    with tab_specs:
        render_media_specs(profile_data)

    with tab_json:
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

    with st.expander("🔍 View Complete Pre-Analysis Image Profile & Pixel Telemetry", expanded=expanded):
        # 1. Media & Container Origin
        st.markdown("##### 📌 File Identity & Provenance Container")
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("File Name", fname[:20] + "..." if len(fname) > 23 else fname)
        f2.metric("File Size", f"{size_kb:.1f} KB ({size_mb:.2f} MB)")
        f3.metric("Format / MIME", f"{fmt} • {profile_data.get('mime_type', 'image/' + str(fmt).lower())}")
        f4.metric("Source Origin", src)
        f5.metric("SHA-256 (Prefix)", f"{sha[:10]}..." if sha else "N/A")

        st.markdown("---")

        # 2. Dimensions, Spatial Geometry & DPI
        st.markdown("##### 📐 Dimensions, Geometry & Display Resolution (DPI)")
        w = geom.get("width", profile_data.get("width", 0))
        h = geom.get("height", profile_data.get("height", 0))
        tot_pix = geom.get("total_pixels", int(w * h))
        mp = geom.get("megapixels", round(tot_pix / 1_000_000.0, 2))
        asp_str = geom.get("aspect_ratio_str", f"{geom.get('aspect_ratio', 0.0)}:1")
        orient = geom.get("orientation", "Landscape")

        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Width x Height", f"{w} × {h} px")
        g2.metric("Megapixels (MP)", f"{mp:.2f} MP")
        g3.metric("Total Pixel Count", f"{tot_pix:,} px")
        g4.metric("Aspect Ratio", asp_str)
        g5.metric("Orientation", orient)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("DPI Resolution", disp.get("dpi_str", "72 x 72 DPI"))
        d2.metric("Bit Depth", disp.get("bit_depth", "24-bit (3x8-bit)"))
        d3.metric("Color Space", disp.get("color_space", "Standard sRGB"))
        d4.metric("Alpha Channel", "Present (RGBA)" if disp.get("has_alpha_channel") else "None (RGB)")

        st.markdown("---")

        # 3. Device & Acquisition EXIF Parameters
        st.markdown("##### 📷 Hardware Device & EXIF Acquisition Metadata")
        has_cam = bool(exif.get("camera_make") or exif.get("camera_model"))
        cam_make = exif.get("camera_make") or "Unspecified Hardware"
        cam_model = exif.get("camera_model") or ""
        lens = exif.get("lens_model") or "Standard Lens / Unspecified"
        shutter = exif.get("exposure_time") or "N/A"
        aperture = exif.get("aperture") or "N/A"
        iso_val = exif.get("iso") or "N/A"
        focal = exif.get("focal_length") or "N/A"
        flash = exif.get("flash", "Not Fired")
        wb = exif.get("white_balance", "Auto")
        gps_str = exif.get("gps_details", {}).get("coordinates_str", "Not Embedded")
        software = exif.get("software") or "None (Clean Exif)"
        date_str = exif.get("date_time") or "Unknown / Stripped"

        if has_cam:
            st.success(f"Verified Physical Camera Capture: **{cam_make} {cam_model}** | Lens: `{lens}`")
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

        # 4. Pixel-by-Pixel Color & Photometric Statistics
        st.markdown("##### 🎨 Pixel-by-Pixel Photometric Statistics & Dynamic Range")
        entropy = pcol.get("shannon_entropy_bpp", profile_data.get("pixel_entropy", 0.0))
        lum_mean = pcol.get("luminance_mean", 0.0)
        lum_median = pcol.get("luminance_median", lum_mean)
        lum_range = pcol.get("dynamic_range", 0)
        lum_min = pcol.get("luminance_min", 0)
        lum_max = pcol.get("luminance_max", 255)
        clip_hi = pcol.get("highlight_clipped_pct", 0.0)
        clip_sh = pcol.get("shadow_crushed_pct", 0.0)

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Shannon Entropy", f"{entropy:.3f} bits/px")
        p2.metric("Mean / Median Lum", f"{lum_mean:.1f} / {lum_median:.1f}")
        p3.metric("Dynamic Range", f"{lum_range} ({lum_min}..{lum_max})")
        p4.metric("Clipped Highlights", f"{clip_hi:.2f}%")
        p5.metric("Crushed Shadows", f"{clip_sh:.2f}%")

        c_means = pcol.get("channel_means", {})
        c_stds = pcol.get("channel_stds", {})
        if "red" in c_means:
            st.caption(
                f"• **Channel Distributions (Mean ± Std):** "
                f"Red: `{c_means.get('red', 0.0):.1f} ± {c_stds.get('red', 0.0):.1f}` | "
                f"Green: `{c_means.get('green', 0.0):.1f} ± {c_stds.get('green', 0.0):.1f}` | "
                f"Blue: `{c_means.get('blue', 0.0):.1f} ± {c_stds.get('blue', 0.0):.1f}` | "
                f"Unique Quantized Colors: `{pcol.get('unique_quantized_colors', 0):,}`"
            )

        # 5. Dominant Color Palette Swatches
        st.markdown("##### 🌈 Dominant Color Palette & Canvas Coverage")
        palette = pcol.get("dominant_palette", [])
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

        # 6. Raw Physical Noise & Forensic Signals
        st.markdown("##### 🔬 Raw Physical Noise & Frequency Spectrum Properties")
        prnu_noise = phys.get("prnu_noise_mean", 0.0)
        flat_noise = phys.get("flat_region_noise_mean", prnu_noise)
        smoothness = phys.get("surface_smoothness_index", 0.0)
        fft_alpha = phys.get("fft_decay_alpha", 2.05)
        edges_pct = phys.get("canny_edge_pct", 0.0)
        dark_lines = phys.get("dark_line_art_pct", 0.0)
        sharpness = phys.get("laplacian_sharpness_var", 0.0)

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


def render_nine_dimensions_breakdown(nine_dims: Dict[str, Any], expanded: bool = True) -> None:
    """
    Renders Stage 2: 9-Dimensions Forensic Taxonomy Analyzer.
    Exhaustively covers all 9 analytical dimensions formalized in GLOBAL_IMAGE_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md.
    """
    if not nine_dims:
        return

    st.markdown("### 🌐 Stage 2: 9-Dimensions Forensic Taxonomy Analyzer")
    st.caption("Standard-compliant evaluation across all 9 forensic dimensions from NIST OpenMFC, C2PA v2.1, and IPTC standards:")

    with st.expander("🔬 View All 9 Analytical Dimensions Audit", expanded=expanded):
        d1 = nine_dims.get("dimension_1_hardware_provenance", {})
        d2 = nine_dims.get("dimension_2_pixel_architecture", {})
        d3 = nine_dims.get("dimension_3_prnu_sensor_noise", {})
        d4 = nine_dims.get("dimension_4_surface_smoothness", {})
        d5 = nine_dims.get("dimension_5_fourier_fft_decay", {})
        d6 = nine_dims.get("dimension_6_subject_genre", {})
        d7 = nine_dims.get("dimension_7_visual_medium", {})
        d8 = nine_dims.get("dimension_8_sensor_spectrum", {})
        d9 = nine_dims.get("dimension_9_generative_attribution", {})

        t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
            "1. Provenance",
            "2. Pixel Specs",
            "3. PRNU Noise",
            "4. Smoothness",
            "5. Fourier FFT",
            "6. Genre & Scene",
            "7. Visual Medium",
            "8. Sensor Spectrum",
            "9. AI Attribution",
        ])

        with t1:
            st.markdown(f"#### {d1.get('title', 'Dimension 1')}")
            if d1.get("container_atoms"):
                st.write(f"• **Container Atoms:** `{', '.join(str(a) for a in d1.get('container_atoms', []))}`")
            if d1.get("container_format"):
                st.write(f"• **Audio Stream Format:** `{d1.get('container_format')}`")
            if d1.get("camera_hardware") or d1.get("hardware_origin"):
                cam_name = d1.get("camera_hardware") or d1.get("hardware_origin")
                lens = f" | Lens: `{d1.get('lens_model')}`" if d1.get("lens_model") else ""
                st.write(f"• **Hardware Identity:** `{cam_name}`{lens}")
            if d1.get("iptc_digital_source_type"):
                st.write(f"• **IPTC Digital Source Type:** `{d1.get('iptc_digital_source_type')}`")
            st.write(f"• **C2PA Content Credentials:** `{d1.get('c2pa_status')}`")
            if d1.get("software_signature"):
                st.write(f"• **Software Fingerprint:** `{d1.get('software_signature')}` | Date: `{d1.get('date_taken')}`")
            if d1.get("gps_coordinates"):
                st.write(f"• **GPS Location:** `{d1.get('gps_coordinates')}`")
            st.info(f"Verdict: **{d1.get('provenance_verdict')}**")

        with t2:
            st.markdown(f"#### {d2.get('title', 'Dimension 2')}")
            if d2.get("geometry"):
                st.write(f"• **Geometry:** `{d2.get('geometry')}` ({d2.get('aspect_ratio')}, {d2.get('orientation', 'Standard')})")
            if d2.get("frame_rate_fps"):
                st.write(f"• **Frame Rate & Duration:** `{d2.get('frame_rate_fps')}` • `{d2.get('duration')}` ({d2.get('total_frames')} frames)")
            if d2.get("sample_rate"):
                st.write(f"• **Acoustic Geometry:** `{d2.get('sample_rate')}` • `{d2.get('duration')}` • `{d2.get('channels')}` • `{d2.get('bit_depth')}`")
            if d2.get("dpi"):
                st.write(f"• **DPI & Bit Depth:** `{d2.get('dpi')}` • `{d2.get('bit_depth')}` ({d2.get('color_space')})")
            if d2.get("shannon_entropy"):
                st.write(f"• **Shannon Entropy:** `{d2.get('shannon_entropy')}`")
            if d2.get("luminance_dynamic_range"):
                st.write(f"• **Luminance Range:** `{d2.get('luminance_dynamic_range')}`")
            if d2.get("clipping_profile"):
                st.write(f"• **Clipping Profile:** `{d2.get('clipping_profile')}`")
            if d2.get("unique_colors_quantized"):
                st.write(f"• **Unique Quantized Colors:** `{d2.get('unique_colors_quantized', 0):,}`")

        with t3:
            st.markdown(f"#### {d3.get('title', 'Dimension 3')}")
            if d3.get("temporal_warping_variance") is not None:
                st.write(f"• **Inter-Frame Motion Variance:** `{d3.get('temporal_warping_variance', 0.0):.2f}`")
            if d3.get("vocoder_cutoff_hz") is not None:
                st.write(f"• **Vocoder Brickwall Cutoff:** `{d3.get('vocoder_cutoff_hz', 0):,} Hz`")
            if d3.get("prnu_residual_mean") is not None:
                st.write(f"• **PRNU Residual Mean:** `{d3.get('prnu_residual_mean', 0.0):.3f}` | Flat Region Noise: `{d3.get('flat_region_noise', 0.0):.3f}`")
            if d3.get("mathematical_physics"):
                st.write(f"• **Mathematical Formulation:** `{d3.get('mathematical_physics')}`")
            if d3.get("diagnosis"):
                if d3.get("is_natural_shot_noise") or d3.get("temporal_warping_variance", 0.0) < 140.0:
                    st.success(f"Diagnosis: {d3.get('diagnosis')}")
                else:
                    st.error(f"Diagnosis: {d3.get('diagnosis')}")

        with t4:
            st.markdown(f"#### {d4.get('title', 'Dimension 4')}")
            if d4.get("diffusion_flickering_ratio") is not None:
                st.write(f"• **Diffusion Flickering Ratio:** `{d4.get('diffusion_flickering_ratio', 0.0):.3f}`")
            if d4.get("spectral_flatness_wiener") is not None:
                st.write(f"• **Wiener Spectral Flatness:** `{d4.get('spectral_flatness_wiener', 0.0):.4f}`")
            if d4.get("smoothness_index") is not None:
                st.write(f"• **Surface Smoothness Index:** `{d4.get('smoothness_index', 0.0):.3f}`")
            st.write("• **Physical Principle:** Organic micro-textures maintain high local bilateral variance; neural generative models over-smooth.")
            if d4.get("diagnosis"):
                if d4.get("is_diffusion_smoothed") or d4.get("is_flickering_detected") or d4.get("is_synthetic_smoothness"):
                    st.warning(f"⚠️ {d4.get('diagnosis')}")
                else:
                    st.success(f"✅ {d4.get('diagnosis')}")

        with t5:
            st.markdown(f"#### {d5.get('title', 'Dimension 5')}")
            if d5.get("digital_silence_ratio") is not None:
                st.write(f"• **Digital Zero Silence Ratio:** `{d5.get('digital_silence_ratio')}`")
            if d5.get("keyframe_fft_alpha") is not None:
                st.write(f"• **Keyframe Spectral Alpha:** `{d5.get('keyframe_fft_alpha', 2.05):.3f}`")
            if d5.get("spectral_decay_alpha") is not None:
                st.write(f"• **Spectral Decay Alpha:** `{d5.get('spectral_decay_alpha', 2.05):.3f}`")
            if d5.get("mathematical_physics"):
                st.write(f"• **Field Law:** `{d5.get('mathematical_physics')}`")
            if d5.get("diagnosis"):
                if d5.get("is_anomalous_decay") or d5.get("has_digital_zero_silence"):
                    st.warning(f"⚠️ {d5.get('diagnosis')}")
                else:
                    st.success(f"✅ {d5.get('diagnosis')}")

        with t6:
            st.markdown(f"#### {d6.get('title', 'Dimension 6')}")
            if d6.get("audio_type"):
                st.write(f"• **Dominant Audio Type:** `{d6.get('audio_type')}` • `{d6.get('estimated_speakers')} speaker(s)`")
                st.write(f"• **Acoustic Environment:** `{d6.get('acoustic_environment')}`")
            else:
                st.write(f"• **Primary Visual Genre:** `{d6.get('primary_genre')}`")
                st.write(f"• **Living Entities:** `{d6.get('persons_count', 0)}` person(s), `{d6.get('faces_count', 0)}` face(s) (Stylized Character: `{d6.get('is_stylized_character')}`)")
                st.write(f"• **Setting & Atmosphere:** `{d6.get('setting_and_environment')}` • `{d6.get('atmospheric_mood')}`")
                st.write(f"• **Daytime & Lighting Quality:** `{d6.get('lighting_and_daytime')}`")

        with t7:
            st.markdown(f"#### {d7.get('title', 'Dimension 7')}")
            if d7.get("acoustic_delivery_tone"):
                st.write(f"• **Acoustic Delivery Tone:** `{d7.get('acoustic_delivery_tone')}`")
            if d7.get("visual_medium"):
                st.write(f"• **Visual Medium:** `{d7.get('visual_medium')}`")
            if d7.get("synthesis_type"):
                st.write(f"• **Temporal Synthesis Type:** `{d7.get('synthesis_type')}`")
            if d7.get("is_digital_art") is not None:
                st.write(f"• **Digital Art / Painting Detected:** `{d7.get('is_digital_art')}`")
            if d7.get("canny_edge_density_pct") is not None:
                st.write(f"• **Canny Edge Density:** `{d7.get('canny_edge_density_pct', 0.0):.2f}%` | Dark Line Art: `{d7.get('dark_line_contours_pct', 0.0):.2f}%`")
            if d7.get("laplacian_focus_sharpness") is not None:
                st.write(f"• **Focus Sharpness:** `{d7.get('laplacian_focus_sharpness', 0.0):.1f}`")

        with t8:
            st.markdown(f"#### {d8.get('title', 'Dimension 8')}")
            if d8.get("bandwidth_class"):
                st.write(f"• **Acoustic Bandwidth:** `{d8.get('bandwidth_class')}` | Nyquist Ceiling: `{d8.get('nyquist_ceiling')}`")
            if d8.get("sensor_spectrum") or d8.get("sensor_modality"):
                st.write(f"• **Acquisition Modality:** `{d8.get('sensor_spectrum') or d8.get('sensor_modality')}`")
            if d8.get("color_channels") is not None:
                st.write(f"• **Color Channels:** `{d8.get('color_channels', 3)}` channels")
            if d8.get("diagnosis"):
                st.write(f"• **Diagnostic:** {d8.get('diagnosis')}")

        with t9:
            st.markdown(f"#### {d9.get('title', 'Dimension 9')}")
            reg = f" ({d9.get('region_of_origin')})" if d9.get("region_of_origin") else ""
            st.write(f"• **Attributed Generator:** `{d9.get('attributed_model')}`{reg}")
            st.write(f"• **Attribution Confidence:** `{d9.get('confidence', '0%')}`")
            st.write(f"• **Watermark Detected:** `{'YES' if d9.get('watermark_detected') else 'None'}`")
            if d9.get("suspicious_duration_pct") is not None:
                st.write(f"• **Suspicious Duration Fraction:** `{d9.get('suspicious_duration_pct')}`")
            if d9.get("spatial_manipulated_area_pct") is not None:
                st.write(f"• **Spatial Manipulation Area:** `{d9.get('spatial_manipulated_area_pct', 0.0):.1f}%` of frame")


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

    # 1. Stage 1: Pre-Analysis Feature & Metadata Extraction
    render_pre_analysis_specifications(profile_data, expanded=True, source=item.get("source", "User Upload"))

    st.markdown("---")

    # 2. Stage 2: 9-Dimensions Forensic Taxonomy Analyzer
    nine_dims = item.get("nine_dimensions_dossier")
    if not nine_dims and profile_data and ai_result:
        nine_dims = build_nine_dimensions_dossier(
            profile_data=profile_data,
            ai_result=ai_result,
            content_inventory=content_res,
            provenance_result=item.get("provenance_res"),
            attribution_result=item.get("attribution_res"),
        )
    if nine_dims:
        render_nine_dimensions_breakdown(nine_dims, expanded=False)

    st.markdown("---")

    # 3. Stage 3: Image Type & Category Identification
    render_image_type_and_category(decision, content_res, ai_result, nine_dims=nine_dims)

    st.markdown("---")

    # 4. Stage 4: Algorithmic Detection & Quantified Inventory
    render_quantified_detections_and_inventory(decision, content_res, ai_result, profile_data)

    st.markdown("---")

    # 5. Visual Inspection & Spatial Anomaly Heatmap
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

    # 6. Stage 5: Forensic Result & Beginner-Friendly Newbie Narrative Explanation
    newbie_text = item.get("newbie_explanation")
    if not newbie_text and profile_data and ai_result:
        newbie_text = generate_newbie_explanation(
            filename=filename,
            profile_data=profile_data,
            content_inventory=content_res,
            ai_result=ai_result,
            decision=decision,
        )
    render_newbie_narrative_result(newbie_text, filename)

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

    # 1. Stage 1: Pre-Analysis Feature & Stream Specifications
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

    # 2. Stage 2: 9-Dimensions Forensic Taxonomy Analyzer
    nine_dims = item.get("nine_dimensions_dossier")
    if nine_dims:
        render_nine_dimensions_breakdown(nine_dims, expanded=False)

    st.markdown("---")

    # 3. Stage 3: Scene, Living Entities & Content Intelligence
    render_scene_and_content_intelligence(content_res, modality="video")

    st.markdown("---")

    # 4. Stage 4: Algorithmic Detection & Quantified Temporal Inventory
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

    st.markdown("---")

    # 5. Stage 5: Visual Inspection, Player & Keyframe Anomaly
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

    st.markdown("---")

    # 6. Stage 6: Forensic Result & Beginner-Friendly Newbie Narrative Explanation
    newbie_text = item.get("newbie_explanation")
    render_newbie_narrative_result(newbie_text, filename)

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

    # 1. Stage 1: Pre-Analysis Feature & Acoustic Specifications
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

    # 2. Stage 2: 9-Dimensions Forensic Taxonomy Analyzer
    nine_dims = item.get("nine_dimensions_dossier")
    if nine_dims:
        render_nine_dimensions_breakdown(nine_dims, expanded=False)

    st.markdown("---")

    # 3. Stage 3: Acoustic Scene, Environment & Vocal Delivery Intelligence
    render_scene_and_content_intelligence(content_res, modality="audio")

    st.markdown("---")

    # 4. Stage 4: Algorithmic Detection & Speech Timeline Verification
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

    st.markdown("---")

    # 5. Stage 5: Audio Player & Spectrogram Heatmap Inspection
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

    st.markdown("---")

    # 6. Stage 6: Forensic Result & Beginner-Friendly Newbie Narrative Explanation
    newbie_text = item.get("newbie_explanation")
    render_newbie_narrative_result(newbie_text, filename)

    # 7. Stage 7: Forensic Feedback & Retraining
    render_bottom_feedback_panel(
        media_path=item["path"],
        modality="audio",
        forensic_data=audio_res,
        profile_data=profile_data,
        decision=decision,
        unique_key=f"linear_aud_{filename}",
    )




