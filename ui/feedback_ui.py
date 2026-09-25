"""
Feedback and Continuous Learning UI Components for Streamlit.
Provides:
1. Low-level media & pixel forensic specifications card.
2. User multi-criteria 1-10 rating widget for online continual learning.
3. Interactive Continuous Learning & Forensic Memory Bank Dashboard.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from learning.auto_learner import AutoLearner
from learning.forensic_memory import ForensicMemory
from utils.media_profiler import profile_media


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
                memory = ForensicMemory()
                ratings = {
                    "real_score": real_score,
                    "synthetic_score": synth_score,
                    "forged_score": forged_score,
                    "undetected_score": undetected_score,
                }
                reg_res = memory.register_feedback(
                    media_path=media_path,
                    modality=modality,
                    ratings=ratings,
                    ground_truth=ground_truth,
                    forensic_data=forensic_data,
                    user_notes=user_notes,
                    generator_tag=generator_tag,
                )

            if reg_res.get("success"):
                st.success("✅ **Forensic Feedback Successfully Remembered!**")
                update_info = reg_res.get("auto_update", {})
                modifications = update_info.get("modifications", [])
                if modifications:
                    st.info("🧠 **Dynamic Calibration Updates:**\n" + "\n".join(f"• {m}" for m in modifications))

                finetune = update_info.get("online_finetune", {})
                if finetune.get("status") == "success":
                    st.success(f"⚡ **Online Neural Checkpoint Updated!** ({finetune.get('samples_trained')} samples trained)")
                elif finetune.get("status") == "skipped":
                    st.caption(f"ℹ️ Online fine-tuning: {finetune.get('reason')}")
            else:
                st.error("Failed to register feedback.")


def render_learning_dashboard() -> None:
    """Renders the Forensic Memory Bank & Continuous Learning dashboard."""
    st.subheader("🧠 Forensic Memory Bank & Dynamic Learning Dashboard")
    st.write(
        "Real-time overview of learned media fingerprints, dynamically-calibrated feature weights, "
        "and user rating distributions across all inspected images, videos, and audio clips."
    )

    memory = ForensicMemory()
    stats = memory.get_memory_stats()
    calib = memory.load_calibration()
    all_records = memory.load_memory()

    # Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Remembered Media", stats.get("total_samples", 0))
    by_mod = stats.get("by_modality", {})
    m2.metric("Images / Videos / Audio", f"{by_mod.get('image', 0)} / {by_mod.get('video', 0)} / {by_mod.get('audio', 0)}")
    avg_r = stats.get("average_ratings", {})
    m3.metric("Avg Synthetic Rating", f"{avg_r.get('synthetic', 0.0):.1f} / 10")
    m4.metric("Avg Real Rating", f"{avg_r.get('real', 0.0):.1f} / 10")

    st.markdown("---")

    # Dynamic Weights and Calibration Offsets
    st.subheader("⚙️ Dynamically-Learned Calibration Weights")
    st.caption("These weights automatically adjust whenever user feedback flags missed artifacts or false positives.")

    weights = calib.get("feature_weights", {})
    offsets = calib.get("sensitivity_offsets", {})

    wcol1, wcol2, wcol3, wcol4 = st.columns(4)
    wcol1.metric("Noise Residual Weight", f"{weights.get('noise_residual', 0.35):.2f}")
    wcol2.metric("Surface Smoothness Weight", f"{weights.get('surface_smoothness', 0.30):.2f}")
    wcol3.metric("2D FFT Power Decay Weight", f"{weights.get('fft_decay', 0.20):.2f}")
    wcol4.metric("Vocoder Cutoff Weight", f"{weights.get('vocoder_cutoff', 0.40):.2f}")

    ocol1, ocol2, ocol3 = st.columns(3)
    ocol1.metric("Noise Center Offset", f"{offsets.get('noise_center_offset', 0.0):+.2f}")
    ocol2.metric("Smooth Center Offset", f"{offsets.get('smooth_center_offset', 0.0):+.2f}")
    ocol3.metric("Audio AI Bias Offset", f"{offsets.get('audio_ai_offset', 0.0):+.2f}")

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button("⚡ Run Continual Retraining Now", key="btn_run_retrain"):
            with st.spinner("Retraining neural weights on remembered feedback samples..."):
                learner = AutoLearner()
                res = learner._attempt_online_finetune(all_records)
            if res.get("status") == "success":
                st.success(f"✅ Checkpoint fine-tuned on {res.get('samples_trained')} samples! Loss: {res.get('losses')}")
            else:
                st.info(f"Retraining note: {res.get('reason') or res.get('error')}")

    st.markdown("---")

    # Remembered Media Records List
    st.subheader(f"📚 Remembered Media Library ({len(all_records)} Entries)")
    if not all_records:
        st.info("No media feedback registered yet. Analyze any image, video, or audio and submit ratings above to populate memory.")
        return

    for rec in reversed(all_records[-15:]):
        r_id = rec.get("id", "")[:8]
        mod = rec.get("modality", "").upper()
        gt = rec.get("ground_truth", "Unknown")
        ratings = rec.get("ratings", {})
        gen_tag = rec.get("generator_tag") or "Untagged"
        fname = rec.get("file_identity", {}).get("filename", "Unknown File")
        ts = rec.get("timestamp", "")[:19].replace("T", " ")

        with st.expander(f"[{mod}] {fname} — Label: {gt} (Rated {ratings.get('synthetic_score', 0)}/10 Synthetic, {ratings.get('real_score', 0)}/10 Real) — {ts}"):
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Real Score", f"{ratings.get('real_score', 0)}/10")
            rc2.metric("Synthetic Score", f"{ratings.get('synthetic_score', 0)}/10")
            rc3.metric("Forged Score", f"{ratings.get('forged_score', 0)}/10")
            rc4.metric("Undetected Score", f"{ratings.get('undetected_score', 0)}/10")

            st.write(f"• **Generator/Device Tag:** `{gen_tag}`")
            if rec.get("user_notes"):
                st.write(f"• **Notes:** {rec.get('user_notes')}")

            specs = rec.get("specifications", {})
            if specs:
                st.json({
                    "sha256": rec.get("file_identity", {}).get("sha256"),
                    "specifications": specs,
                    "forensic_metrics": rec.get("forensic_metrics", {}),
                })


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
    Renders the unified forensic analysis and intelligence panel for the 2nd top column.
    Provides multi-audience readability (beginners, non-tech, professionals, forensic experts).
    """
    if not decision:
        return

    st.subheader("📊 Forensic Analysis & Media Intelligence")

    # 1. Executive Verdict Card (Plain English + Color-Coded)
    status = decision.get("final_status", "UNDETERMINED_OOD")
    probs = decision.get("authenticity_probabilities", {})
    p_ai = float(probs.get("p_ai", 0.0))
    p_real = float(probs.get("p_real", 0.0))
    p_und = float(probs.get("p_undecided", 100.0))

    if status == "LIKELY_SYNTHETIC":
        st.error(f"🚨 **Verdict: AI-Generated / Synthetic Media ({p_ai:.1f}% Confidence)**")
        st.write("📌 *Plain-English Summary:* This media displays distinctive mathematical telltales of AI generative diffusion models (e.g. absent physical camera noise, artificial surface smoothing, or latent frequency anomalies).")
    elif status == "LIKELY_AUTHENTIC":
        st.success(f"✅ **Verdict: Authentic Camera / Sensor Capture ({p_real:.1f}% Confidence)**")
        st.write("📌 *Plain-English Summary:* This file exhibits natural physical camera sensor noise (PRNU), organic optical light distribution, and expected hardware compression traces.")
    elif status == "PARTIALLY_SYNTHETIC_OR_EDITED":
        st.warning(f"⚠️ **Verdict: Partially Synthetic / Manipulated ({p_ai:.1f}% AI Signals)**")
        st.write("📌 *Plain-English Summary:* While portions of the media appear authentic, localized regions exhibit digital tampering, face-swapping, or generative AI inpainting.")
    else:
        st.info(f"❓ **Verdict: Undetermined / Out-of-Distribution**")
        st.write("📌 *Plain-English Summary:* Forensic indicators are balanced or heavy compression has degraded sensor signals. The system maintains formal scientific uncertainty rather than guessing.")

    # 2. Probability Gauges with Visual Progress Bars
    st.markdown("##### 📈 Authenticity Probabilities")
    st.caption("Calibrated Bayesian probabilities separating synthetic generation from genuine capture.")
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("🤖 AI-Generated", f"{p_ai:.1f}%")
    col_p2.metric("📷 Camera Authentic", f"{p_real:.1f}%")
    col_p3.metric("❓ Undetermined", f"{p_und:.1f}%")

    st.progress(min(1.0, max(0.0, p_ai / 100.0)), text=f"Generative AI Probability: {p_ai:.1f}%")

    st.markdown("---")

    # 3. Global AI Generator Attribution & Fingerprint
    attr = decision.get("model_attribution", {})
    if attr:
        st.markdown("##### 🌐 Global AI Generator Attribution")
        st.caption("Cross-matched against international AI models (US, China, Europe, Open-Source).")
        acol1, acol2, acol3 = st.columns(3)
        acol1.metric("Attributed Model", attr.get("attributed_model", "Unattributable"))
        acol2.metric("Region / Country", attr.get("region_of_origin", "Global"))
        acol3.metric("Attribution Match", f"{int(attr.get('attribution_confidence', 0) * 100)}%")

        candidates = attr.get("top_candidates", [])
        if candidates:
            cand_items = [f"**{c.get('model_name')}** ({c.get('region')}): `{c.get('attribution_probability')}%`" for c in candidates[:3]]
            st.caption("Top Candidates: " + " • ".join(cand_items))

        if attr.get("watermark_detected"):
            st.warning("⚠️ **Visible Impressioned Logo/Emblem Detected** in media corner.")

    # 4. Content & Scene Intelligence ("What is depicted in the media")
    st.markdown("##### 🌐 Scene, Content & Environmental Depiction")
    st.caption("Contextual breakdown of entities, environment, lighting, and tone present in the file.")
    inv = decision.get("content_inventory", {})
    if inv:
        s1, s2 = st.columns(2)
        with s1:
            st.write(f"• **Depiction / Purpose:** `{inv.get('photographic_purpose', 'General Scene')}`")
            st.write(f"• **Setting & Surroundings:** `{inv.get('location_context', 'N/A')}` • `{inv.get('setting_type', 'General')}`")
            st.write(f"• **Daytime & Lighting:** `{inv.get('estimated_daytime', 'Daylight')}` ({inv.get('lighting_style', 'Ambient')})")
            st.write(f"• **Atmosphere & Tone:** `{inv.get('color_tone', 'Neutral')}` • `{inv.get('atmospheric_mood', 'Balanced')}`")
        with s2:
            st.write(f"• **Living Humans:** `{inv.get('persons_count', 0)}` person(s), `{inv.get('faces_count', 0)}` face(s)")
            if inv.get("animals_detected"):
                st.write(f"• **Animals:** {', '.join(inv.get('animal_types', []))}")
            if inv.get("vehicles_detected"):
                st.write(f"• **Vehicles:** {', '.join(inv.get('vehicle_types', []))}")
            items = inv.get("identified_items", [])
            if items:
                st.write(f"• **Identified Items:** {', '.join(f'`{i}`' for i in items[:3])}")
            if inv.get("dominant_audio_type") and inv.get("dominant_audio_type") != "N/A":
                st.write(f"• **Audio Acoustics:** `{inv.get('dominant_audio_type')}` ({inv.get('acoustic_environment', 'Studio')})")

    # 5. Provenance & C2PA Credentials
    prov = decision.get("provenance", {})
    st.markdown("##### 🛡️ Provenance & Content Credentials (C2PA)")
    pr1, pr2, pr3 = st.columns(3)
    pr1.metric("C2PA Manifest", "PRESENT" if prov.get("c2pa_present") else "ABSENT")
    pr2.metric("Signature", prov.get("c2pa_signature", "Absent"))
    pr3.metric("Hardware EXIF", "Found" if prov.get("hardware_make") else "None")

    if prov.get("hardware_make"):
        st.caption(f"📷 Camera Hardware: **{prov.get('hardware_make')} {prov.get('hardware_model') or ''}**")
    else:
        st.caption("ℹ️ *NIST Rule: Absence of C2PA/EXIF metadata indicates UNKNOWN provenance, not evidence of synthetic creation.*")

    # 6. Localization
    loc = decision.get("localization", {})
    if loc.get("suspicious_image_area_pct", 0) > 0 or loc.get("suspicious_video_duration_pct", 0) > 0 or loc.get("suspicious_audio_duration_pct", 0) > 0:
        st.markdown("##### 📍 Manipulation Localization")
        if loc.get("suspicious_image_area_pct", 0) > 0:
            st.write(f"• **Manipulated Frame Area:** `{loc.get('suspicious_image_area_pct', 0):.1f}%` of pixels")
        if loc.get("suspicious_video_duration_pct", 0) > 0:
            st.write(f"• **Manipulated Timeline Duration:** `{loc.get('suspicious_video_duration_pct', 0):.1f}%` of video")
        if loc.get("suspicious_audio_duration_pct", 0) > 0:
            st.write(f"• **Manipulated Speech Duration:** `{loc.get('suspicious_audio_duration_pct', 0):.1f}%` of audio")

    # 7. Complete Forensic Audit Trail
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
                memory = ForensicMemory()
                ratings = {
                    "real_score": real_score,
                    "synthetic_score": synth_score,
                    "forged_score": forged_score,
                    "undetected_score": undetected_score,
                }
                reg_res = memory.register_feedback(
                    media_path=media_path,
                    modality=modality,
                    ratings=ratings,
                    ground_truth=ground_truth,
                    forensic_data=forensic_data,
                    user_notes=user_notes,
                    generator_tag=generator_tag,
                )

            if reg_res.get("success"):
                st.success("✅ **Forensic Feedback Successfully Remembered!**")
                update_info = reg_res.get("auto_update", {})
                modifications = update_info.get("modifications", [])
                if modifications:
                    st.info("🧠 **Dynamic Calibration Updates:**\n" + "\n".join(f"• {m}" for m in modifications))

                finetune = update_info.get("online_finetune", {})
                if finetune.get("status") == "success":
                    st.success(f"⚡ **Online Neural Checkpoint Updated!** ({finetune.get('samples_trained')} samples trained)")
                elif finetune.get("status") == "skipped":
                    st.caption(f"ℹ️ Online fine-tuning: {finetune.get('reason')}")
            else:
                st.error("Failed to register feedback.")

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
        import json
        st.download_button(
            label="💾 Download Official Forensic Dossier (JSON)",
            data=json.dumps(full_dossier_payload, indent=2, default=str),
            file_name=f"forensic_dossier_{Path(media_path).stem}.json",
            mime="application/json",
            key=f"{unique_key}_dl_btn",
        )

