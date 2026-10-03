"""ui.results.content: Scene, entity and inventory sections."""
from __future__ import annotations
from typing import Any, Dict, Optional
import streamlit as st
from image_detector.explain import IPTC_SOURCE_TYPE_MAPPING


def render_scene_and_content_intelligence(content_data: Dict[str, Any], modality: str = "image") -> None:
    """
    Renders comprehensive scene, content, environment, daytime, tone,
    entities, items, and purpose analysis BEFORE presenting AI detection.
    """
    if not content_data or "error" in content_data:
        return

    st.subheader("Scene and content")
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
        col1.metric(" Purpose / Depiction", purpose.get("photographic_purpose", "General Depiction"))
        col2.metric(" Setting & Environment", f"{env.get('location_context', 'Indoor')} • {env.get('setting_type', 'General')}")
        col3.metric(" Daytime & Lighting", lighting.get("estimated_daytime", "Daylight"))

        # Row 2: Living Entities & Items
        ecol1, ecol2, ecol3 = st.columns(3)
        human_str = f"{humans.get('persons_count', 0)} person(s), {humans.get('faces_count', 0)} face(s)" if humans.get("has_humans") else "None detected"
        ecol1.metric(" Living Humans", human_str)

        anim_list = animals.get("animal_types", [])
        ecol2.metric(" Animals Detected", ", ".join(anim_list) if anim_list else "None detected")

        veh_list = vehicles.get("vehicle_types", [])
        ecol3.metric(" Vehicles Detected", ", ".join(veh_list) if veh_list else "None detected")

        # Row 3: Items, Tone, Lighting details in expandable container
        with st.expander("View Full Scene Breakdown (Items, Color Tone, Lighting Details)", expanded=True):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**Objects**")
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
                st.markdown("**Atmosphere and lighting**")
                st.write(f"• **Color Tone:** `{tone.get('color_tone', 'Neutral')}`")
                st.write(f"• **Atmospheric Mood:** `{tone.get('atmospheric_mood', 'Balanced')}`")
                st.write(f"• **Lighting Style:** `{lighting.get('lighting_style', 'Ambient')}`")
                st.write(f"• **Color Temperature:** `{lighting.get('color_temperature', 'Neutral')}`")
                st.write(f"• **Vegetation Coverage:** `{env.get('vegetation_coverage_pct', 0)}%` | **Sky/Water:** `{env.get('sky_water_coverage_pct', 0)}%`")

    elif modality == "audio":
        col1, col2, col3 = st.columns(3)
        col1.metric(" Audio Stream Type", content_data.get("dominant_audio_type", "Audio"))
        col2.metric(" Acoustic Environment", content_data.get("acoustic_environment", "Studio"))
        col3.metric(" Vocal Delivery & Tone", content_data.get("vocal_tone_and_delivery", "Conversational"))

        with st.expander("View Acoustic Environment & Speech Details", expanded=True):
            st.write(f"• **Audio Purpose / Genre:** `{content_data.get('audio_purpose', 'N/A')}`")
            st.write(f"• **Estimated Speakers:** `{content_data.get('estimated_speakers', 0)}`")
            st.write(f"• **Silence Ratio:** `{content_data.get('silence_ratio_pct', 0)}%`")
            st.write(f"• **Dynamic Range Index:** `{content_data.get('dynamic_range_index', 0)}`")


def render_image_type_and_category(
    decision: Dict[str, Any],
    content_res: Dict[str, Any],
    ai_result: Dict[str, Any],
    nine_dims: Optional[Dict[str, Any]] = None,
) -> None:
    """Renders Image Type and Category Identification."""
    st.markdown("#### Category")
    st.caption("Classified taxonomy ontology, IPTC digital source type, visual genre, medium, and sensor spectrum.")

    tax_label = decision.get("taxonomy_label") or ai_result.get("taxonomy_label", "Analyzed Media")
    tax_state = decision.get("taxonomy_state") or ai_result.get("taxonomy_state", "UNDECIDED")
    iptc_code = (
        nine_dims.get("dimension_1", {}).get("iptc_digital_source_type")
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
    """Renders Algorithmic Detection & Quantified Inventory (Percentages & Counts)."""
    st.markdown("#### What was detected")
    st.caption("Precise percentages and counts for every entity, item, pixel property, and forensic probe identified in the image.")

    probs = decision.get("authenticity_probabilities", {})
    p_ai = float(probs.get("p_ai", ai_result.get("ai_percentage", 0.0)))
    p_real = float(probs.get("p_real", ai_result.get("real_percentage", 0.0)))
    p_und = float(probs.get("p_undecided", ai_result.get("undecided_percentage", 100.0)))
    ai_area = float(ai_result.get("ai_spatial_area_pct", 0.0))

    q1, q2, q3, q4 = st.columns(4)
    q1.metric(" P(AI-Generated)", f"{p_ai:.1f}%")
    q2.metric(" P(Authentic Camera)", f"{p_real:.1f}%")
    q3.metric(" P(Undetermined)", f"{p_und:.1f}%")
    q4.metric(" Manipulated Area", f"{ai_area:.1f}%")

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
