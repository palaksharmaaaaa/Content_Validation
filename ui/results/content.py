"""ui.results.content: Scene, entity and inventory sections."""
from __future__ import annotations
from typing import Any, Dict, Optional
import streamlit as st

from ui.layout import render_table
from image_detector.explain import IPTC_SOURCE_TYPE_MAPPING


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return ""


def _recognition_rows(details: list, noun: str) -> list:
    rows = []
    for d in details:
        looks_like = d.get("recognized_as") or "not sure"
        rows.append({noun: str(d.get("name", "")).title(), "Looks like": f"{str(looks_like).title()} ({_pct(d.get('recognition_confidence'))})" if d.get("recognition_confidence") is not None else str(looks_like).title(),
                     "Detector confidence": _pct(d.get("score"))})
    return rows


def _person_label(group: int) -> str:
    return "?" if group is None or group < 0 else f"Person {chr(ord('A') + group % 26)}"


def _render_visual_scene(content: Dict[str, Any]) -> None:
    """Scene, people, animals, vehicles, objects, colours and atmosphere for an image or video frame."""
    humans = (content.get("entities") or {}).get("humans", {})
    animals = (content.get("entities") or {}).get("animals", {})
    vehicles = content.get("vehicles", {})
    items = content.get("contents_and_items", {})
    env = content.get("environment_and_surroundings", {})
    lighting = content.get("lighting_and_daytime", {})
    tone = content.get("tone_and_mood", {})
    purpose = content.get("purpose_and_depiction", {})

    place = str(env.get("setting_type") or env.get("setting") or "Unknown")
    where = {True: "indoors", False: "outdoors"}.get(env.get("indoor"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Place", place + (f" ({where})" if where else ""))
    c2.metric("Lighting", str(lighting.get("estimated_daytime") or "Unknown"))
    c3.metric("Kind of photo", str(purpose.get("primary_genre") or "General scene"))
    if env.get("scene_candidates"):
        st.caption("Other places considered: " + ", ".join(f"{c['label']} ({_pct(c['probability'])})" for c in env["scene_candidates"][1:]))

    st.markdown("**People**")
    persons, faces = humans.get("persons_count", 0), humans.get("faces_count", 0)
    st.write(f"{persons} person(s) detected, {faces} face(s) analysed" + (f", {humans.get('distinct_people_in_faces')} {'person' if humans.get('distinct_people_in_faces') == 1 else 'different people'}" if faces else ""))
    face_rows = []
    for i, d in enumerate((humans.get("deepfake_analysis") or {}).get("details", []), 1):
        expr = d.get("expression") or {}
        face_rows.append({"Face": i, "Expression": f"{str(expr.get('label', 'unknown')).title()} ({_pct(expr.get('confidence'))})" if expr else "unknown",
                          "Same person as": _person_label(d.get("same_person_group")), "Size (px)": f"{d['bbox'][2]} x {d['bbox'][3]}"})
    render_table(face_rows)

    animal_rows = _recognition_rows(animals.get("details", []), "Animal")
    vehicle_rows = _recognition_rows(vehicles.get("details", []), "Vehicle")
    if animal_rows:
        st.markdown("**Animals**")
        render_table(animal_rows)
    if vehicle_rows:
        st.markdown("**Vehicles**")
        render_table(vehicle_rows)
    if not animal_rows and not vehicle_rows:
        st.caption("No animals or vehicles detected.")

    st.markdown("**Objects**")
    item_list = items.get("identified_items", [])
    st.write(", ".join(f"`{item}`" for item in item_list) if item_list else "No prominent everyday objects.")

    colors = content.get("colors", [])
    if colors:
        st.markdown("**Dominant colours**")
        swatches = "".join(
            f'<span style="display:inline-block;margin:0 .6rem .4rem 0;white-space:normal"><span style="display:inline-block;width:1.1rem;height:1.1rem;'
            f'border-radius:.25rem;vertical-align:middle;margin-right:.35rem;border:1px solid rgba(128,128,128,.5);background:{c["hex"]}"></span>'
            f'{c["color_name"]} {c["coverage_pct"]:.0f}%</span>' for c in colors)
        st.markdown(swatches, unsafe_allow_html=True)

    st.markdown("**Atmosphere**")
    st.write(f"Colour tone **{tone.get('color_tone', 'Neutral')}**, contrast **{tone.get('atmospheric_mood', 'Balanced')}**, light **{lighting.get('lighting_style', 'Ambient')}**.")


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
        _render_visual_scene(content_data)

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
