"""
Unified Multi-Modal Decision Engine & Media Forensics Dossier Generator.
Produces a NIST-aligned forensic report clearly separating:
1. Authenticity Probability: P(AI) + P(Real) + P(Undecided) = 100%.
2. Localization: Spatial area manipulated (%) and Temporal duration manipulated (%).
3. Content Inventory: Persons, Faces, Text blocks, Objects, and Audio speech/music.
4. Provenance & C2PA Credentials: Cryptographic manifests, signatures, and EXIF traces.
5. Cross-Modal Consistency: Synchronization, asymmetry, and acoustic scene plausibility.
6. Forensic Evidence Trail: Exhaustive audit trail of all physical and spectral indicators.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def generate_final_decision(
    file_validation: Dict[str, Any],
    quality_result: Dict[str, Any],
    ai_result: Optional[Dict[str, Any]] = None,
    audio_result: Optional[Dict[str, Any]] = None,
    content_inventory: Optional[Dict[str, Any]] = None,
    provenance_result: Optional[Dict[str, Any]] = None,
    cross_modal_result: Optional[Dict[str, Any]] = None,
    attribution_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates a structured, evidence-backed forensic dossier report.
    """
    if not file_validation.get("readable", False):
        return {
            "content_valid": False,
            "final_status": "INVALID_FILE",
            "reason": file_validation.get("error", "File is unreadable or corrupted."),
            "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
            "content_inventory": {},
            "localization": {},
            "provenance": {},
            "evidence_trail": ["File failed initial container reading and format integrity."],
        }

    # Quality status
    is_blank = quality_result.get("is_blank", False) or quality_result.get("content_status") == "INVALID"
    if is_blank:
        return {
            "content_valid": False,
            "final_status": "BLANK_OR_DEGRADED",
            "reason": "Media exhibits negligible information entropy or blank content.",
            "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
            "content_inventory": content_inventory or {},
            "localization": {},
            "provenance": provenance_result or {},
            "evidence_trail": ["Media contains blank, zero-variance, or completely degraded frames."],
        }

    evidence_trail: List[str] = []

    # 1. Modality AI probabilities and evidence
    active_probs = []

    # Image
    img_ai = 0.0
    img_real = 0.0
    img_spatial_area = 0.0
    if ai_result and ai_result.get("is_available"):
        img_ai = float(ai_result.get("ai_percentage", 0.0))
        img_real = float(ai_result.get("real_percentage", 0.0))
        img_spatial_area = float(ai_result.get("ai_spatial_area_pct", 0.0))
        active_probs.append((img_ai, img_real, 1.0))
        for cue in ai_result.get("forensic_cues", []):
            evidence_trail.append(f"[Image Forensic] {cue}")

    # Video
    vid_ai = 0.0
    vid_real = 0.0
    vid_dur_pct = 0.0
    ai_vid = quality_result.get("ai_video_rating", {})
    if ai_vid:
        vid_ai = float(ai_vid.get("ai_percentage", 0.0))
        vid_real = float(ai_vid.get("real_percentage", 0.0))
        vid_dur_pct = float(ai_vid.get("details", {}).get("ai_duration_pct", 0.0))
        active_probs.append((vid_ai, vid_real, 1.2))  # slightly higher weight for temporal video
        temp_cons = quality_result.get("temporal_consistency", {})
        if temp_cons.get("temporal_warping_risk") in ("HIGH", "CRITICAL"):
            evidence_trail.append(
                f"[Video Temporal] High inter-frame warping risk (motion delta: {temp_cons.get('mean_motion_delta', 0.0):.2f})"
            )

    # Audio
    aud_ai = 0.0
    aud_real = 0.0
    aud_dur_pct = 0.0
    if audio_result and audio_result.get("has_audio_track"):
        aud_ai = float(audio_result.get("ai_percentage", 0.0))
        aud_real = float(audio_result.get("real_percentage", 0.0))
        aud_dur_pct = float(audio_result.get("ai_duration_pct", 0.0))
        active_probs.append((aud_ai, aud_real, 1.0))
        for cue in audio_result.get("forensic_cues", []):
            evidence_trail.append(f"[Audio Forensic] {cue}")

    # 2. Provenance Evidence
    prov = provenance_result or {}
    c2pa_data = prov.get("c2pa", {})
    if c2pa_data.get("c2pa_present"):
        if c2pa_data.get("ai_declaration"):
            evidence_trail.append("[C2PA Provenance] Manifest explicitly asserts generative AI creation.")
            active_probs.append((98.0, 1.0, 2.0))
        elif c2pa_data.get("is_signed"):
            evidence_trail.append("[C2PA Provenance] Valid cryptographically-signed Content Credentials detected.")
            active_probs.append((5.0, 95.0, 2.0))
    elif prov.get("exif", {}).get("ai_signature_found"):
        evidence_trail.append(f"[Metadata Provenance] {prov['exif'].get('signature_details')}")
        active_probs.append((95.0, 5.0, 1.5))
    elif prov.get("exif", {}).get("camera_make"):
        evidence_trail.append(f"[Hardware Provenance] Camera capture hardware recorded: {prov['exif'].get('camera_make')} {prov['exif'].get('camera_model')}")

    # 3. Cross-Modal Evidence
    if cross_modal_result and cross_modal_result.get("is_multimodal"):
        for cm_cue in cross_modal_result.get("cues", []):
            evidence_trail.append(f"[Cross-Modal] {cm_cue}")
        if cross_modal_result.get("tampering_risk") in ("HIGH", "CRITICAL"):
            active_probs.append((88.0, 10.0, 1.5))

    # 4. Global Generative Model Attribution Evidence
    if attribution_result:
        for attr_cue in attribution_result.get("attribution_cues", []):
            evidence_trail.append(f"[Model Attribution] {attr_cue}")
        attr_conf = float(attribution_result.get("attribution_confidence", 0.0))
        attr_key = attribution_result.get("model_key", "")
        if attr_conf >= 0.45 and attr_key not in ("unknown_ai", ""):
            active_probs.append((min(99.0, attr_conf * 100.0), 1.0, 1.35))
            evidence_trail.append(
                f"[Model Attribution] High-confidence fingerprint alignment: {attribution_result.get('attributed_model')} "
                f"({attribution_result.get('region_of_origin')}) at {int(attr_conf * 100)}% match."
            )

    # 5. Calibrated Probability Aggregation
    if active_probs:
        total_w = sum(w for _, _, w in active_probs)
        weighted_ai = sum(ai * w for ai, _, w in active_probs) / total_w
        weighted_real = sum(real * w for _, real, w in active_probs) / total_w
    else:
        weighted_ai, weighted_real = 50.0, 50.0

    gap = abs(weighted_ai - weighted_real)
    p_undecided = max(4.0, (100.0 - gap) * 0.18)
    norm_factor = (100.0 - p_undecided) / max(1.0, (weighted_ai + weighted_real))

    p_ai = round(weighted_ai * norm_factor, 1)
    p_real = round(weighted_real * norm_factor, 1)
    p_undecided = round(100.0 - (p_ai + p_real), 1)

    ai_detected = (p_ai >= 50.0)

    # Status & Reason
    if p_ai >= 65.0:
        final_status = "LIKELY_SYNTHETIC"
        reason = "Forensic signals indicate high likelihood of synthetic/generative AI creation."
    elif p_real >= 65.0:
        final_status = "LIKELY_AUTHENTIC"
        reason = "Physical sensor noise, natural optics, and provenance support authentic capture."
    elif p_ai >= 50.0:
        final_status = "PARTIALLY_SYNTHETIC_OR_EDITED"
        reason = "Significant forensic anomalies detected; partial synthesis or post-processing likely."
    else:
        final_status = "UNDETERMINED_OOD"
        reason = "Forensic evidence is balanced or out-of-distribution; explicit uncertainty maintained."

    # 5. Localization
    localization_info = {
        "suspicious_image_area_pct": img_spatial_area,
        "suspicious_video_duration_pct": vid_dur_pct,
        "suspicious_audio_duration_pct": aud_dur_pct,
    }

    # 6. Comprehensive Content & Scene Inventory
    inventory = content_inventory or {}
    humans = inventory.get("entities", {}).get("humans", {})
    animals = inventory.get("entities", {}).get("animals", {})
    vehicles = inventory.get("vehicles", {})
    items = inventory.get("contents_and_items", {})
    env = inventory.get("environment_and_surroundings", {})
    lighting = inventory.get("lighting_and_daytime", {})
    tone = inventory.get("tone_and_mood", {})
    purpose = inventory.get("purpose_and_depiction", {})

    persons_cnt = humans.get("persons_count", inventory.get("persons_count", 0))
    faces_cnt = humans.get("faces_count", inventory.get("faces_count", 0))
    text_cnt = items.get("text_regions_count", inventory.get("text_regions_count", 0))
    scene_typ = env.get("setting_type", inventory.get("scene_type", "General Scene"))
    location_ctx = env.get("location_context", "N/A")
    daytime_est = lighting.get("estimated_daytime", "Daylight")
    lighting_sty = lighting.get("lighting_style", "Ambient")
    color_tone_est = tone.get("color_tone", "Neutral")
    mood_est = tone.get("atmospheric_mood", "Balanced")
    purpose_est = purpose.get("photographic_purpose", "General Depiction")
    depiction_sum = purpose.get("depiction_summary", "")
    identified_items_list = items.get("identified_items", [])
    animal_types_list = animals.get("animal_types", [])
    vehicle_types_list = vehicles.get("vehicle_types", [])

    audio_spk = inventory.get("estimated_speakers", inventory.get("audio_speakers", 0))
    audio_typ = inventory.get("dominant_audio_type", "N/A")
    audio_env = inventory.get("acoustic_environment", "N/A")
    audio_tone = inventory.get("vocal_tone_and_delivery", "N/A")

    return {
        "content_valid": True,
        "final_status": final_status,
        "reason": reason,
        "ai_detected": ai_detected,
        "authenticity_probabilities": {
            "p_ai": p_ai,
            "p_real": p_real,
            "p_undecided": p_undecided,
        },
        "content_inventory": {
            "persons_count": persons_cnt,
            "faces_count": faces_cnt,
            "animals_detected": len(animal_types_list) > 0,
            "animal_types": animal_types_list,
            "vehicles_detected": len(vehicle_types_list) > 0,
            "vehicle_types": vehicle_types_list,
            "identified_items": identified_items_list,
            "text_regions_count": text_cnt,
            "setting_type": scene_typ,
            "scene_type": scene_typ,
            "location_context": location_ctx,
            "estimated_daytime": daytime_est,
            "lighting_style": lighting_sty,
            "color_tone": color_tone_est,
            "atmospheric_mood": mood_est,
            "photographic_purpose": purpose_est,
            "depiction_summary": depiction_sum,
            "audio_speakers": audio_spk,
            "dominant_audio_type": audio_typ,
            "acoustic_environment": audio_env,
            "vocal_tone": audio_tone,
        },
        "forensic_modality_results": {
            "image_ai_pct": img_ai if ai_result else None,
            "video_ai_pct": vid_ai if ai_vid else None,
            "audio_ai_pct": aud_ai if audio_result and audio_result.get("has_audio_track") else None,
        },
        "localization": localization_info,
        "provenance": {
            "c2pa_present": c2pa_data.get("c2pa_present", False),
            "c2pa_signature": "Valid" if c2pa_data.get("is_signed") else ("Unsigned" if c2pa_data.get("c2pa_present") else "Absent"),
            "ai_declared_in_c2pa": c2pa_data.get("ai_declaration", False),
            "hardware_make": prov.get("exif", {}).get("camera_make"),
            "hardware_model": prov.get("exif", {}).get("camera_model"),
            "software": prov.get("exif", {}).get("software"),
            "provenance_verdict": prov.get("provenance_verdict", "PROVENANCE_UNKNOWN"),
        },
        "cross_modal": cross_modal_result or {},
        "model_attribution": attribution_result or {
            "attributed_model": "Unattributable / Custom Fine-Tuned Model",
            "model_key": "unknown_ai",
            "region_of_origin": "Global / Open-Source",
            "attribution_confidence": 0.0,
            "attribution_cues": [],
            "top_candidates": [],
        },
        "evidence_trail": evidence_trail,
    }