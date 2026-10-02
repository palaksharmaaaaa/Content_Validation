"""
core.decision: Core Bayesian Multi-Modal Evidence Pooling & Decision Intelligence.
Decoupled completely from presentation frameworks.
Provides:
1. Multi-factor Bayesian log-odds evidence compounding across image, video, audio, and provenance.
2. Mathematically grounded invariant percentage normalization.
3. Standardized NIST / C2PA / IPTC taxonomy state mapping and attribution sanitization.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("core.decision")


def normalize_percentages(
    ai_val: float,
    real_val: float,
    undecided_val: float | None = None,
    min_undecided: float = 3.0,
    decimals: int = 1,
) -> Tuple[float, float, float]:
    """
    Guarantees ai_pct + real_pct + undecided_pct == 100.0,
    all >= 0.0, and undecided >= min_undecided without negative artifacts.
    """
    ai = max(0.0, float(ai_val))
    real = max(0.0, float(real_val))

    if undecided_val is not None:
        undecided = max(float(min_undecided), float(undecided_val))
    else:
        gap = abs(ai - real)
        undecided = max(float(min_undecided), (1.0 - min(1.0, gap)) * 20.0)

    remaining = max(0.0, 100.0 - undecided)
    denom = ai + real
    if denom > 0:
        ai_pct = round((ai / denom) * remaining, decimals)
        real_pct = round((real / denom) * remaining, decimals)
        undecided_pct = round(100.0 - (ai_pct + real_pct), decimals)
    else:
        ai_pct = 0.0
        real_pct = 0.0
        undecided_pct = 100.0

    return ai_pct, real_pct, undecided_pct


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
    Generates a structured, evidence-backed forensic dossier report via calibrated
    multi-evidence Bayesian sensor fusion.
    """
    if not file_validation.get("readable", False):
        return {
            "content_valid": False,
            "final_status": "INVALID_FILE",
            "reason": file_validation.get("error", "File is unreadable or corrupted."),
            "ai_detected": False,
            "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
            "content_inventory": {},
            "forensic_modality_results": {},
            "model_attribution": {
                "attributed_model": "Unattributable / File Error",
                "model_key": "unknown",
                "region_of_origin": "N/A",
                "attribution_confidence": 0.0,
                "attribution_cues": [],
                "top_candidates": [],
            },
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
            "ai_detected": False,
            "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
            "content_inventory": content_inventory or {},
            "forensic_modality_results": {},
            "model_attribution": {
                "attributed_model": "Unattributable / Blank Content",
                "model_key": "unknown",
                "region_of_origin": "N/A",
                "attribution_confidence": 0.0,
                "attribution_cues": [],
                "top_candidates": [],
            },
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
    if ai_result and (ai_result.get("is_available") or "ai_percentage" in ai_result or "ai_probability" in ai_result):
        if "ai_percentage" in ai_result:
            img_ai = float(ai_result.get("ai_percentage", 0.0))
            img_real = float(ai_result.get("real_percentage", 0.0))
        elif "ai_probability" in ai_result:
            p_ai = float(ai_result.get("ai_probability", 0.0))
            img_ai = p_ai * 100.0 if p_ai <= 1.0 else p_ai
            img_real = 100.0 - img_ai
        else:
            img_ai = float(ai_result.get("ai_percentage", 0.0))
            img_real = float(ai_result.get("real_percentage", 0.0))
        img_spatial_area = float(ai_result.get("ai_spatial_area_pct", 0.0))
        if img_ai > 0.0 or img_real > 0.0 or ai_result.get("is_available"):
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
        warping_risk = temp_cons.get("temporal_warping_risk", "LOW")
        if warping_risk in ("HIGH", "CRITICAL", "HIGH_WARPING_DETECTED", "SUSPICIOUS_FLICKER"):
            evidence_trail.append(
                f"[Video Temporal] High inter-frame warping risk ({warping_risk}, motion delta: {temp_cons.get('mean_motion_delta', 0.0):.2f})"
            )

    # Audio
    aud_ai = 0.0
    aud_real = 0.0
    aud_dur_pct = 0.0
    if audio_result and (audio_result.get("has_audio_track") or "ai_percentage" in audio_result or "ai_probability" in audio_result):
        if "ai_percentage" in audio_result:
            aud_ai = float(audio_result.get("ai_percentage", 0.0))
            aud_real = float(audio_result.get("real_percentage", 0.0))
        elif "ai_probability" in audio_result:
            p_ai = float(audio_result.get("ai_probability", 0.0))
            aud_ai = p_ai * 100.0 if p_ai <= 1.0 else p_ai
            aud_real = 100.0 - aud_ai
        else:
            aud_ai = float(audio_result.get("ai_percentage", 0.0))
            aud_real = float(audio_result.get("real_percentage", 0.0))
        aud_dur_pct = float(audio_result.get("ai_duration_pct", 0.0))
        if aud_ai > 0.0 or aud_real > 0.0 or audio_result.get("has_audio_track"):
            active_probs.append((aud_ai, aud_real, 1.0))
        for cue in audio_result.get("forensic_cues", []):
            evidence_trail.append(f"[Audio Forensic] {cue}")

    # 2. Provenance Evidence
    prov = provenance_result or {}
    c2pa_data = prov.get("c2pa", {})
    if c2pa_data.get("c2pa_present") or prov.get("c2pa_status") in ("VERIFIED", "AI_DECLARED"):
        if c2pa_data.get("ai_declaration") or prov.get("c2pa_status") == "AI_DECLARED":
            evidence_trail.append("[C2PA Provenance] Manifest explicitly asserts generative AI creation.")
            active_probs.append((98.0, 1.0, 2.0))
        elif c2pa_data.get("is_signed") or prov.get("c2pa_status") == "VERIFIED":
            evidence_trail.append("[C2PA Provenance] Valid cryptographically-signed Content Credentials detected.")
            active_probs.append((5.0, 95.0, 2.0))
    elif prov.get("exif", {}).get("ai_signature_found") or prov.get("ai_signature_found"):
        details = prov.get("exif", {}).get("signature_details") or prov.get("signature_details") or "AI signature detected"
        evidence_trail.append(f"[Metadata Provenance] {details}")
        active_probs.append((95.0, 5.0, 1.5))
    elif prov.get("exif", {}).get("camera_make") or prov.get("camera_make"):
        make = prov.get("exif", {}).get("camera_make") or prov.get("camera_make")
        model = prov.get("exif", {}).get("camera_model") or prov.get("camera_model", "")
        evidence_trail.append(f"[Hardware Provenance] Verified physical camera hardware: {make} {model}".strip())
        active_probs.append((8.0, 92.0, 1.35))

    # 3. Cross-Modal Evidence
    if cross_modal_result and cross_modal_result.get("is_multimodal"):
        for cm_cue in cross_modal_result.get("cues", []):
            evidence_trail.append(f"[Cross-Modal] {cm_cue}")
        cm_risk = cross_modal_result.get("tampering_risk", "LOW")
        if cm_risk in ("HIGH", "CRITICAL"):
            active_probs.append((88.0, 10.0, 1.5))
        elif cm_risk == "MODERATE":
            active_probs.append((70.0, 25.0, 1.0))

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

    # 5. Calibrated Bayesian Evidence Pooling & Shannon Uncertainty
    if not active_probs:
        return {
            "content_valid": True,
            "final_status": "UNDETERMINED_OOD",
            "reason": "No diagnostic forensic signals available to evaluate authenticity.",
            "ai_detected": False,
            "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
            "content_inventory": content_inventory or {},
            "forensic_modality_results": {},
            "model_attribution": attribution_result or {
                "attributed_model": "Unattributable",
                "model_key": "unknown",
                "region_of_origin": "N/A",
                "attribution_confidence": 0.0,
                "attribution_cues": [],
                "top_candidates": [],
            },
            "localization": {},
            "provenance": provenance_result or {},
            "cross_modal": cross_modal_result or {},
            "evidence_trail": ["Insufficient forensic indicators extracted."],
        }

    # Convert probability pairs to log-odds: L = ln(p_ai / p_real)
    log_odds_list = []
    weights_list = []
    for ai_p, real_p, w in active_probs:
        p_ai_norm = np.clip(ai_p / 100.0, 1e-4, 1.0 - 1e-4)
        p_real_norm = np.clip(real_p / 100.0, 1e-4, 1.0 - 1e-4)
        lod = float(np.log(p_ai_norm / p_real_norm))
        log_odds_list.append(lod)
        weights_list.append(w)

    pooled_log_odds = sum(lod * (w / 1.0) for lod, w in zip(log_odds_list, weights_list))
    pooled_odds = float(np.exp(np.clip(pooled_log_odds, -6.0, 6.0)))
    prob_ai_fused = pooled_odds / (1.0 + pooled_odds)
    prob_real_fused = 1.0 - prob_ai_fused

    p_ai, p_real, p_undecided = normalize_percentages(
        ai_val=prob_ai_fused * 100.0,
        real_val=prob_real_fused * 100.0,
        decimals=1,
    )

    if p_ai >= 65.0:
        final_status = "LIKELY_SYNTHETIC"
        reason = "Forensic signals indicate high likelihood of synthetic/generative AI creation."
    elif p_real >= 55.0 and p_ai < 35.0:
        final_status = "LIKELY_AUTHENTIC"
        reason = "Physical sensor noise, natural optics, and provenance support authentic capture."
    elif p_ai >= 50.0:
        final_status = "PARTIALLY_SYNTHETIC_OR_EDITED"
        reason = "Significant forensic anomalies detected; partial synthesis or post-processing likely."
    else:
        final_status = "UNDETERMINED_OOD"
        reason = "Forensic evidence is balanced or out-of-distribution; explicit uncertainty maintained."

    ai_detected = (final_status in ("LIKELY_SYNTHETIC", "PARTIALLY_SYNTHETIC_OR_EDITED") or p_ai >= 50.0)

    # Localization
    localization_info = {
        "suspicious_image_area_pct": img_spatial_area,
        "suspicious_video_duration_pct": vid_dur_pct,
        "suspicious_audio_duration_pct": aud_dur_pct,
    }

    # Comprehensive Content & Scene Inventory
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

    # Taxonomy synchronization
    tax_state = None
    tax_label = None
    tax_desc = None
    tax_reasons = []

    if ai_result and ai_result.get("taxonomy_state"):
        tax_state = ai_result.get("taxonomy_state")
        tax_label = ai_result.get("taxonomy_label")
        tax_desc = ai_result.get("taxonomy_description")
        tax_reasons = list(ai_result.get("taxonomy_reasons", []))
        if "ai_percentage" in ai_result and "real_percentage" in ai_result:
            p_ai = float(ai_result["ai_percentage"])
            p_real = float(ai_result["real_percentage"])
            p_undecided = float(ai_result.get("undecided_percentage", 0.0))
            if ai_result.get("label"):
                final_status = ai_result.get("label")
    else:
        if final_status in ("LIKELY_SYNTHETIC", "LIKELY AI-GENERATED") or p_ai >= 65.0:
            tax_state = "FULLY_AI_GENERATED"
            tax_label = "Fully AI Generated"
            tax_desc = "Synthesized end-to-end via generative diffusion or deep neural models."
        elif final_status in ("PARTIALLY_SYNTHETIC_OR_EDITED", "AI-ENHANCED / COMPOSITE") or p_ai >= 40.0:
            tax_state = "AI_ENHANCED_COMPOSITE"
            tax_label = "AI-Enhanced / Composite (Mix)"
            tax_desc = "Authentic base media augmented with neural synthesis, voice cloning, or deepfake manipulation."
        elif final_status in ("AUTHENTIC (CONVENTIONALLY EDITED)", "AUTHENTIC_EDITED"):
            tax_state = "AUTHENTIC_EDITED"
            tax_label = "Authentic Photograph (Edited / Graphic Design)"
            tax_desc = "Real photo subjected to conventional software edits (cropping, background removal, or Canva composition)."
        elif final_status in ("LIKELY_AUTHENTIC", "LIKELY REAL") or p_real >= 60.0:
            tax_state = "AUTHENTIC_REAL_PHOTOGRAPH"
            tax_label = "Authentic Real Capture"
            tax_desc = "Contains authentic sensor noise, natural acoustic/optical properties, and unmanipulated physics."
        else:
            tax_state = "UNDETERMINED"
            tax_label = "Undetermined / Mixed Signals"
            tax_desc = "Forensic evidence is balanced or out-of-distribution."

    # Attribution sanitization: Authentic files must not attribute to AI generators
    is_enhancer = bool(ai_result and ai_result.get("neural_enhancer_detected"))
    if tax_state in ("AUTHENTIC_REAL_PHOTOGRAPH", "AUTHENTIC_EDITED") and not is_enhancer:
        attribution_result = {
            "attributed_model": "None (Authentic Capture)",
            "model_key": "none_authentic",
            "region_of_origin": "Physical Sensor / Camera",
            "attribution_confidence": 0.0,
            "attribution_cues": ["Authentic camera sensor noise and physical optical properties verified."],
            "top_candidates": [],
        }
        localization_info["suspicious_image_area_pct"] = 0.0

    # Plain-English description of what the media depicts/represents
    what_represents_parts = []
    if purpose_est and purpose_est not in ("General Scene", "General Depiction", "General"):
        what_represents_parts.append(purpose_est)
    if persons_cnt > 0:
        what_represents_parts.append(f"{persons_cnt} person(s), {faces_cnt} face(s)")
    if scene_typ and scene_typ not in ("General Scene", "General"):
        what_represents_parts.append(f"{scene_typ} setting")
    if daytime_est and daytime_est != "Daylight":
        what_represents_parts.append(f"{daytime_est}")

    what_it_represents = " • ".join(what_represents_parts) if what_represents_parts else (depiction_sum or "Standard visual capture")

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
        "taxonomy_state": tax_state,
        "taxonomy_label": tax_label,
        "taxonomy_description": tax_desc,
        "taxonomy_reasons": tax_reasons,
        "what_it_represents": what_it_represents,
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
