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
from dataclasses import dataclass, field
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
    Single source of truth for percentage normalization, used by audio_detector,
    image_detector, video_detector, and this module's own generate_final_decision.
    Guarantees ai_pct + real_pct + undecided_pct == 100.0 and all values >= 0.0.

    Two call conventions, matching the two real use cases in this codebase:

    - undecided_val given (per-modality detectors, which already derived their own
      undecided margin from Shannon epistemic uncertainty): all three values are
      scaled together by a single factor so they sum to 100, preserving their
      relative proportions. This is the exact algorithm the per-modality scoring
      modules used before being consolidated here -- do not change it without
      re-validating every detector.py threshold that was calibrated against it.
    - undecided_val omitted (the cross-modal fusion path below, which has no
      precomputed entropy margin to work with): undecided is instead derived from
      how close ai/real are to each other (a wide gap -> low undecided), then
      ai/real are rescaled into whatever percentage remains.
    """
    ai = max(0.0, float(ai_val))
    real = max(0.0, float(real_val))

    if undecided_val is not None:
        undecided = max(float(min_undecided), float(undecided_val))
        total = ai + real + undecided
        if total <= 0.0:
            return 0.0, 0.0, 100.0
        scale = 100.0 / total
        ai_pct = round(ai * scale, decimals)
        real_pct = round(real * scale, decimals)
        undecided_pct = round(max(0.0, 100.0 - (ai_pct + real_pct)), decimals)
        return ai_pct, real_pct, undecided_pct

    gap = min(1.0, abs(ai - real) / 100.0)  # ai/real are percentages (0-100); the margin needs a 0-1 gap
    undecided = max(float(min_undecided), (1.0 - gap) * 20.0)
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


_STATUS_LIKELY_SYNTHETIC = "LIKELY_SYNTHETIC"
_STATUS_LIKELY_AUTHENTIC = "LIKELY_AUTHENTIC"
_STATUS_PARTIAL = "PARTIALLY_SYNTHETIC_OR_EDITED"
_STATUS_UNDETERMINED = "UNDETERMINED"
_AUTHENTIC_TAXONOMY = ("AUTHENTIC_REAL_PHOTOGRAPH", "AUTHENTIC_EDITED")
_SCORE_BACKED_AUTHENTIC = ("AUTHENTIC_REAL_PHOTOGRAPH", "AUTHENTIC_EDITED", "AUTHENTIC_RECAPTURED_SCREEN")


def _attribution_stub(label: str, model_key: str = "unknown", region: str = "N/A") -> Dict[str, Any]:
    return {
        "attributed_model": label,
        "model_key": model_key,
        "region_of_origin": region,
        "attribution_confidence": 0.0,
        "attribution_cues": [],
        "top_candidates": [],
    }


def _terminal_dossier(
    status: str,
    reason: str,
    trail: List[str],
    *,
    content_valid: bool,
    attribution: Dict[str, Any],
    inventory: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    cross_modal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dossier for outcomes that stop before any fusion (invalid file, blank media, no usable evidence)."""
    out: Dict[str, Any] = {
        "content_valid": content_valid,
        "final_status": status,
        "reason": reason,
        "ai_detected": False,
        "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
        "content_inventory": inventory or {},
        "forensic_modality_results": {},
        "model_attribution": attribution,
        "localization": {},
        "provenance": provenance or {},
        "evidence_trail": trail,
    }
    if cross_modal is not None:
        out["cross_modal"] = cross_modal
    return out


@dataclass
class _Evidence:
    """Evidence gathered from every modality before fusion."""

    probs: List[Tuple[float, float, float]] = field(default_factory=list)  # (ai%, real%, weight)
    trail: List[str] = field(default_factory=list)
    img_ai: float = 0.0
    img_spatial_area: float = 0.0
    vid_ai: float = 0.0
    vid_dur_pct: float = 0.0
    has_video: bool = False
    aud_ai: float = 0.0
    aud_real: float = 0.0
    aud_dur_pct: float = 0.0
    c2pa: Dict[str, Any] = field(default_factory=dict)


def _read_pair(result: Dict[str, Any]) -> Tuple[float, float]:
    """(ai%, real%) from either {ai_percentage, real_percentage} or {ai_probability} (0-1 or 0-100)."""
    if "ai_percentage" in result:
        return float(result.get("ai_percentage", 0.0)), float(result.get("real_percentage", 0.0))
    if "ai_probability" in result:
        p = float(result.get("ai_probability", 0.0))
        ai = p * 100.0 if p <= 1.0 else p
        return ai, 100.0 - ai
    return 0.0, 0.0


def _collect_image(ev: _Evidence, ai_result: Optional[Dict[str, Any]]) -> None:
    if not (ai_result and (ai_result.get("is_available") or "ai_percentage" in ai_result or "ai_probability" in ai_result)):
        return
    ev.img_ai, img_real = _read_pair(ai_result)
    ev.img_spatial_area = float(ai_result.get("ai_spatial_area_pct", 0.0))
    if ev.img_ai > 0.0 or img_real > 0.0 or ai_result.get("is_available"):
        ev.probs.append((ev.img_ai, img_real, 1.0))
    ev.trail.extend(f"[Image Forensic] {cue}" for cue in ai_result.get("forensic_cues", []))


def _collect_video(ev: _Evidence, video_result: Optional[Dict[str, Any]], quality_result: Dict[str, Any]) -> None:
    vid = video_result if (video_result and "ai_percentage" in video_result) else quality_result.get("ai_video_rating", {})
    if not vid:
        return
    ev.has_video = True
    ev.vid_ai = float(vid.get("ai_percentage", 0.0))
    vid_real = float(vid.get("real_percentage", 0.0))
    ev.vid_dur_pct = float(vid.get("ai_duration_pct", vid.get("details", {}).get("ai_duration_pct", 0.0)))
    ev.probs.append((ev.vid_ai, vid_real, 1.2))  # slightly higher weight for temporal video
    ev.trail.extend(f"[Video Forensic] {cue}" for cue in vid.get("forensic_cues", []))
    temp = (video_result or {}).get("temporal_consistency") or quality_result.get("temporal_consistency", {})
    risk = temp.get("temporal_warping_risk", "LOW")
    if risk in ("HIGH", "CRITICAL", "HIGH_WARPING_DETECTED", "SUSPICIOUS_FLICKER"):
        ev.trail.append(f"[Video Temporal] High inter-frame warping risk ({risk}, motion delta: {temp.get('mean_motion_delta', 0.0):.2f})")


def _collect_audio(ev: _Evidence, audio_result: Optional[Dict[str, Any]]) -> None:
    if not (audio_result and (audio_result.get("has_audio_track") or "ai_percentage" in audio_result or "ai_probability" in audio_result)):
        return
    ev.aud_ai, ev.aud_real = _read_pair(audio_result)
    ev.aud_dur_pct = float(audio_result.get("ai_duration_pct", 0.0))
    if ev.aud_ai > 0.0 or ev.aud_real > 0.0 or audio_result.get("has_audio_track"):
        ev.probs.append((ev.aud_ai, ev.aud_real, 1.0))
    ev.trail.extend(f"[Audio Forensic] {cue}" for cue in audio_result.get("forensic_cues", []))


def _collect_provenance(ev: _Evidence, prov: Dict[str, Any]) -> None:
    """Only declared-AI provenance is scored. C2PA marker presence and camera EXIF are unauthenticated: reported, never scored."""
    c2pa = prov.get("c2pa") or prov  # nested (UI validator) or flat (package provenance) shape
    ev.c2pa = c2pa
    exif = prov.get("exif", {})
    if c2pa.get("c2pa_present") or prov.get("c2pa_status") in ("VERIFIED", "AI_DECLARED"):
        if c2pa.get("ai_declaration") or prov.get("c2pa_status") == "AI_DECLARED":
            ev.trail.append("[C2PA Provenance] Manifest explicitly asserts generative AI creation.")
            ev.probs.append((98.0, 1.0, 2.0))
        elif c2pa.get("is_signed") or prov.get("c2pa_status") == "VERIFIED":
            ev.trail.append("[C2PA Provenance] Content Credentials signature markers present (not cryptographically verified; no score credit).")
    elif exif.get("ai_signature_found") or prov.get("ai_signature_found"):
        details = exif.get("signature_details") or prov.get("signature_details") or "AI signature detected"
        ev.trail.append(f"[Metadata Provenance] {details}")
        ev.probs.append((95.0, 5.0, 1.5))
    elif exif.get("camera_make") or prov.get("camera_make"):
        make = exif.get("camera_make") or prov.get("camera_make")
        model = exif.get("camera_model") or prov.get("camera_model", "")
        ev.trail.append(f"[Hardware Provenance] Camera hardware EXIF tags present (unauthenticated, not scored): {make} {model}".strip())


def _collect_cross_modal(ev: _Evidence, cross_modal: Optional[Dict[str, Any]]) -> None:
    if not (cross_modal and cross_modal.get("is_multimodal")):
        return
    ev.trail.extend(f"[Cross-Modal] {cue}" for cue in cross_modal.get("cues", []))
    risk = cross_modal.get("tampering_risk", "LOW")
    if risk in ("HIGH", "CRITICAL"):
        ev.probs.append((88.0, 10.0, 1.5))
    elif risk == "MODERATE":
        ev.probs.append((70.0, 25.0, 1.0))


def _collect_attribution(ev: _Evidence, attribution: Optional[Dict[str, Any]]) -> None:
    """Attribution is explanation only: it derives from the same pixels/samples, so scoring it would double count."""
    if not attribution:
        return
    ev.trail.extend(f"[Model Attribution] {cue}" for cue in attribution.get("attribution_cues", []))
    conf = float(attribution.get("attribution_confidence", 0.0))
    if conf >= 0.45 and attribution.get("model_key", "") not in ("unknown_ai", ""):
        ev.trail.append(
            f"[Model Attribution] High-confidence fingerprint alignment: {attribution.get('attributed_model')} "
            f"({attribution.get('region_of_origin')}) at {int(conf * 100)}% match."
        )


def _decision_mode(ai_result: Optional[Dict[str, Any]], audio_result: Optional[Dict[str, Any]], ev: "_Evidence") -> str:
    """'image_authoritative' only for a lone image result carrying a taxonomy state; otherwise 'fused'."""
    if ai_result and ai_result.get("taxonomy_state") and not ev.has_video and not (audio_result and ev.aud_ai + ev.aud_real > 0.0):
        return "image_authoritative"
    return "fused"


def _authentic_attribution() -> Dict[str, Any]:
    return {
        "attributed_model": "None (Authentic Capture)",
        "model_key": "none_authentic",
        "region_of_origin": "Physical Sensor / Camera",
        "attribution_confidence": 0.0,
        "attribution_cues": ["Camera-like sensor noise and optical properties observed (heuristic; metadata is unauthenticated)."],
        "top_candidates": [],
    }


def _fuse(probs: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Weighted natural-log-odds pooling of (ai%, real%, weight) triples -> (p_ai, p_real, p_undecided) percentages."""
    pooled = 0.0
    for ai_p, real_p, w in probs:
        p_ai = np.clip(ai_p / 100.0, 1e-4, 1.0 - 1e-4)
        p_real = np.clip(real_p / 100.0, 1e-4, 1.0 - 1e-4)
        pooled += float(np.log(p_ai / p_real)) * w
    odds = float(np.exp(np.clip(pooled, -6.0, 6.0)))
    p_ai_fused = odds / (1.0 + odds)
    return normalize_percentages(ai_val=p_ai_fused * 100.0, real_val=(1.0 - p_ai_fused) * 100.0, decimals=1)


def _status_from_probabilities(p_ai: float, p_real: float) -> Tuple[str, str]:
    if p_ai >= 65.0:
        return _STATUS_LIKELY_SYNTHETIC, "Forensic signals indicate high likelihood of synthetic/generative AI creation."
    if p_real >= 55.0 and p_ai < 35.0:
        return _STATUS_LIKELY_AUTHENTIC, "Physical sensor noise, natural optics, and provenance support authentic capture."
    if p_ai >= 50.0:
        return _STATUS_PARTIAL, "Significant forensic anomalies detected; partial synthesis or post-processing likely."
    return _STATUS_UNDETERMINED, "Forensic evidence is balanced or out-of-distribution; explicit uncertainty maintained."


def _taxonomy_from_fused(final_status: str, p_ai: float, p_real: float) -> Tuple[str, str, str]:
    if final_status == _STATUS_UNDETERMINED and p_ai < 50.0:
        return ("UNDETERMINED", "Undetermined / Mixed Signals", "Forensic evidence is balanced or out-of-distribution.")
    if final_status in ("LIKELY_SYNTHETIC", "LIKELY AI-GENERATED") or p_ai >= 65.0:
        return ("FULLY_AI_GENERATED", "Fully AI Generated",
                "Synthesized end-to-end via generative diffusion or deep neural models.")
    if final_status in ("PARTIALLY_SYNTHETIC_OR_EDITED", "AI-ENHANCED / COMPOSITE") or p_ai >= 40.0:
        return ("AI_ENHANCED_COMPOSITE", "AI-Enhanced / Composite (Mix)",
                "Authentic base media augmented with neural synthesis, voice cloning, or deepfake manipulation.")
    if final_status in ("AUTHENTIC (CONVENTIONALLY EDITED)", "AUTHENTIC_EDITED"):
        return ("AUTHENTIC_EDITED", "Authentic Photograph (Edited / Graphic Design)",
                "Real photo subjected to conventional software edits (cropping, background removal, or Canva composition).")
    if final_status in ("LIKELY_AUTHENTIC", "LIKELY REAL") or p_real >= 60.0:
        return ("AUTHENTIC_REAL_PHOTOGRAPH", "Authentic Real Capture",
                "Contains authentic sensor noise, natural acoustic/optical properties, and unmanipulated physics.")
    return ("UNDETERMINED", "Undetermined / Mixed Signals", "Forensic evidence is balanced or out-of-distribution.")


def _inventory_summary(inv: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens the nested content inventory (image/video or audio shape) into the dossier's inventory block."""
    humans = inv.get("entities", {}).get("humans", {})
    animals = inv.get("entities", {}).get("animals", {})
    items = inv.get("contents_and_items", {})
    env = inv.get("environment_and_surroundings", {})
    lighting = inv.get("lighting_and_daytime", {})
    tone = inv.get("tone_and_mood", {})
    purpose = inv.get("purpose_and_depiction", {})
    scene = env.get("setting_type", inv.get("scene_type", "General Scene"))
    animal_types = animals.get("animal_types", [])
    vehicle_types = inv.get("vehicles", {}).get("vehicle_types", [])
    return {
        "persons_count": humans.get("persons_count", inv.get("persons_count", 0)),
        "faces_count": humans.get("faces_count", inv.get("faces_count", 0)),
        "animals_detected": len(animal_types) > 0,
        "animal_types": animal_types,
        "vehicles_detected": len(vehicle_types) > 0,
        "vehicle_types": vehicle_types,
        "identified_items": items.get("identified_items", []),
        "text_regions_count": items.get("text_regions_count", inv.get("text_regions_count", 0)),
        "setting_type": scene,
        "scene_type": scene,
        "location_context": env.get("location_context", "N/A"),
        "estimated_daytime": lighting.get("estimated_daytime", "Daylight"),
        "lighting_style": lighting.get("lighting_style", "Ambient"),
        "color_tone": tone.get("color_tone", "Neutral"),
        "atmospheric_mood": tone.get("atmospheric_mood", "Balanced"),
        "photographic_purpose": purpose.get("photographic_purpose", "General Depiction"),
        "depiction_summary": purpose.get("depiction_summary", ""),
        "audio_speakers": inv.get("estimated_speakers", inv.get("audio_speakers", 0)),
        "dominant_audio_type": inv.get("dominant_audio_type", "N/A"),
        "acoustic_environment": inv.get("acoustic_environment", "N/A"),
        "vocal_tone": inv.get("vocal_tone_and_delivery", "N/A"),
    }


def _what_it_represents(inv: Dict[str, Any]) -> str:
    parts = []
    purpose = inv["photographic_purpose"]
    if purpose and purpose not in ("General Scene", "General Depiction", "General"):
        parts.append(purpose)
    if inv["persons_count"] > 0:
        parts.append(f"{inv['persons_count']} person(s), {inv['faces_count']} face(s)")
    if inv["setting_type"] and inv["setting_type"] not in ("General Scene", "General"):
        parts.append(f"{inv['setting_type']} setting")
    if inv["estimated_daytime"] and inv["estimated_daytime"] != "Daylight":
        parts.append(f"{inv['estimated_daytime']}")
    return " • ".join(parts) if parts else (inv["depiction_summary"] or "Standard visual capture")


def _provenance_summary(prov: Dict[str, Any], c2pa: Dict[str, Any]) -> Dict[str, Any]:
    present = c2pa.get("c2pa_present", False)
    signature = "Marker present (unverified)" if c2pa.get("is_signed") else ("No signature marker" if present else "Absent")
    exif = prov.get("exif", {})
    return {
        "c2pa_present": present,
        "c2pa_signature": signature,
        "ai_declared_in_c2pa": c2pa.get("ai_declaration", False),
        "hardware_make": exif.get("camera_make"),
        "hardware_model": exif.get("camera_model"),
        "software": exif.get("software"),
        "provenance_verdict": prov.get("provenance_verdict", "PROVENANCE_UNKNOWN"),
    }


def generate_final_decision(
    file_validation: Dict[str, Any],
    quality_result: Dict[str, Any],
    ai_result: Optional[Dict[str, Any]] = None,
    audio_result: Optional[Dict[str, Any]] = None,
    content_inventory: Optional[Dict[str, Any]] = None,
    provenance_result: Optional[Dict[str, Any]] = None,
    cross_modal_result: Optional[Dict[str, Any]] = None,
    attribution_result: Optional[Dict[str, Any]] = None,
    *,
    video_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates a structured forensic dossier via multi-evidence log-odds fusion.

    `decision_mode` in the result states how the verdict was formed:
    - "image_authoritative": a single image result carrying a taxonomy state is the verdict (its own
      pooled, trust-policy-aware score is final; nothing else is fused on top of it).
    - "fused": video/audio/cross-modal/declared-AI provenance evidence is pooled. Attribution is explanation
      only (it derives from the same pixels/samples and would double count) and camera EXIF is never scored
      here (unauthenticated and forgeable; the image detector applies its own trust policy).

    Probabilities are heuristic and uncalibrated (`calibration_status`).
    """
    if not file_validation.get("readable", False):
        return _terminal_dossier(
            "INVALID_FILE", file_validation.get("error", "File is unreadable or corrupted."),
            ["File failed initial container reading and format integrity."],
            content_valid=False, attribution=_attribution_stub("Unattributable / File Error"),
        )
    if quality_result.get("is_blank", False) or quality_result.get("content_status") == "INVALID":
        return _terminal_dossier(
            "BLANK_OR_DEGRADED", "Media exhibits negligible information entropy or blank content.",
            ["Media contains blank, zero-variance, or completely degraded frames."],
            content_valid=False, attribution=_attribution_stub("Unattributable / Blank Content"),
            inventory=content_inventory, provenance=provenance_result,
        )

    prov = provenance_result or {}
    ev = _Evidence()
    _collect_image(ev, ai_result)
    _collect_video(ev, video_result, quality_result)
    _collect_audio(ev, audio_result)
    _collect_provenance(ev, prov)
    _collect_cross_modal(ev, cross_modal_result)
    _collect_attribution(ev, attribution_result)

    if not ev.probs:
        return _terminal_dossier(
            _STATUS_UNDETERMINED, "No diagnostic forensic signals available to evaluate authenticity.",
            ["Insufficient forensic indicators extracted."],
            content_valid=True, attribution=attribution_result or _attribution_stub("Unattributable"),
            inventory=content_inventory, provenance=provenance_result, cross_modal=cross_modal_result or {},
        )

    p_ai, p_real, p_undecided = _fuse(ev.probs)
    final_status, reason = _status_from_probabilities(p_ai, p_real)
    ai_detected = final_status in (_STATUS_LIKELY_SYNTHETIC, _STATUS_PARTIAL) or p_ai >= 50.0

    decision_mode = _decision_mode(ai_result, audio_result, ev)
    tax_reasons: List[str] = []
    if decision_mode == "image_authoritative":
        tax_state, tax_label, tax_desc = (ai_result.get(k) for k in ("taxonomy_state", "taxonomy_label", "taxonomy_description"))
        tax_reasons = list(ai_result.get("taxonomy_reasons", []))
        if "ai_percentage" in ai_result and "real_percentage" in ai_result:
            p_ai, p_real = float(ai_result["ai_percentage"]), float(ai_result["real_percentage"])
            p_undecided = float(ai_result.get("undecided_percentage", 0.0))
            final_status = ai_result.get("label") or final_status
            if tax_state in _SCORE_BACKED_AUTHENTIC and _status_from_probabilities(p_ai, p_real)[0] != _STATUS_LIKELY_AUTHENTIC:
                # The category must never claim more than the score supports: "real" needs real >= 55 % and AI < 35 %.
                tax_state, tax_label, tax_desc = _taxonomy_from_fused(_STATUS_UNDETERMINED, p_ai, p_real)
                tax_reasons = tax_reasons + [f"The score ({p_ai:.0f}% AI, {p_real:.0f}% real) is too close to call, so no verdict is given."]
                final_status, reason = _STATUS_UNDETERMINED, "Forensic evidence is balanced; explicit uncertainty maintained."
    else:
        tax_state, tax_label, tax_desc = _taxonomy_from_fused(final_status, p_ai, p_real)

    localization = {
        "suspicious_image_area_pct": ev.img_spatial_area,
        "suspicious_video_duration_pct": ev.vid_dur_pct,
        "suspicious_audio_duration_pct": ev.aud_dur_pct,
    }

    # Authentic files must not attribute to AI generators (unless a neural enhancer was detected).
    if tax_state in _AUTHENTIC_TAXONOMY and not (ai_result and ai_result.get("neural_enhancer_detected")):
        attribution_result = _authentic_attribution()
        localization["suspicious_image_area_pct"] = 0.0

    inventory = _inventory_summary(content_inventory or {})
    return {
        "content_valid": True,
        "final_status": final_status,
        "reason": reason,
        "ai_detected": ai_detected,
        "decision_mode": decision_mode,
        "calibration_status": "UNCALIBRATED_HEURISTIC",
        "authenticity_probabilities": {"p_ai": p_ai, "p_real": p_real, "p_undecided": p_undecided},
        "taxonomy_state": tax_state,
        "taxonomy_label": tax_label,
        "taxonomy_description": tax_desc,
        "taxonomy_reasons": tax_reasons,
        "what_it_represents": _what_it_represents(inventory),
        "content_inventory": inventory,
        "forensic_modality_results": {
            "image_ai_pct": ev.img_ai if ai_result else None,
            "video_ai_pct": ev.vid_ai if ev.has_video else None,
            "audio_ai_pct": ev.aud_ai if audio_result and audio_result.get("has_audio_track") else None,
        },
        "localization": localization,
        "provenance": _provenance_summary(prov, ev.c2pa),
        "cross_modal": cross_modal_result or {},
        "model_attribution": attribution_result or {
            "attributed_model": "Unattributable / Custom Fine-Tuned Model",
            "model_key": "unknown_ai",
            "region_of_origin": "Global / Open-Source",
            "attribution_confidence": 0.0,
            "attribution_cues": [],
            "top_candidates": [],
        },
        "evidence_trail": ev.trail,
    }
