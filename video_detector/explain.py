"""
video_detector.explain: Plain-English Newbie Explanation & 9-Dimensions Video Forensic Dossier Generator.
Transforms complex temporal telemetry, inter-frame motion vectors, and generative video indicators
into clear, engaging narrative explanations for beginners and comprehensive audits for professionals.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, NamedTuple

from video_detector.config import NOISE_AI_THRESHOLD, NOISE_BASELINE

logger = logging.getLogger("video_detector.explain")


class _VideoDossierContext(NamedTuple):
    """Values derived once from the inputs and shared by the per-dimension builders."""

    prof: Any
    geom: Any
    vid_res: Any
    prov: Any
    attr: Any
    inv: Any
    entities: Any
    humans: Any
    env: Any
    light: Any
    tone: Any
    temp: Any
    w: Any
    h: Any
    fps: Any
    duration: Any
    total_frames: Any


def _video_dimension_1(c: _VideoDossierContext) -> Dict[str, Any]:
    """1. Acquisition Hardware & Provenance Identity"""
    has_c2pa = c.prov.get("c2pa_present", False)
    atoms = c.prov.get("container_atoms", [])
    d1 = {
        "dimension_id": 1,
        "title": "Dimension 1: Acquisition Hardware & Video Container Provenance",
        "description": "Inspects container atom structure (moov, mvhd, udta), encoder signatures, and C2PA manifests.",
        "container_atoms": atoms or [],
        "c2pa_status": "Present (markers only; not cryptographically verified)" if has_c2pa else "Absent (Neutral)",
        "hardware_origin": c.prov.get("camera_make") or "No camera make recorded",
        "provenance_verdict": c.prov.get("provenance_status", "PROVENANCE_UNKNOWN"),
    }
    return d1


def _video_dimension_2(c: _VideoDossierContext) -> Dict[str, Any]:
    """2. Photometric & Temporal Geometry"""
    d2 = {
        "dimension_id": 2,
        "title": "Dimension 2: Spatial Geometry & Temporal Framing Architecture",
        "description": "Examines resolution, frame rate (FPS), duration, aspect ratio, and total frame buffer depth.",
        "dimensions": f"{c.w} x {c.h} px" if c.w and c.h else "Not recorded",
        "frame_rate": f"{c.fps:.2f} FPS" if c.fps else "Not recorded",
        "duration": f"{c.duration:.2f} seconds ({c.total_frames} frames)" if c.duration else "Not recorded",
        "aspect_ratio": f"{c.geom['aspect_ratio']:.2f}:1" if c.geom.get("aspect_ratio") else "Not recorded",
        "bitrate_kbps": c.prof.get("bitrate_kbps"),
    }
    return d2


def _video_dimension_3(c: _VideoDossierContext) -> Dict[str, Any]:
    """3. Sensor Shot Noise & Micro-Grain Consistency"""
    noise_mean = c.vid_res.get("mean_frame_noise")
    d3 = {
        "dimension_id": 3,
        "title": "Dimension 3: Sensor Shot Noise & Micro-Grain Consistency",
        "description": "Measures the median-filter noise residual of the sampled frames (a heuristic stand-in for sensor noise, not PRNU).",
        "noise_score": noise_mean,
    }
    if noise_mean is None:
        d3.update(is_natural_noise=None, diagnosis="Not measured: no frame had enough detail to measure noise on.")
        return d3
    natural = noise_mean - NOISE_BASELINE >= NOISE_AI_THRESHOLD
    d3.update(
        is_natural_noise=natural,
        diagnosis=(
            f"Noise level ({noise_mean:.2f}) is above the level typical of denoised generated frames."
            if natural
            else f"Low noise floor ({noise_mean:.2f}); similar to the smooth output of generative denoisers, but also to clean or heavily compressed camera footage."
        ),
    )
    return d3


def _video_dimension_4(c: _VideoDossierContext) -> Dict[str, Any]:
    """4. Inter-Frame Motion Vectors & Warping"""
    motion_var = c.temp.get("motion_variance")
    warping_risk = c.temp.get("temporal_warping_risk")
    d4 = {
        "dimension_id": 4,
        "title": "Dimension 4: Inter-Frame Motion Vectors & Morphological Warping",
        "description": "Compares the variance of frame-to-frame differences against thresholds for warping, flicker and frozen footage.",
        "motion_variance": None if motion_var is None else round(motion_var, 2),
        "warping_risk": warping_risk,
    }
    if motion_var is None or warping_risk is None:
        d4["diagnosis"] = "Not measured: fewer than two frames could be compared."
    elif warping_risk in ("HIGH_WARPING_DETECTED", "SUSPICIOUS_FLICKER"):
        d4["diagnosis"] = f"Inter-frame change is unusually irregular ({warping_risk}); can indicate warping surfaces, but also fast cuts or heavy compression."
    elif warping_risk == "UNNATURAL_FREEZE":
        d4["diagnosis"] = f"Almost no frame-to-frame change ({warping_risk}); inconsistent with live motion capture."
    else:
        d4["diagnosis"] = "Frame-to-frame change is within the range of ordinary footage."
    return d4


def _video_dimension_5(c: _VideoDossierContext) -> Dict[str, Any]:
    """5. Temporal Diffusion Flickering & Latent Drift"""
    flicker = c.vid_res.get("diffusion_flicker") or {}
    score = flicker.get("flicker_score") if flicker.get("frames_compared", 0) >= 3 else None
    d5 = {
        "dimension_id": 5,
        "title": "Dimension 5: Temporal Diffusion Flickering & Latent Drift",
        "description": "Counts reversals of the frame-to-frame brightness change, a pattern seen in frame-by-frame diffusion generation.",
        "flicker_score": score,
        "has_diffusion_flicker": flicker.get("has_diffusion_flicker"),
    }
    if score is None:
        d5["diagnosis"] = "Not measured: fewer than three frames were sampled."
    elif flicker.get("has_diffusion_flicker"):
        d5["diagnosis"] = f"Brightness flickers frame to frame (score {score:.2f}); a pattern of unconstrained generation, though some cameras and codecs also produce it."
    else:
        d5["diagnosis"] = f"No rapid brightness flicker (score {score:.2f})."
    return d5


def _video_dimension_6(c: _VideoDossierContext) -> Dict[str, Any]:
    """6. Visual Genre & Semantic Subject-Matter Catalog"""
    d6 = {
        "dimension_id": 6,
        "title": "Dimension 6: Visual Genre & Semantic Subject-Matter Catalog",
        "description": "Catalogs entities, persons, faces, scene settings, and environmental lighting.",
        "persons_count": c.humans.get("persons_count", c.inv.get("persons_count", 0)),
        "faces_count": c.humans.get("faces_count", c.inv.get("faces_count", 0)),
        "setting": " • ".join(dict.fromkeys(str(p) for p in (c.env.get("setting_type"), c.env.get("setting")) if p)) or "Not determined",
        "lighting": c.light.get("daytime") or "Not determined",
        "identified_items": c.inv.get("contents_and_items", {}).get("identified_items", []),
    }
    return d6


def _video_dimension_7(c: _VideoDossierContext) -> Dict[str, Any]:
    """7. Video Synthesis Medium & Generative Model Archetype"""
    d7 = {
        "dimension_id": 7,
        "title": "Dimension 7: Video Synthesis Medium & Generative Model Archetype",
        "description": "Looks for what separates camera footage from generated or rendered footage: motion that warps, flicker between frames, missing grain.",
        "verdict": c.vid_res.get("label") or "Not determined",
        "ai_likelihood_pct": c.vid_res.get("ai_percentage"),
        "visual_medium": "Not determined",
        "is_ai_video": {"LIKELY AI-GENERATED": True, "LIKELY REAL": False}.get(c.vid_res.get("label")),
    }
    return d7


def _video_dimension_8(c: _VideoDossierContext) -> Dict[str, Any]:
    """8. Sensor Spectrum & Dynamic Range"""
    d8 = {
        "dimension_id": 8,
        "title": "Dimension 8: Sensor Spectrum & Dynamic Range",
        "description": "The average fine-grain level of the sampled frames. Colour dynamics and highlight or shadow preservation are not evaluated for video.",
        "mean_frame_noise": c.vid_res.get("mean_frame_noise"),
        "sensor_modality": "Not determined",
    }
    return d8


def _video_dimension_9(c: _VideoDossierContext) -> Dict[str, Any]:
    """9. Foundation Model Attribution & Watermarking"""
    d9 = {
        "dimension_id": 9,
        "title": "Dimension 9: Foundation Model Attribution & Watermarking",
        "description": "Names a video generator only when the file's own metadata declares one; otherwise reports none.",
        "attributed_generator": c.attr.get("attributed_model") or "Not attributable",
        "confidence": f"{int(c.attr.get('attribution_confidence', 0.0) * 100)}%",
        "watermark_detected": None,                      # no watermark detector exists for video: not measured
        "suspicious_duration_pct": f"{c.vid_res['ai_duration_pct']:.1f}%" if c.vid_res.get("ai_duration_pct") is not None else None,
    }
    return d9


def build_video_nine_dimensions_dossier(
    profile_data: Dict[str, Any],
    video_result: Dict[str, Any],
    content_inventory: Optional[Dict[str, Any]] = None,
    provenance_result: Optional[Dict[str, Any]] = None,
    attribution_result: Optional[Dict[str, Any]] = None,
    cross_modal_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Constructs an exhaustive 9-dimensional forensic analysis dossier for video assets."""
    prof = profile_data or {}
    geom = prof.get("geometry", {})
    vid_res = video_result or {}
    prov = provenance_result or {}
    attr = attribution_result or {}
    inv = content_inventory or {}
    entities = inv.get("living_entities") or inv.get("entities") or {}
    humans = entities.get("humans", {})
    env = inv.get("environment_and_surroundings") or inv.get("environment") or {}
    light = inv.get("lighting_and_daytime") or {}
    tone = inv.get("tone_and_mood") or {}
    temp = vid_res.get("temporal_consistency", {})

    w = geom.get("width", 0)
    h = geom.get("height", 0)
    fps = geom.get("fps", 0.0)
    duration = geom.get("duration_seconds", 0.0)
    total_frames = geom.get("total_frames", 0)

    c = _VideoDossierContext(prof=prof, geom=geom, vid_res=vid_res, prov=prov, attr=attr, inv=inv, entities=entities, humans=humans, env=env, light=light, tone=tone, temp=temp, w=w, h=h, fps=fps, duration=duration, total_frames=total_frames)
    return {
        "dimension_1": _video_dimension_1(c),
        "dimension_2": _video_dimension_2(c),
        "dimension_3": _video_dimension_3(c),
        "dimension_4": _video_dimension_4(c),
        "dimension_5": _video_dimension_5(c),
        "dimension_6": _video_dimension_6(c),
        "dimension_7": _video_dimension_7(c),
        "dimension_8": _video_dimension_8(c),
        "dimension_9": _video_dimension_9(c),
    }


def generate_video_newbie_explanation(
    filename: str,
    profile_data: Dict[str, Any],
    content_inventory: Dict[str, Any],
    video_result: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    """Generates an engaging, accessible narrative explanation of video forensics for non-technical users."""
    geom = profile_data.get("geometry", {})
    w = geom.get("width", 0)
    h = geom.get("height", 0)
    duration = geom.get("duration_seconds", 0.0)
    fps = geom.get("fps", 0.0)

    probs = decision.get("authenticity_probabilities", {})
    p_ai = probs.get("p_ai", 0.0)
    p_real = probs.get("p_real", 0.0)
    final_status = decision.get("final_status", "")

    inv = content_inventory or {}
    humans = inv.get("living_entities", {}).get("humans") or inv.get("entities", {}).get("humans", {})
    persons = humans.get("persons_count", inv.get("persons_count", 0))
    faces = humans.get("faces_count", inv.get("faces_count", 0))

    if persons == 1:
        subject_desc = f"an individual ({faces} visible face)" if faces > 0 else "an individual"
    elif persons > 1:
        subject_desc = f"a group of {persons} individuals ({faces} visible faces)"
    elif humans or "persons_count" in inv:
        subject_desc = "a scene with no people"
    else:
        subject_desc = None

    fps_text = f"running at **{fps:.1f} frames per second**" if fps else "with a frame rate that is not recorded in the file"
    section_what = (
        f"### 📹 What We Identified in this Video\n\n"
        f"This file (`{filename}`) is a **{duration:.1f}-second video recording** formatted at **{w} × {h} pixels** "
        f"{fps_text}." + (f" The video depicts **{subject_desc}**." if subject_desc else "") + "\n\n"
    )

    section_newbie = "### 💡 How Would You Explain This to a Newbie?\n\n"

    temporal = video_result.get("temporal_consistency") or {}
    flicker = video_result.get("diffusion_flicker") or {}
    measured = []
    if temporal.get("temporal_warping_risk") is not None:
        measured.append(f"frame-to-frame motion variance **{temporal.get('motion_variance', 0.0)}** (risk: {str(temporal['temporal_warping_risk']).replace('_', ' ').lower()})")
    if flicker.get("has_diffusion_flicker") is not None:
        measured.append("**diffusion-style flicker " + ("was" if flicker.get("has_diffusion_flicker") else "was not") + " seen** between frames")
    if video_result.get("mean_frame_noise") is not None:
        measured.append(f"average fine-grain level per frame **{video_result['mean_frame_noise']}**")
    in_this_file = ("In this file we measured " + ", ".join(measured) + ".") if measured else "No temporal measurement was available for this file."

    if final_status == "BLANK_OR_DEGRADED":
        body = "**The Simple Takeaway:** The video has nothing to analyse (every sampled frame is blank), so no verdict is given."
    elif final_status == "UNDETERMINED":
        body = (
            f"**The Simple Takeaway:** The evidence does not settle it (heuristic, uncalibrated AI-likelihood **{p_ai:.1f}%**). "
            f"{in_this_file} Treat this as inconclusive."
        )
    elif "SYNTHETIC" in final_status or p_ai >= 55.0:
        body = (
            f"**The Simple Takeaway:** Our heuristic (uncalibrated) score is **{p_ai:.1f}% AI-likelihood**, so this video leans towards "
            f"**generated or heavily synthesised** footage. That is a ranking aid, not proof.\n\n"
            f"**What that is based on:** {in_this_file} Footage generated by video models tends to show motion that warps or "
            f"flickers between frames and to lack camera grain; real footage shows neither. Heavy compression and editing can also produce these signs."
        )
    else:
        body = (
            f"**The Simple Takeaway:** This video is **consistent with a camera recording** (heuristic, uncalibrated estimate **{p_real:.1f}%**; "
            f"a ranking aid, not proof).\n\n"
            f"**What that is based on:** {in_this_file} No clear sign of generation was found, but generated clips that are short, compressed or "
            f"re-filmed can look the same to these checks."
        )

    return section_what + section_newbie + body
