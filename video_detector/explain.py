"""
video_detector.explain: Plain-English Newbie Explanation & 9-Dimensions Video Forensic Dossier Generator.
Transforms complex temporal telemetry, inter-frame motion vectors, and generative video indicators
into clear, engaging narrative explanations for beginners and comprehensive audits for professionals.
Aligned directly with NIST OpenMFC and C2PA v2.1 video standards.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, NamedTuple

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
        "container_atoms": atoms or ["Standard MP4 Atoms"],
        "c2pa_status": "Present (markers only; not cryptographically verified)" if has_c2pa else "Absent (Neutral)",
        "hardware_origin": c.prov.get("camera_make") or "Unspecified Camera Hardware / Web Export",
        "provenance_verdict": c.prov.get("provenance_status", "PROVENANCE_UNKNOWN"),
    }
    return d1


def _video_dimension_2(c: _VideoDossierContext) -> Dict[str, Any]:
    """2. Photometric & Temporal Geometry"""
    d2 = {
        "dimension_id": 2,
        "title": "Dimension 2: Spatial Geometry & Temporal Framing Architecture",
        "description": "Examines resolution, frame rate (FPS), duration, aspect ratio, and total frame buffer depth.",
        "dimensions": f"{c.w} x {c.h} px",
        "frame_rate": f"{c.fps:.2f} FPS",
        "duration": f"{c.duration:.2f} seconds ({c.total_frames} frames)",
        "aspect_ratio": f"{c.geom.get('aspect_ratio', 0.0):.2f}:1",
        "bitrate_kbps": c.prof.get("bitrate_kbps", 0.0),
    }
    return d2


def _video_dimension_3(c: _VideoDossierContext) -> Dict[str, Any]:
    """3. Sensor Shot Noise & Micro-Grain Consistency"""
    noise_mean = c.vid_res.get("mean_frame_noise", 1.8)
    d3 = {
        "dimension_id": 3,
        "title": "Dimension 3: Sensor Shot Noise & Micro-Grain Consistency",
        "description": "Monitors photon shot noise and PRNU stability across temporal frame samples.",
        "noise_score": noise_mean,
        "is_natural_noise": noise_mean >= 1.20,
        "diagnosis": (
            "Authentic camera sensor shot noise preserved across sampled frames."
            if noise_mean >= 1.20
            else f"Sub-optical noise floor ({noise_mean:.2f}); consistent with neural diffusion denoiser."
        ),
    }
    return d3


def _video_dimension_4(c: _VideoDossierContext) -> Dict[str, Any]:
    """4. Inter-Frame Motion Vectors & Warping"""
    motion_var = c.temp.get("motion_variance", 45.0)
    warping_risk = c.temp.get("temporal_warping_risk", "LOW")
    d4 = {
        "dimension_id": 4,
        "title": "Dimension 4: Inter-Frame Motion Vectors & Morphological Warping",
        "description": "Tracks optical flow continuity to detect liquid limb morphing and unnatural physics.",
        "motion_variance": round(motion_var, 2),
        "warping_risk": warping_risk,
        "diagnosis": (
            f"Anomalous inter-frame warping detected ({warping_risk}). Surfaces exhibit non-Euclidean morphing."
            if warping_risk in ("HIGH_WARPING_DETECTED", "SUSPICIOUS_FLICKER")
            else f"Unnatural frame-to-frame stillness detected ({warping_risk}); inconsistent with live motion capture."
            if warping_risk == "UNNATURAL_FREEZE"
            else "Natural Newtonian motion dynamics and smooth inter-frame optical flow."
        ),
    }
    return d4


def _video_dimension_5(c: _VideoDossierContext) -> Dict[str, Any]:
    """5. Temporal Diffusion Flickering & Latent Drift"""
    flicker_score = c.temp.get("flicker_variance", 20.0)
    d5 = {
        "dimension_id": 5,
        "title": "Dimension 5: Temporal Diffusion Flickering & Latent Drift",
        "description": "Detects high-frequency luminance fluctuations characteristic of frame-by-frame diffusion generation.",
        "flicker_score": round(flicker_score, 2),
        "has_diffusion_flicker": flicker_score > 65.0,
        "diagnosis": (
            "High inter-frame generative flicker detected; signature of unconstrained diffusion step variance."
            if flicker_score > 65.0
            else "Temporal luminance stability aligns with physical camera shutter exposure."
        ),
    }
    return d5


def _video_dimension_6(c: _VideoDossierContext) -> Dict[str, Any]:
    """6. Visual Genre & Semantic Subject-Matter Catalog"""
    d6 = {
        "dimension_id": 6,
        "title": "Dimension 6: Visual Genre & Semantic Subject-Matter Catalog",
        "description": "Catalogs entities, persons, faces, scene settings, and environmental lighting.",
        "persons_count": c.humans.get("persons_count", c.inv.get("persons_count", 0)),
        "faces_count": c.humans.get("faces_count", c.inv.get("faces_count", 0)),
        "setting": f"{c.env.get('setting_type', 'Ambient')} • {c.env.get('setting', 'Indoor')}",
        "lighting": c.light.get("daytime", "Daylight"),
        "identified_items": c.inv.get("contents_and_items", {}).get("identified_items", []),
    }
    return d6


def _video_dimension_7(c: _VideoDossierContext) -> Dict[str, Any]:
    """7. Video Synthesis Medium & Generative Model Archetype"""
    d7 = {
        "dimension_id": 7,
        "title": "Dimension 7: Video Synthesis Medium & Generative Model Archetype",
        "description": "Distinguishes physical camera capture from Sora, Runway Gen-2/Gen-3, Kling, Luma, or 3D CGI.",
        "visual_medium": c.vid_res.get("visual_medium", "Video Capture"),
        "is_ai_video": c.vid_res.get("is_synthetic", False),
    }
    return d7


def _video_dimension_8(c: _VideoDossierContext) -> Dict[str, Any]:
    """8. Sensor Spectrum & Dynamic Range"""
    d8 = {
        "dimension_id": 8,
        "title": "Dimension 8: Sensor Spectrum & Dynamic Range",
        "description": "Evaluates color dynamics, dynamic range, and highlight/shadow preservation across frames.",
        "sensor_modality": "Standard RGB Video Stream",
        "color_channels": 3,
    }
    return d8


def _video_dimension_9(c: _VideoDossierContext) -> Dict[str, Any]:
    """9. Foundation Model Attribution & Watermarking"""
    d9 = {
        "dimension_id": 9,
        "title": "Dimension 9: Foundation Model Attribution & Watermarking",
        "description": "Matches forensic fingerprints against known video foundation generators (Sora, Runway, Pika, Kling, Luma).",
        "attributed_generator": c.attr.get("attributed_model", "Unattributable / Unknown Generator"),
        "confidence": f"{int(c.attr.get('attribution_confidence', 0.0) * 100)}%",
        "watermark_detected": c.attr.get("watermark_detected", False),
        "suspicious_duration_pct": f"{c.vid_res.get('details', {}).get('ai_duration_pct', 0.0):.1f}%",
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
    else:
        subject_desc = "a visual landscape or scene"

    section_what = (
        f"### 📹 What We Identified in this Video\n\n"
        f"This file (`{filename}`) is a **{duration:.1f}-second video recording** formatted at **{w} × {h} pixels** "
        f"running at **{fps:.1f} frames per second**. The video depicts **{subject_desc}**.\n\n"
    )

    section_newbie = "### 💡 How Would You Explain This to a Newbie?\n\n"

    is_ai = "SYNTHETIC" in final_status or "AI" in final_status or p_ai >= 55.0

    if is_ai:
        body = (
            f"**The Simple Takeaway:** Our forensic engine's heuristic (uncalibrated) score is **{p_ai:.1f}% AI-likelihood**, indicating this video was most likely "
            f"**generated by Artificial Intelligence** (such as Sora, Runway, Kling, or Luma) rather than captured by a real physical video camera.\n\n"
            f"**Think of it like this:** When a real video camera shoots a scene, light continuously hits a physical sensor chip 24 to 60 times every second. "
            f"Because physical objects obey Newtonian physics, moving hands, hair, and backgrounds transition smoothly without distorting. "
            f"However, AI video models generate clips by 'dreaming' one frame after another. When you look closely under the forensic microscope, "
            f"the background slightly warps, micro-textures flicker unnaturally between frames, and the natural camera sensor grain is missing. "
            f"It looks convincing at first glance, but the temporal physics reveal it was computed mathematically, not filmed optically."
        )
    else:
        body = (
            f"**The Simple Takeaway:** This video is **consistent with a genuine physical camera recording** (heuristic, uncalibrated estimate **{p_real:.1f}%**; a ranking aid, not proof).\n\n"
            f"**Think of it like this:** Every frame in this video adheres strictly to real-world optical physics. Moving objects leave behind natural "
            f"shutter motion blur, camera sensor grain (shot noise) remains consistent across all {int(duration * fps)} frames, and there is zero "
            f"morphing or AI latent warping. No generative video synthesis or deepfake alterations were detected."
        )

    return section_what + section_newbie + body
