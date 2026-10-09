"""
video_detector.dimension_checks.lifecycle: transcoding-cascade likelihood.
RELIABILITY class: explains why forensic traces may be degraded; never changes P(AI).
"""
from __future__ import annotations

from typing import Any, Dict

from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from video_detector.dimension_checks import _common as C
from video_detector.dimension_checks import container

PLATFORM_SIZES = {(1080, 1920), (720, 1280), (1920, 1080), (1280, 720), (640, 360), (854, 480), (480, 854), (360, 640)}


def get_stream(ctx: CheckContext) -> Dict[str, Any]:
    st = ctx.extra.get("stream")
    if st is None:
        st = C.stream_info(ctx.path)
        ctx.extra["stream"] = st
    return st


def bits_per_pixel(ctx: CheckContext) -> float:
    st = get_stream(ctx)
    try:
        size = ctx.path.stat().st_size
    except OSError:
        return 0.0
    px = st["width"] * st["height"] * st["fps"] * st["duration"]
    return (size * 8.0 / px) if px > 0 else 0.0


def assess_transcoding(ctx: CheckContext) -> Dict[str, Any]:
    st = get_stream(ctx)
    cues, score = [], 0.0
    bpp = bits_per_pixel(ctx)
    if 0 < bpp < 0.05:
        score += 0.30
        cues.append(f"very low bitrate ({bpp:.3f} bits per pixel per frame)")
    elif 0 < bpp < 0.10:
        score += 0.15
        cues.append(f"low bitrate ({bpp:.3f} bits per pixel per frame)")
    w, h = st["width"], st["height"]
    if (w, h) in PLATFORM_SIZES and (h > w or (w, h) in {(640, 360), (854, 480)}):
        score += 0.20
        cues.append(f"canonical platform size {w}x{h}" + (" (vertical)" if h > w else ""))
    head, _t, _s = C.read_windows(ctx.path)
    if C.sniff_video_format(head) in ("mp4", "mov"):
        boxes = container.check_isobmff_boxes(ctx).data
        enc = container.check_encoder_sei(ctx).data.get("encoders", {})
        meta = container.check_mp4_metadata(ctx).data
        if boxes.get("layout") == "FASTSTART" and (enc or "libavformat" in meta.get("libraries", [])):
            score += 0.25
            cues.append("fast-start layout with software-encoder strings (web export / re-encode)")
        if not meta.get("creation") and not meta.get("text_keys"):
            score += 0.15
            cues.append("container metadata stripped")
    likelihood = round(min(1.0, score), 2)
    level = "HIGH" if likelihood >= 0.6 else "MODERATE" if likelihood >= 0.35 else "LOW"
    return {"likelihood": likelihood, "level": level, "cues": cues}


@registry.register("video", "transcoding_cascade", phase="post")
def check_transcoding_cascade(ctx: CheckContext) -> Finding:
    res = assess_transcoding(ctx)
    high = res["likelihood"] >= 0.6
    return Finding(
        check_id="transcoding_cascade", dimension="Section21.5", stage="lifecycle", title="Transcoding-cascade likelihood",
        status=FindingStatus.WARN if high else FindingStatus.INFO, severity=Severity.LOW if high else Severity.NONE,
        evidence_class=EvidenceClass.RELIABILITY,
        detail=(f"Likelihood of multi-generation re-encoding: {res['level']} ({res['likelihood']:.2f}). "
                + ("Cues: " + "; ".join(res["cues"]) + ". Sensor-noise, flicker and motion cues may be degraded." if res["cues"] else "No re-encoding cues.")),
        data={"likelihood": res["likelihood"], "likelihood_level": res["level"], "cues": res["cues"]},
    )
