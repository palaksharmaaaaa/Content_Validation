"""
video_detector.dimension_checks.reliability: confidence limiters.
RELIABILITY class: informational, never alters P(AI).
"""
from __future__ import annotations

from typing import List

from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from video_detector.dimension_checks import container, signal
from video_detector.dimension_checks.lifecycle import assess_transcoding, bits_per_pixel, get_stream

LOW_RES_MIN_SIDE = 360
SHORT_SECONDS = 2.0
HEAVY_COMPRESSION_BPP = 0.05


@registry.register("video", "confidence_limiters", phase="post")
def check_confidence_limiters(ctx: CheckContext) -> Finding:
    limiters: List[str] = []
    st = get_stream(ctx)
    if st["width"] and st["height"] and min(st["width"], st["height"]) < LOW_RES_MIN_SIDE:
        limiters.append(f"Low resolution ({st['width']}x{st['height']}): pixel-level and noise cues are unreliable")
    if 0 < st["duration"] < SHORT_SECONDS:
        limiters.append(f"Very short clip ({st['duration']:.1f} s): temporal statistics are unstable")
    bpp = bits_per_pixel(ctx)
    if 0 < bpp < HEAVY_COMPRESSION_BPP:
        limiters.append(f"Heavy compression ({bpp:.3f} bits per pixel per frame): fine texture and noise cues are suppressed")
    if ctx.extra.get("frames") is not None and len(ctx.extra["frames"]) >= 4:
        if signal.check_interlacing(ctx).data.get("combed"):
            limiters.append("Interlaced / combed frames: motion and noise cues are distorted")
        if signal.check_temporal_cadence(ctx).data.get("pattern") in ("HEAVY_DUPLICATION", "REGULAR_DUPLICATION", "PULLDOWN_3_2"):
            limiters.append("Duplicated-frame cadence: the true frame rate is lower than the container rate, so motion cues are unreliable")
    head_tables = container.check_isobmff_tables(ctx)
    if head_tables.status == FindingStatus.INFO and any(not t.get("cfr") for t in head_tables.data.get("tracks", [])):
        limiters.append("Variable frame rate: frame-to-frame statistics are not uniformly spaced")
    if assess_transcoding(ctx)["likelihood"] >= 0.6:
        limiters.append("Probable multi-generation re-encoding: forensic traces may be degraded or laundered")
    level = "NORMAL" if not limiters else ("REDUCED" if len(limiters) < 3 else "LOW")
    return Finding(
        check_id="confidence_limiters", dimension="Section21.7", stage="reliability", title="Confidence limiters",
        status=FindingStatus.WARN if limiters else FindingStatus.PASS, severity=Severity.LOW if limiters else Severity.NONE,
        evidence_class=EvidenceClass.RELIABILITY,
        detail=("Reliability " + level + ": " + "; ".join(limiters) + "." if limiters else
                "No known confidence limiters; the detectors are operating in their normal envelope."),
        data={"level": level, "limiters": limiters},
    )
