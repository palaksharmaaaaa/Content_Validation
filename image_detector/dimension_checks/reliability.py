"""
image_detector.dimension_checks.reliability: confidence limiters.

Explains conditions under which the physical-signal detectors are less trustworthy. RELIABILITY
class: informational, never alters P(AI).
"""
from __future__ import annotations

from typing import List

from PIL import Image

from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from image_detector.dimension_checks import _common as C
from image_detector.dimension_checks.lifecycle import assess_platform_reencode

LOW_RES_MIN_SIDE = 256
HEAVY_COMPRESSION_QUALITY = 70


@registry.register("image", "confidence_limiters", phase="post")
def check_confidence_limiters(ctx: CheckContext) -> Finding:
    limiters: List[str] = []
    try:
        with Image.open(ctx.path) as im:
            w, h = im.size
            tables = dict(getattr(im, "quantization", {}) or {})
    except Exception:
        w = h = 0
        tables = {}
    w = int(ctx.profile.get("width") or w)
    h = int(ctx.profile.get("height") or h)
    if w and h and min(w, h) < LOW_RES_MIN_SIDE:
        limiters.append(f"Low resolution ({min(w, h)} px shortest side): pixel-level cues are unreliable")
    q = C.estimate_libjpeg_quality(tables) if tables else None
    if q is not None and q < HEAVY_COMPRESSION_QUALITY:
        limiters.append(f"Heavy JPEG compression (quality about {q}): sensor noise and spectral cues are suppressed")
    reenc = assess_platform_reencode(ctx)
    if reenc["likelihood"] >= 0.6:
        limiters.append("Probable platform re-encode: provenance and forensic traces may have been laundered or degraded")
    if ctx.ai_result.get("screenshot_detected"):
        limiters.append("Screenshot: camera sensor physics do not apply")
    level = "NORMAL" if not limiters else ("REDUCED" if len(limiters) < 3 else "LOW")
    return Finding(
        check_id="confidence_limiters", dimension="Section22.6", stage="reliability", title="Confidence limiters",
        status=FindingStatus.WARN if limiters else FindingStatus.PASS,
        severity=Severity.LOW if limiters else Severity.NONE,
        evidence_class=EvidenceClass.RELIABILITY,
        detail=("Reliability " + level + ": " + "; ".join(limiters) + "." if limiters else
                "No known confidence limiters; the detectors are operating in their normal envelope."),
        data={"level": level, "limiters": limiters},
    )
