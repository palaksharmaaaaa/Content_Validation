"""
image_detector.dimension_checks.lifecycle: platform re-encode likelihood.

RELIABILITY class: explains why forensic traces may be degraded; never changes P(AI).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from PIL import Image

from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from image_detector.dimension_checks import _common as C

logger = logging.getLogger(__name__)

COMMON_PLATFORM_LONG_EDGES = {720, 960, 1080, 1280, 1440, 1600, 2048}

W_STRIPPED_EXIF = 0.35
W_COMMON_EDGE = 0.25
W_STANDARD_TABLES = 0.25
W_SUBSAMPLED = 0.15


def assess_platform_reencode(ctx: CheckContext) -> Dict[str, Any]:
    cues = []
    score = 0.0
    try:
        with Image.open(ctx.path) as im:
            fmt = (im.format or "").upper()
            w, h = im.size
            exif = im.getexif()
            has_camera_exif = bool(exif.get(271) and exif.get(272))
            tables = dict(getattr(im, "quantization", {}) or {})
            sub = None
            if fmt == "JPEG":
                from PIL import JpegImagePlugin

                try:
                    sub = JpegImagePlugin.get_sampling(im)
                except Exception:
                    sub = None
    except Exception as exc:
        logger.debug("assess_platform_reencode: ignored %s: %s", type(exc).__name__, exc)
        return {"likelihood": 0.0, "level": "UNKNOWN", "cues": []}
    w = int(ctx.profile.get("width") or w)
    h = int(ctx.profile.get("height") or h)
    if fmt == "JPEG":
        if not has_camera_exif:
            score += W_STRIPPED_EXIF
            cues.append("camera EXIF stripped")
        if max(w, h) in COMMON_PLATFORM_LONG_EDGES:
            score += W_COMMON_EDGE
            cues.append(f"long edge {max(w, h)} px matches common platform downscale targets")
        q = C.estimate_libjpeg_quality(tables)
        if q is not None and 40 <= q <= 88:
            score += W_STANDARD_TABLES
            cues.append(f"standard libjpeg tables at quality about {q} (typical server re-encode)")
        if sub == 2:
            score += W_SUBSAMPLED
            cues.append("4:2:0 chroma subsampling")
    likelihood = round(min(1.0, score), 2)
    level = "HIGH" if likelihood >= 0.6 else "MODERATE" if likelihood >= 0.35 else "LOW"
    return {"likelihood": likelihood, "level": level, "cues": cues}


@registry.register("image", "platform_reencode", phase="post")
def check_platform_reencode(ctx: CheckContext) -> Finding:
    res = assess_platform_reencode(ctx)
    high = res["likelihood"] >= 0.6
    return Finding(
        check_id="platform_reencode", dimension="Section22.5", stage="lifecycle",
        title="Platform re-encode likelihood",
        status=FindingStatus.WARN if high else FindingStatus.INFO,
        severity=Severity.LOW if high else Severity.NONE,
        evidence_class=EvidenceClass.RELIABILITY,
        detail=(f"Likelihood of social/messaging re-encode: {res['level']} ({res['likelihood']:.2f}). "
                + ("Cues: " + "; ".join(res["cues"]) + ". Sensor-noise and compression traces may be degraded." if res["cues"] else "No re-encode cues.")),
        data={"likelihood": res["likelihood"], "likelihood_level": res["level"], "cues": res["cues"]},
    )
