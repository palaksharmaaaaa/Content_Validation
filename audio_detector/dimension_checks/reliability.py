"""
audio_detector.dimension_checks.reliability: confidence limiters (report Section 21.7).
RELIABILITY class: informational, never alters P(AI).
"""
from __future__ import annotations

from typing import List

from audio_detector.dimension_checks import signal
from audio_detector.dimension_checks.lifecycle import _duration, assess_transcoding
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity

SHORT_SECONDS = 3.0
CLIPPING_RATIO = 0.01


@registry.register("audio", "confidence_limiters", phase="post")
def check_confidence_limiters(ctx: CheckContext) -> Finding:
    limiters: List[str] = []
    dur = _duration(ctx)
    if 0 < dur < SHORT_SECONDS:
        limiters.append(f"Very short recording ({dur:.1f} s): temporal and spectral statistics are unstable")
    if ctx.extra.get("samples") is not None:
        tel = signal.check_telephony_channel(ctx).data
        if tel.get("narrowband_telephony"):
            limiters.append("Narrowband telephony channel: cues above 4 kHz are absent, so spoof-detection reliability is reduced")
        elif tel.get("band_limited_4k"):
            limiters.append("Energy confined below about 4 kHz: high-frequency vocoder cues are absent")
        loud = signal.check_loudness_dynamics(ctx).data
        if loud.get("clipping_ratio", 0.0) > CLIPPING_RATIO:
            limiters.append(f"Heavy clipping ({loud['clipping_ratio'] * 100:.1f}% of samples): spectral cues are distorted")
    if assess_transcoding(ctx)["likelihood"] >= 0.6:
        limiters.append("Probable multi-generation lossy transcoding: forensic traces may be degraded or laundered")
    level = "NORMAL" if not limiters else ("REDUCED" if len(limiters) < 3 else "LOW")
    return Finding(
        check_id="confidence_limiters", dimension="Section21.7", stage="reliability", title="Confidence limiters",
        status=FindingStatus.WARN if limiters else FindingStatus.PASS, severity=Severity.LOW if limiters else Severity.NONE,
        evidence_class=EvidenceClass.RELIABILITY,
        detail=("Reliability " + level + ": " + "; ".join(limiters) + "." if limiters else
                "No known confidence limiters; the detectors are operating in their normal envelope."),
        data={"level": level, "limiters": limiters},
    )
