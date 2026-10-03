"""
audio_detector.dimension_checks.lifecycle: transcoding-cascade likelihood (report Section 21.5).
RELIABILITY class: explains why forensic traces may be degraded; never changes P(AI).
"""
from __future__ import annotations

from typing import Any, Dict

from audio_detector.dimension_checks import _common as C
from audio_detector.dimension_checks import container, signal
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity

LOSSY_FORMATS = {"mp3", "aac", "ogg", "m4a"}


def _duration(ctx: CheckContext) -> float:
    s = ctx.extra.get("samples")
    if s is not None and len(s) >= 3 and s[2]:
        return float(s[2])
    return float(ctx.profile.get("duration_seconds") or 0.0)


def assess_transcoding(ctx: CheckContext) -> Dict[str, Any]:
    head, _t, size = C.read_windows(ctx.path)
    fmt = C.sniff_audio_format(head)
    dur = _duration(ctx)
    cues, score = [], 0.0
    if fmt in LOSSY_FORMATS:
        score += 0.30
        cues.append(f"lossy container ({fmt})")
        if dur > 0 and size * 8 / dur / 1000.0 <= 96.0:
            score += 0.25
            cues.append(f"low average bitrate (about {size * 8 / dur / 1000.0:.0f} kbps)")
    if ctx.extra.get("samples") is not None:
        tel = signal.check_telephony_channel(ctx).data
        if tel.get("narrowband_telephony"):
            score += 0.30
            cues.append("narrowband telephony band-limiting")
        elif tel.get("band_limited_4k"):
            score += 0.15
            cues.append("energy confined below about 4 kHz")
    if fmt in ("wav", "flac"):
        fh = signal.check_fake_hires(ctx).data
        if fh.get("verdict") in ("POSSIBLE_LOSSY_ORIGIN", "UPSAMPLED_FROM_LOWER_RATE"):
            score += 0.30
            cues.append("lossless container with a lossy-style or upsampling band-limit")
    if fmt == "mp3":
        lp = container.check_mp3_encoder_tag(ctx).data.get("lowpass_hz")
        if lp and lp <= 16500:
            score += 0.20
            cues.append(f"MP3 encoder lowpass at {lp} Hz")
    likelihood = round(min(1.0, score), 2)
    level = "HIGH" if likelihood >= 0.6 else "MODERATE" if likelihood >= 0.35 else "LOW"
    return {"likelihood": likelihood, "level": level, "cues": cues, "format": fmt}


@registry.register("audio", "transcoding_cascade", phase="post")
def check_transcoding_cascade(ctx: CheckContext) -> Finding:
    res = assess_transcoding(ctx)
    high = res["likelihood"] >= 0.6
    return Finding(
        check_id="transcoding_cascade", dimension="Section21.5", stage="lifecycle", title="Transcoding-cascade likelihood",
        status=FindingStatus.WARN if high else FindingStatus.INFO, severity=Severity.LOW if high else Severity.NONE,
        evidence_class=EvidenceClass.RELIABILITY,
        detail=(f"Likelihood of multi-generation lossy transcoding: {res['level']} ({res['likelihood']:.2f}). "
                + ("Cues: " + "; ".join(res["cues"]) + ". Vocoder, flatness and silence cues may be degraded." if res["cues"] else "No transcoding cues.")),
        data={"likelihood": res["likelihood"], "likelihood_level": res["level"], "cues": res["cues"]},
    )
