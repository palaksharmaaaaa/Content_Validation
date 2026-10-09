"""
video_detector.dimension_checks.signal: signal-level checks.

PHYSICAL_SIGNAL class but informational: interlacing and frame-cadence findings describe how a clip was
captured/converted and how reliable temporal cues are; they do not indicate AI generation and carry no
log-odds. Frames come from ``ctx.extra["frames"]`` (consecutive grayscale arrays) or are read from the file.
"""
from __future__ import annotations

from typing import List

import numpy as np

from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from video_detector.dimension_checks import _common as C

COMB_RATIO = 1.0
MOTION_MIN = 0.3
CADENCE_MIN_FRAMES = 20
REPEAT_FRACTION = 0.15
HEAVY_DUP_FRACTION = 0.40
CUT_FACTOR = 4.0
CUT_MIN_DIFF = 20.0


def _f(check_id, title, status, severity, detail, data=None) -> Finding:
    return Finding(check_id=check_id, dimension="K", stage="signal", title=title, status=status, severity=severity,
                   evidence_class=EvidenceClass.PHYSICAL_SIGNAL, detail=detail, data=data or {})


def _frames(ctx: CheckContext) -> List[np.ndarray]:
    fr = ctx.extra.get("frames")
    if fr is None:
        fr = C.read_consecutive_gray(ctx.path)
        ctx.extra["frames"] = fr
    return fr


def _frame_diffs(frames: List[np.ndarray]) -> np.ndarray:
    return np.array([float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32)))) for a, b in zip(frames[:-1], frames[1:])])


def _container_fiel(ctx: CheckContext):
    try:
        head, _t, _s = C.read_windows(ctx.path)
    except OSError:
        return None
    if C.sniff_video_format(head) not in ("mp4", "mov"):
        return None
    top = C.walk_top_level(ctx.path)
    moov = C.load_moov(ctx.path, top)
    for t in (moov or {}).get("tracks", []):
        if t.get("fiel"):
            return t["fiel"]
    return None


@registry.register("video", "interlacing")
def check_interlacing(ctx: CheckContext) -> Finding:
    title = "Interlacing / combing"
    frames = _frames(ctx)
    if len(frames) < 4:
        return _f("interlacing", title, FindingStatus.NOT_APPLICABLE, Severity.NONE, "Too few decodable frames.")
    diffs = _frame_diffs(frames)
    ratios = []
    for i in range(1, len(frames)):
        if diffs[i - 1] < MOTION_MIN:
            continue
        f = frames[i].astype(np.float32)
        adj = float(np.mean(np.abs(f[1:] - f[:-1])))
        two = float(np.mean(np.abs(f[2:] - f[:-2])))
        ratios.append(adj / (two + 1e-6))
    fiel = _container_fiel(ctx)
    if len(ratios) < 3:
        return _f("interlacing", title, FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  "Not enough motion in the first frames to measure combing.", {"container_fiel": fiel})
    ratio = float(np.median(ratios))
    combed = ratio > COMB_RATIO
    data = {"combing_ratio": round(ratio, 2), "combed": combed, "moving_frames": len(ratios), "container_fiel": fiel}
    interlaced_flag = bool(fiel and fiel.get("count", 1) == 2)
    if combed:
        return _f("interlacing", title, FindingStatus.INFO, Severity.LOW,
                  f"Row-combing on moving frames (ratio {ratio:.2f}): the content is interlaced or was not de-interlaced. "
                  "Combing artifacts distort motion and noise cues.", data)
    if interlaced_flag:
        return _f("interlacing", title, FindingStatus.INFO, Severity.NONE,
                  "The container flags interlaced fields, but no combing was measured (static content or already de-interlaced).", data)
    return _f("interlacing", title, FindingStatus.PASS, Severity.NONE,
              f"No combing measured (ratio {ratio:.2f}); consistent with progressive capture.", data)


@registry.register("video", "temporal_cadence")
def check_temporal_cadence(ctx: CheckContext) -> Finding:
    title = "Frame cadence (duplicates / pulldown)"
    frames = _frames(ctx)
    if len(frames) < CADENCE_MIN_FRAMES:
        return _f("temporal_cadence", title, FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  f"Fewer than {CADENCE_MIN_FRAMES} decodable consecutive frames.")
    d = _frame_diffs(frames)
    med = float(np.percentile(d, 75))  # robust motion level even when many frames are duplicates
    if med < 0.5:
        return _f("temporal_cadence", title, FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  "Too little frame-to-frame change to analyze cadence (static content).")
    repeats = np.where(d < REPEAT_FRACTION * med)[0]
    cuts = np.where((d > CUT_FACTOR * med) & (d > CUT_MIN_DIFF))[0]
    frac = len(repeats) / len(d)
    data = {"repeat_frames": int(len(repeats)), "repeat_fraction": round(frac, 3), "hard_cuts": int(len(cuts)),
            "pattern": "NONE", "period": None}
    if frac >= HEAVY_DUP_FRACTION:
        data["pattern"] = "HEAVY_DUPLICATION"
        return _f("temporal_cadence", title, FindingStatus.INFO, Severity.LOW,
                  f"{frac * 100:.0f}% of frames repeat the previous one: low true frame rate upsampled into a higher-rate container "
                  "(slow capture, frame-rate conversion, or interpolation).", data)
    if len(repeats) >= 3:
        gaps = np.diff(repeats)
        vals, counts = np.unique(gaps, return_counts=True)
        top = int(vals[np.argmax(counts)])
        if counts.max() >= 0.8 * len(gaps):
            data["period"] = top
            if top == 5:
                data["pattern"] = "PULLDOWN_3_2"
                return _f("temporal_cadence", title, FindingStatus.INFO, Severity.LOW,
                          "A frame repeats every 5 frames: 3:2 pulldown cadence (24 -> 30 fps film-to-video conversion). Splices in "
                          "broadcast archives break this cadence.", data)
            data["pattern"] = "REGULAR_DUPLICATION"
            return _f("temporal_cadence", title, FindingStatus.INFO, Severity.LOW,
                      f"A frame repeats every {top} frames: regular frame-rate conversion by duplication.", data)
    if len(repeats) >= 1:
        data["pattern"] = "ISOLATED_DUPLICATES"
        return _f("temporal_cadence", title, FindingStatus.INFO, Severity.LOW,
                  f"{len(repeats)} isolated duplicate frame(s) without a regular period (dropped frames repaired by duplication, "
                  "encoder frame skips, or a pause).", data)
    extra = f" {len(cuts)} hard cut(s) / discontinuities seen." if len(cuts) else ""
    return _f("temporal_cadence", title, FindingStatus.PASS, Severity.NONE,
              "No duplicate-frame cadence detected in the first frames." + extra, data)
