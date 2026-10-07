"""core.perception.age_video: minor screening across the sampled frames of a video.

A video is only cleared if every examined frame is clear: one frame with a possible minor sends the whole video to review
(``review_required``), exactly as for a still image. The result also says how often it happened, so a reviewer can tell a
child who is on screen throughout from one flicker:

  * priority REPEATED      flagged in at least two frames and in at least a quarter of the examined frames
  * priority SINGLE_FRAME  flagged in one frame, or in under a quarter of them
  * priority NONE          nothing flagged in any examined frame

Two entry points: ``screen_video_file`` (adaptive sampling straight from the file, preferred) and ``screen_video_frames`` (a list of
frames the caller already has, evenly spread).

Only the sampled frames are examined; a child who appears and leaves between two samples is not seen. The result states how
many frames it covered so the gap is visible, and ``max_frames`` trades time (about a second a frame on CPU) for coverage.
No tracking or identity matching is done: the same child in ten frames counts as ten sightings.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.perception.age import AgeEstimator, get_age_estimator

DEFAULT_MAX_FRAMES = 16
REPEATED_MIN_FRAMES = 2
REPEATED_MIN_SHARE = 0.25
ADAPTIVE_MAX_FRAMES = 60


def pick_frames(n_available: int, max_frames: int) -> List[int]:
    """Indices of up to ``max_frames`` frames spread evenly over ``n_available``, always including the first and last."""
    if n_available <= 0 or max_frames <= 0:
        return []
    if n_available <= max_frames:
        return list(range(n_available))
    return sorted({int(round(i * (n_available - 1) / (max_frames - 1))) for i in range(max_frames)}) if max_frames > 1 else [0]


def screen_video_file(path, estimator: Optional[AgeEstimator] = None, max_frames: int = ADAPTIVE_MAX_FRAMES) -> Dict[str, Any]:
    """Minor screening of a video file with adaptive sampling (core.perception.video_sampling): the frames after every content
    change plus a grid of at most one a second, a frame that looks like the last examined one reusing its result. Same result
    shape as ``screen_video_frames`` plus ``sampling``: how many frames were examined, how many were reused, and the longest
    stretch of the video that no examined frame covers."""
    from core.perception import video_sampling as vs

    est = estimator or get_age_estimator()
    probes = vs.probe_video(path)
    times, info = vs.choose_times(probes, max_frames)
    frames = vs.read_frames(path, times)
    if not frames:
        return {**_empty_result(0), "status": "NO_FRAMES", "review_required": True, "reason": "the video could not be read, so nothing was checked"}

    flagged, ages, statuses, model = [], [], [], None
    last_thumb, last = None, None
    assessed = reused = 0
    for t, frame in frames:
        thumb = vs.thumbnail(frame)
        if last is not None and vs.thumb_diff(thumb, last_thumb) < vs.REUSE_DIFF:
            r, reused = last, reused + 1
        else:
            r, assessed = est.assess(frame), assessed + 1
            last, last_thumb = r, thumb
        model = r.get("model", model)
        statuses.append(r["status"])
        if r["youngest_age"] is not None:
            ages.append(r["youngest_age"])
        if r["review_required"]:
            flagged.append({"frame": None, "time": round(t, 2), "youngest_age": r["youngest_age"], "likely_minor": r["contains_minor"],
                            "assessments": sorted({sub["assessment"] for sub in r["subjects"] if sub["assessment"] != "ADULT"}) or [r["status"]]})
    status = "UNAVAILABLE" if "UNAVAILABLE" in statuses else "OK"
    n_flagged = len(flagged)
    priority = _priority(n_flagged, len(frames))
    sampling = {"duration_seconds": round(probes.duration, 1), "frames_examined": len(frames), "frames_assessed": assessed, "frames_reused": reused,
                "content_changes": info["changes"], "grid_seconds": info["grid_seconds"],
                "longest_unexamined_seconds": vs.longest_gap([t for t, _ in frames], probes.duration)}
    return {**_empty_result(len(frames)), "scope": "video_adaptive_frames", "status": status, "model": model, "frames_available": len(frames),
            "frames_flagged": n_flagged, "priority": priority, "contains_minor": any(f["likely_minor"] for f in flagged),
            "contains_possible_minor": n_flagged > 0, "review_required": n_flagged > 0 or status == "UNAVAILABLE",
            "youngest_age": min(ages) if ages else None, "flagged_frames": flagged, "sampling": sampling}


def _priority(n_flagged: int, n_examined: int) -> str:
    if n_flagged == 0:
        return "NONE"
    if n_flagged >= REPEATED_MIN_FRAMES and n_flagged / max(1, n_examined) >= REPEATED_MIN_SHARE:
        return "REPEATED"
    return "SINGLE_FRAME"


def _empty_result(n_examined: int) -> Dict[str, Any]:
    return {"scope": "video_sampled_frames", "frames_available": n_examined, "frames_examined": n_examined, "frames_flagged": 0, "priority": "NONE",
            "contains_minor": False, "contains_possible_minor": False, "review_required": False, "youngest_age": None, "flagged_frames": [], "model": None}


def screen_video_frames(frames_bgr: Sequence[np.ndarray], timestamps: Optional[Sequence[float]] = None,
                        estimator: Optional[AgeEstimator] = None, max_frames: int = DEFAULT_MAX_FRAMES) -> Dict[str, Any]:
    """Minor screening for a list of video frames (BGR arrays); see the module doc for the result's meaning."""
    est = estimator or get_age_estimator()
    picks = pick_frames(len(frames_bgr), max_frames)
    base: Dict[str, Any] = {"scope": "video_sampled_frames", "frames_available": len(frames_bgr), "frames_examined": len(picks),
                            "frames_flagged": 0, "priority": "NONE", "contains_minor": False, "contains_possible_minor": False,
                            "review_required": False, "youngest_age": None, "flagged_frames": [], "model": None}
    if not picks:
        return {**base, "status": "NO_FRAMES", "review_required": True, "reason": "no frames could be read, so nothing was checked"}

    flagged, ages, statuses, model = [], [], [], None
    for i in picks:
        r = est.assess(frames_bgr[i])
        model = r.get("model", model)
        statuses.append(r["status"])
        if r["youngest_age"] is not None:
            ages.append(r["youngest_age"])
        if r["review_required"]:
            flagged.append({"frame": i, "time": None if timestamps is None or i >= len(timestamps) else timestamps[i],
                            "youngest_age": r["youngest_age"], "likely_minor": r["contains_minor"],
                            "assessments": sorted({s["assessment"] for s in r["subjects"] if s["assessment"] != "ADULT"}) or [r["status"]]})
    status = "UNAVAILABLE" if "UNAVAILABLE" in statuses else "OK"
    n_flagged = len(flagged)
    if n_flagged == 0:
        priority = "NONE"
    elif n_flagged >= REPEATED_MIN_FRAMES and n_flagged / len(picks) >= REPEATED_MIN_SHARE:
        priority = "REPEATED"
    else:
        priority = "SINGLE_FRAME"
    return {**base, "status": status, "model": model, "frames_flagged": n_flagged, "priority": priority,
            "contains_minor": any(f["likely_minor"] for f in flagged), "contains_possible_minor": n_flagged > 0,
            "review_required": n_flagged > 0 or status == "UNAVAILABLE",
            "youngest_age": min(ages) if ages else None, "flagged_frames": flagged}
