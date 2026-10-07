"""core.perception.age_video: minor screening across the sampled frames of a video.

A video is only cleared if every examined frame is clear: one frame with a possible minor sends the whole video to review
(``review_required``), exactly as for a still image. The result also says how often it happened, so a reviewer can tell a
child who is on screen throughout from one flicker:

  * priority REPEATED      flagged in at least two frames and in at least a quarter of the examined frames
  * priority SINGLE_FRAME  flagged in one frame, or in under a quarter of them
  * priority NONE          nothing flagged in any examined frame

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


def pick_frames(n_available: int, max_frames: int) -> List[int]:
    """Indices of up to ``max_frames`` frames spread evenly over ``n_available``, always including the first and last."""
    if n_available <= 0 or max_frames <= 0:
        return []
    if n_available <= max_frames:
        return list(range(n_available))
    return sorted({int(round(i * (n_available - 1) / (max_frames - 1))) for i in range(max_frames)}) if max_frames > 1 else [0]


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
