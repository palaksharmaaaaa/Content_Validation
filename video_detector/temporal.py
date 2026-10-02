"""
video_detector.temporal: Temporal consistency, motion variance, and diffusion flicker forensics.
Analyzes:
1. Inter-frame optical motion delta normalized by temporal stride sqrt(step).
2. Boundary tearing & geometric morphing between consecutive frames.
3. Neural diffusion flickering (luminance/color high-frequency jitter).
4. Contiguous temporal segment grouping with exact timestamp intervals.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_detector.config import (
    MOTION_VAR_HIGH_WARPING,
    MOTION_VAR_SUSPICIOUS_FLICKER,
    MOTION_VAR_UNNATURAL_FREEZE,
)


def compute_interframe_motion_variance(
    frames: List[np.ndarray],
    temporal_step: int = 1,
    motion_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Computes inter-frame motion delta variance.
    Normalized by sqrt(step) so different video frame rates and sampling intervals remain consistent.
    Natural video features smooth, continuous optical flow. Generative AI video often exhibits
    staccato warping, sudden object appearance/disappearance, or unnatural stillness.
    """
    if len(frames) < 2:
        return {
            "mean_motion_delta": 0.0,
            "motion_variance": 0.0,
            "temporal_warping_risk": "LOW",
            "is_anomalous_motion": False,
        }

    deltas: List[float] = []
    norm_factor = math.sqrt(max(1, temporal_step))

    prev_gray = None
    for frame in frames:
        # Check if frame is blank
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(np.var(gray)) < 5.0:
            continue

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            raw_delta = float(np.mean(diff))
            normalized_delta = raw_delta / norm_factor
            deltas.append(normalized_delta)
        prev_gray = gray

    if not deltas:
        return {
            "mean_motion_delta": 0.0,
            "motion_variance": 0.0,
            "temporal_warping_risk": "LOW",
            "is_anomalous_motion": False,
        }

    mean_delta = float(np.mean(deltas))
    var_delta = float(np.var(deltas))

    # Warping risk classification. Variance cutoffs come from motion_thresholds when the
    # caller supplies calibration data (VideoSelfImprover.load_calibration()'s
    # "motion_thresholds" block, adjusted over time by record_feedback); otherwise fall
    # back to the same defaults as config.py's MOTION_VAR_* constants.
    th = motion_thresholds or {}
    high_warping_var = float(th.get("high_warping_var", MOTION_VAR_HIGH_WARPING))
    suspicious_flicker_var = float(th.get("suspicious_flicker_var", MOTION_VAR_SUSPICIOUS_FLICKER))
    unnatural_freeze_var = float(th.get("unnatural_freeze_var", MOTION_VAR_UNNATURAL_FREEZE))

    if var_delta > high_warping_var or (mean_delta > 32.0 and var_delta > high_warping_var * (4.0 / 7.0)):
        warping_risk = "HIGH_WARPING_DETECTED"
        is_anomalous = True
    elif var_delta > suspicious_flicker_var or mean_delta > 22.0:
        warping_risk = "SUSPICIOUS_FLICKER"
        is_anomalous = True
    elif var_delta < unnatural_freeze_var and mean_delta < 1.2:
        warping_risk = "UNNATURAL_FREEZE"
        is_anomalous = True
    else:
        warping_risk = "LOW"
        is_anomalous = False

    return {
        "mean_motion_delta": round(mean_delta, 2),
        "motion_variance": round(var_delta, 2),
        "temporal_warping_risk": warping_risk,
        "is_anomalous_motion": is_anomalous,
    }


def detect_diffusion_flickering(frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    Examines high-frequency frame-to-frame luminance and high-frequency edge shifts.
    Generative video architectures (e.g. DiT) often produce high-frequency texture flickering
    where details shimmer unnaturally across frames.
    """
    if len(frames) < 3:
        return {"flicker_score": 0.0, "has_diffusion_flicker": False}

    luminances = [float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) for f in frames]
    lum_diffs = np.abs(np.diff(luminances))

    # Second derivative of luminance (direction reversals = flicker)
    sign_changes = 0
    diff_signs = np.sign(np.diff(luminances))
    for i in range(len(diff_signs) - 1):
        if diff_signs[i] != 0 and diff_signs[i+1] != 0 and diff_signs[i] != diff_signs[i+1]:
            sign_changes += 1

    flicker_ratio = sign_changes / max(1, len(diff_signs) - 1)
    mean_lum_jump = float(np.mean(lum_diffs)) if len(lum_diffs) > 0 else 0.0

    has_flicker = bool(flicker_ratio > 0.65 and mean_lum_jump > 3.5)
    flicker_score = round(float(flicker_ratio * min(1.0, mean_lum_jump / 10.0)), 2)

    return {
        "flicker_score": flicker_score,
        "flicker_ratio": round(flicker_ratio, 2),
        "mean_lum_jump": round(mean_lum_jump, 2),
        "has_diffusion_flicker": has_flicker,
    }


def group_temporal_segments(
    analyzed_frames: List[Dict[str, Any]], duration_seconds: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Groups contiguous sequence of frames with the same classification status
    into temporal intervals [start_seconds - end_seconds].
    Guarantees the last segment spans to total duration.
    """
    if not analyzed_frames:
        return []

    segments: List[Dict[str, Any]] = []
    curr_label = analyzed_frames[0]["label"]
    start_time = float(analyzed_frames[0].get("timestamp", 0.0))
    count = 1

    for i in range(1, len(analyzed_frames)):
        frame_info = analyzed_frames[i]
        label = frame_info["label"]
        t = float(frame_info.get("timestamp", 0.0))

        if label != curr_label:
            segments.append({
                "start_seconds": round(start_time, 2),
                "end_seconds": round(t, 2),
                "duration_seconds": round(max(0.01, t - start_time), 2),
                "label": curr_label,
                "frame_count": count,
            })
            curr_label = label
            start_time = t
            count = 1
        else:
            count += 1

    final_end = max(duration_seconds, float(analyzed_frames[-1].get("timestamp", start_time)))
    segments.append({
        "start_seconds": round(start_time, 2),
        "end_seconds": round(final_end, 2),
        "duration_seconds": round(max(0.01, final_end - start_time), 2),
        "label": curr_label,
        "frame_count": count,
    })

    return segments
