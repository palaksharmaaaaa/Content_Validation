"""
Frame-level analysis, temporal segment grouping, and inter-frame warping analysis
for video validation and AI detection.
"""
from __future__ import annotations

from typing import Any, Dict, List
import cv2
import numpy as np

from schemas.result_schema import ModalityScore
from validators.quality_validator import evaluate_visual_quality


def group_temporal_segments(analyzed_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups consecutive frames with the same classification status into contiguous temporal intervals.
    e.g. [00:00 - 00:12: REAL], [00:12 - 00:28: AI-GENERATED]
    """
    if not analyzed_frames:
        return []

    segments = []
    current_label = analyzed_frames[0].get("ai_prediction", "UNDECIDED")
    start_time = analyzed_frames[0].get("timestamp_seconds", 0.0)
    end_time = start_time

    for f in analyzed_frames[1:]:
        lbl = f.get("ai_prediction", "UNDECIDED")
        t = f.get("timestamp_seconds", end_time)
        if lbl == current_label:
            end_time = t
        else:
            segments.append({
                "start_seconds": round(start_time, 2),
                "end_seconds": round(end_time, 2),
                "duration_seconds": round(max(0.0, end_time - start_time), 2),
                "label": current_label,
            })
            current_label = lbl
            start_time = t
            end_time = t

    segments.append({
        "start_seconds": round(start_time, 2),
        "end_seconds": round(end_time, 2),
        "duration_seconds": round(max(0.0, end_time - start_time), 2),
        "label": current_label,
    })
    return segments


def compute_temporal_consistency(frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
    """
    Measures frame-to-frame temporal stability vs generative boiling/warping.
    Real video cameras exhibit smooth, physically consistent optical motion.
    AI video models (Sora, Kling, Runway, Pika) exhibit high variance in inter-frame high-frequency residuals.
    """
    if len(frames_bgr) < 2:
        return {"temporal_warping_risk": "INSUFFICIENT_FRAMES", "flicker_score": 0.0, "p_temporal_ai": 0.5}

    diffs = []
    for i in range(len(frames_bgr) - 1):
        f1 = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(frames_bgr[i + 1], cv2.COLOR_BGR2GRAY)
        # Compute inter-frame delta
        delta = cv2.absdiff(f1, f2)
        diffs.append(float(np.mean(delta)))

    diff_arr = np.array(diffs)
    mean_motion = float(np.mean(diff_arr))
    motion_var = float(np.var(diff_arr))

    # Generative video often exhibits abnormal inter-frame flicker or sudden textural shifts
    if motion_var > 45.0 or (mean_motion > 25.0 and motion_var > 30.0):
        risk = "HIGH_WARPING_DETECTED"
        p_temporal_ai = 0.85
    elif motion_var > 20.0:
        risk = "SUSPICIOUS_FLICKER"
        p_temporal_ai = 0.65
    else:
        risk = "SMOOTH_PHYSICAL_MOTION"
        p_temporal_ai = 0.25

    return {
        "temporal_warping_risk": risk,
        "mean_motion_delta": round(mean_motion, 2),
        "temporal_variance": round(motion_var, 2),
        "p_temporal_ai": p_temporal_ai,
    }


def analyze_sampled_frames(
    frames: List[Dict[str, Any]],
    duration_seconds: float,
    total_video_frames: int,
    ai_detector=None,
    sensitivity: str = "high",
) -> Dict[str, Any]:
    """
    Analyzes sampled video frames for visual usability, temporal segmentation,
    inter-frame warping, and frame-level AI detection.
    """
    total_sampled = len(frames)
    if total_sampled == 0:
        return {
            "usable_frames": 0,
            "blank_frames": 0,
            "blurry_frames": 0,
            "usable_percentage": 0.0,
            "blurry_percentage": 0.0,
            "blank_percentage": 0.0,
            "content_coverage": 0.0,
            "ai_video_rating": ModalityScore().to_dict(),
            "temporal_consistency": {},
            "temporal_segments": [],
            "analyzed_frames": [],
        }

    analyzed_frames = []
    blank_count = 0
    blurry_count = 0
    usable_count = 0

    ai_frame_count = 0
    real_frame_count = 0
    undecided_frame_count = 0

    raw_frames = [item["frame"] for item in frames]
    temporal_consistency = compute_temporal_consistency(raw_frames)

    for item in frames:
        frame_bgr = item["frame"]
        frame_idx = item["index"]

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        quality = evaluate_visual_quality(gray, w, h)

        if quality["is_blank"]:
            blank_count += 1
        elif quality["is_blurry"]:
            blurry_count += 1
        else:
            usable_count += 1

        timestamp = 0.0
        if duration_seconds > 0 and total_video_frames > 1:
            timestamp = (frame_idx / (total_video_frames - 1)) * duration_seconds

        frame_result = {
            "frame_index": frame_idx,
            "timestamp_seconds": round(timestamp, 2),
            "is_blank": quality["is_blank"],
            "is_blurry": quality["is_blurry"],
            "blur_score": quality["blur_score"],
            "brightness": quality["brightness"],
        }

        if ai_detector is not None and not quality["is_blank"]:
            pred = ai_detector.predict_frame(frame_bgr, sensitivity=sensitivity)
            lbl = pred.get("label", "UNDECIDED")
            frame_result["ai_prediction"] = lbl
            frame_result["ai_prob"] = pred.get("ai_prob", 0.5)

            if lbl == "LIKELY AI-GENERATED":
                ai_frame_count += 1
            elif lbl == "LIKELY REAL":
                real_frame_count += 1
            else:
                undecided_frame_count += 1
        else:
            frame_result["ai_prediction"] = "UNDECIDED"
            frame_result["ai_prob"] = 0.5
            undecided_frame_count += 1

        analyzed_frames.append(frame_result)

    usable_pct = (usable_count / total_sampled) * 100.0
    blurry_pct = (blurry_count / total_sampled) * 100.0
    blank_pct = (blank_count / total_sampled) * 100.0
    coverage = round(usable_count / total_sampled, 4)

    # Combine frame classification with inter-frame temporal consistency
    frame_ai_ratio = ai_frame_count / float(total_sampled)
    temporal_ai_weight = temporal_consistency.get("p_temporal_ai", 0.5)
    fused_ai_prob = (frame_ai_ratio * 0.70) + (temporal_ai_weight * 0.30)
    fused_real_prob = ((real_frame_count / float(total_sampled)) * 0.70) + ((1.0 - temporal_ai_weight) * 0.30)

    gap = abs(fused_ai_prob - fused_real_prob)
    undecided_pct = max(3.0, (1.0 - gap) * 20.0)
    rem = 100.0 - undecided_pct
    total_p = fused_ai_prob + fused_real_prob if (fused_ai_prob + fused_real_prob) > 0 else 1.0

    video_ai_pct = round((fused_ai_prob / total_p) * rem, 1)
    video_real_pct = round((fused_real_prob / total_p) * rem, 1)
    video_undecided_pct = round(100.0 - (video_ai_pct + video_real_pct), 1)

    threshold = 50.0 if sensitivity == "high" else 58.0
    if video_ai_pct >= threshold:
        label = "LIKELY AI-GENERATED"
    elif video_real_pct >= 58.0:
        label = "LIKELY REAL"
    else:
        label = "UNDECIDED"

    temporal_segments = group_temporal_segments(analyzed_frames)
    ai_segment_seconds = sum(s["duration_seconds"] for s in temporal_segments if s["label"] == "LIKELY AI-GENERATED")
    ai_duration_pct = (ai_segment_seconds / max(0.1, duration_seconds)) * 100.0 if duration_seconds > 0 else video_ai_pct

    video_rating = ModalityScore(
        ai_percentage=video_ai_pct,
        real_percentage=video_real_pct,
        undecided_percentage=video_undecided_pct,
        confidence=round(max(fused_ai_prob, fused_real_prob), 2),
        label=label,
        details={
            "frames_analyzed": total_sampled,
            "ai_frames": ai_frame_count,
            "real_frames": real_frame_count,
            "undecided_frames": undecided_frame_count,
            "ai_duration_pct": round(ai_duration_pct, 1),
            "ai_duration_seconds": round(ai_segment_seconds, 2),
            "temporal_consistency": temporal_consistency,
            "temporal_segments": temporal_segments,
        },
    )

    return {
        "usable_frames": usable_count,
        "blank_frames": blank_count,
        "blurry_frames": blurry_count,
        "usable_percentage": round(usable_pct, 1),
        "blurry_percentage": round(blurry_pct, 1),
        "blank_percentage": round(blank_pct, 1),
        "content_coverage": coverage,
        "ai_video_rating": video_rating.to_dict(),
        "temporal_consistency": temporal_consistency,
        "temporal_segments": temporal_segments,
        "analyzed_frames": analyzed_frames,
    }
