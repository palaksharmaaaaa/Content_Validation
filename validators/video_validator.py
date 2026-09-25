"""
Video validator module: Inspects video frames, quality, usability coverage, and AI characteristics.
"""
from typing import Any, Dict
import cv2
import numpy as np

from config.settings import (
    MIN_CONTENT_COVERAGE,
    PARTIAL_CONTENT_COVERAGE,
)
from video.frame_extractor import get_video_metadata, extract_sampled_frames
from video.frame_analyzer import analyze_sampled_frames


def analyze_video(video_path: str, sample_count: int = 30, ai_detector=None) -> Dict[str, Any]:
    """
    Performs comprehensive video analysis: metadata retrieval, frame sampling,
    usability scoring (blank/blurry), and frame-level AI detection.
    """
    metadata = get_video_metadata(video_path)
    if metadata is None:
        return {
            "valid": False,
            "content_status": "INVALID",
            "error": "Unable to read video metadata.",
        }

    duration = float(metadata.get("duration_seconds", 0.0))
    total_frames = int(metadata.get("frame_count", 0))

    frames = extract_sampled_frames(video_path, num_frames=sample_count)
    if not frames:
        return {
            "valid": False,
            "content_status": "INVALID",
            "metadata": metadata,
            "error": "No readable frames found.",
        }

    analysis = analyze_sampled_frames(
        frames=frames,
        duration_seconds=duration,
        total_video_frames=total_frames,
        ai_detector=ai_detector,
    )

    usable_pct = analysis["usable_percentage"]
    blurry_pct = analysis["blurry_percentage"]
    blank_pct = analysis["blank_percentage"]
    coverage = analysis["content_coverage"]

    # Compute approximate durations
    usable_duration = duration * (usable_pct / 100.0)
    blurry_duration = duration * (blurry_pct / 100.0)
    blank_duration = duration * (blank_pct / 100.0)
    unusable_duration = blurry_duration + blank_duration

    if coverage >= MIN_CONTENT_COVERAGE:
        content_status = "VALID"
    elif coverage >= PARTIAL_CONTENT_COVERAGE:
        content_status = "PARTIALLY_VALID"
    else:
        content_status = "INVALID"

    return {
        "valid": content_status != "INVALID",
        "content_status": content_status,
        "metadata": metadata,
        "frames_analyzed": len(frames),
        "usable_frames": analysis["usable_frames"],
        "valid_frames": analysis["usable_frames"],
        "blank_frames": analysis["blank_frames"],
        "blurry_frames": analysis["blurry_frames"],
        "content_coverage": coverage,
        "usable_percentage": usable_pct,
        "blurry_percentage": blurry_pct,
        "blank_percentage": blank_pct,
        "unusable_percentage": round(blurry_pct + blank_pct, 2),
        "usable_duration_seconds": round(usable_duration, 2),
        "blurry_duration_seconds": round(blurry_duration, 2),
        "blank_duration_seconds": round(blank_duration, 2),
        "unusable_duration_seconds": round(unusable_duration, 2),
        "ai_video_rating": analysis["ai_video_rating"],
        "frame_details": analysis["analyzed_frames"],
    }