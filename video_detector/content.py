"""
video_detector.content: Video Scene, Entity, Environment & Dynamic Lighting Intelligence.
Analyzes keyframe content, detected objects, vehicles, humans, and environmental settings.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from video_detector.face import VideoFaceDeepfakeDetector

logger = logging.getLogger("video_detector.content")


class VideoContentAnalyzer:
    """Scene and content intelligence analyzer for video frames."""

    def __init__(self, image_content_analyzer: Optional[Any] = None):
        """``image_content_analyzer``: any object with ``analyze_image_content(ndarray)``. When supplied (the services
        layer injects the image package's analyzer) the full scene inventory of a representative keyframe is used;
        otherwise a light built-in heuristic keeps this package self-contained."""
        self.face_detector = VideoFaceDeepfakeDetector()
        self.image_content_analyzer = image_content_analyzer

    @staticmethod
    def representative_frame(frames_bgr: List[np.ndarray]) -> np.ndarray:
        """The keyframe ~15% into the sampled timeline (past fade-ins, before late scene changes)."""
        return frames_bgr[min(len(frames_bgr) - 1, int(len(frames_bgr) * 0.15))]

    def analyze_video_frames(self, frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
        """Runs content, environmental, and scene analysis across sampled video keyframes."""
        if not frames_bgr:
            return {
                "living_entities": {"humans": {"count": 0, "faces": 0}},
                "environment": {"setting": "Unknown"},
                "lighting_and_daytime": {"daytime": "Unknown"},
                "purpose_and_depiction": {"primary_genre": "Unknown"},
            }
        if self.image_content_analyzer is not None:
            result = self.image_content_analyzer.analyze_image_content(self.representative_frame(frames_bgr))
            if isinstance(result.get("minors"), dict):
                # One frame cannot clear a video: until the video-wide check exists, a video is never reported as free of minors.
                result["minors"].update(scope="one_frame_of_video", review_required=True)
            return result

        # 1. Face analysis across frames
        face_info = self.face_detector.analyze_video_frames(frames_bgr)
        human_count = face_info.get("faces_detected_max", 0)

        # 2. Lighting & Environment analysis on middle keyframe
        mid_frame = frames_bgr[len(frames_bgr) // 2]
        gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))

        daytime = "Night / Low-Light" if mean_lum < 55.0 else ("Daylight" if mean_lum < 200.0 else "Overexposed Daylight")

        # Environmental setting heuristics
        hsv = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2HSV)
        green_ratio = float(np.sum(cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)) > 0)) / float(mid_frame.shape[0] * mid_frame.shape[1])
        blue_ratio = float(np.sum(cv2.inRange(hsv, (90, 40, 40), (130, 255, 255)) > 0)) / float(mid_frame.shape[0] * mid_frame.shape[1])

        if green_ratio > 0.18:
            setting = "Outdoor / Nature"
        elif blue_ratio > 0.22:
            setting = "Outdoor / Open Sky or Coast"
        else:
            setting = "Indoor or Studio Scene"

        genre = "Portrait / Presentation" if human_count == 1 else ("Group / Social Scene" if human_count > 1 else "Cinematic / General Scene")

        return {
            "living_entities": {
                "humans": {
                    "count": human_count,
                    "faces": human_count,
                    "deepfake_analysis": face_info,
                },
            },
            "environment": {
                "setting": setting,
                "vegetation_ratio": round(green_ratio, 3),
                "sky_ratio": round(blue_ratio, 3),
            },
            "lighting_and_daytime": {
                "daytime": daytime,
                "mean_luminance": round(mean_lum, 1),
            },
            "purpose_and_depiction": {
                "primary_genre": genre,
            },
        }
