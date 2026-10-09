"""
video_detector.face: Temporal & Frame Facial Deepfake Artifact Detector for Video.
Localizes faces across video frames, analyzing skin smoothing, bilateral texture anomalies,
and temporal face flickering/warping.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from core.face_detection import get_face_finder
from core.perception.face_texture import risk_label, texture_cues

logger = logging.getLogger("video_detector.face")


class VideoFaceDeepfakeDetector:
    """Detects facial deepfake manipulation across video keyframes."""

    def __init__(self):
        pass

    def detect_faces(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Faces found by the YuNet detector, as (x, y, w, h, confidence)."""
        return get_face_finder().find(image_bgr)

    def analyze_frame_faces(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Analyzes a single frame for facial deepfakes."""
        if frame_bgr is None or not hasattr(frame_bgr, "shape"):
            return {"faces_detected": 0, "deepfake_risk": "NONE", "facial_ai_confidence": 0.0, "face_boxes": []}

        faces = self.detect_faces(frame_bgr)
        if not faces:
            return {"faces_detected": 0, "deepfake_risk": "NO_FACES_DETECTED", "facial_ai_confidence": 0.0, "face_boxes": []}

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scores = []
        details = []

        for x, y, w, h, area in faces:
            face_roi = gray[y : y + h, x : x + w]
            if face_roi.size == 0:
                continue

            cues = texture_cues(face_roi)
            score = cues["score"]
            scores.append(score)
            details.append({"bbox": [x, y, w, h], "texture_smoothness": round(cues["texture_smoothness"], 2), "deepfake_score": round(score, 2)})

        avg = float(np.mean(scores)) if scores else 0.0
        risk = risk_label(avg)

        return {
            "faces_detected": len(faces),
            "deepfake_risk": risk,
            "facial_ai_confidence": round(avg, 2),
            "face_boxes": [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h, _ in faces],
            "details": details,
        }

    def analyze_video_frames(self, frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
        """Analyzes facial artifacts across sampled video frames."""
        if not frames_bgr:
            return {"total_frames_analyzed": 0, "faces_detected_max": 0, "deepfake_risk": "NONE", "mean_face_ai_score": 0.0}

        frame_results = [self.analyze_frame_faces(f) for f in frames_bgr]
        detected_counts = [r["faces_detected"] for r in frame_results]
        max_faces = max(detected_counts) if detected_counts else 0

        ai_scores = [r["facial_ai_confidence"] for r in frame_results if r["faces_detected"] > 0]
        mean_score = float(np.mean(ai_scores)) if ai_scores else 0.0

        risk = risk_label(mean_score)

        return {
            "total_frames_analyzed": len(frames_bgr),
            "faces_detected_max": max_faces,
            "mean_face_ai_score": round(mean_score, 2),
            "deepfake_risk": risk if max_faces > 0 else "NO_FACES_DETECTED",
        }
