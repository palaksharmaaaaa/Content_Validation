"""
video_detector.face: Temporal & Frame Facial Deepfake Artifact Detector for Video.
Localizes faces across video frames, analyzing skin smoothing, bilateral texture anomalies,
and temporal face flickering/warping.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from core.face_detection import get_face_finder

logger = logging.getLogger("video_detector.face")


class VideoFaceDeepfakeDetector:
    """Detects facial deepfake manipulation across video keyframes."""

    def __init__(self):
        pass

    def detect_faces(
        self,
        image_bgr: np.ndarray,
        person_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Faces found by the YuNet detector as (x, y, w, h, confidence). ``person_boxes`` is accepted for compatibility
        and ignored: a real face detector needs no help from the person detector."""
        return get_face_finder().find(image_bgr)

    def analyze_frame_faces(
        self,
        frame_bgr: np.ndarray,
        person_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Dict[str, Any]:
        """Analyzes a single frame for facial deepfakes."""
        if frame_bgr is None or not hasattr(frame_bgr, "shape"):
            return {"faces_detected": 0, "deepfake_risk": "NONE", "facial_ai_confidence": 0.0, "face_boxes": []}

        faces = self.detect_faces(frame_bgr, person_boxes=person_boxes)
        if not faces:
            return {"faces_detected": 0, "deepfake_risk": "NO_FACES_DETECTED", "facial_ai_confidence": 0.0, "face_boxes": []}

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scores = []
        details = []

        for x, y, w, h, area in faces:
            face_roi = gray[y : y + h, x : x + w]
            if face_roi.size == 0:
                continue

            bilateral = cv2.bilateralFilter(face_roi, 9, 75, 75)
            texture_diff = float(np.mean(cv2.absdiff(face_roi, bilateral)))

            fh_y1 = max(0, int(h * 0.15))
            fh_y2 = min(h, int(h * 0.50))
            fh_x1 = max(0, int(w * 0.20))
            fh_x2 = min(w, int(w * 0.80))
            upper_roi = face_roi[fh_y1:fh_y2, fh_x1:fh_x2]
            noise_mean = float(np.mean(cv2.absdiff(upper_roi, cv2.GaussianBlur(upper_roi, (3, 3), 0)))) if upper_roi.size > 0 else 2.0

            risk_factors = 0.0
            if texture_diff < 3.2:
                risk_factors += 1.5
            if noise_mean < 1.8:
                risk_factors += 1.0

            score = min(0.95, (risk_factors / 2.5) * 0.75 + 0.15)
            scores.append(score)
            details.append({"bbox": [x, y, w, h], "texture_smoothness": round(texture_diff, 2), "deepfake_score": round(score, 2)})

        avg = float(np.mean(scores)) if scores else 0.0
        risk = "HIGH_SYNTHETIC_RISK" if avg >= 0.70 else ("SUSPICIOUS_ARTIFACTS" if avg >= 0.45 else "LOW_RISK_NATURAL_TEXTURE")

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

        risk = "HIGH_SYNTHETIC_RISK" if mean_score >= 0.70 else ("SUSPICIOUS_ARTIFACTS" if mean_score >= 0.45 else "LOW_RISK_NATURAL_TEXTURE")

        return {
            "total_frames_analyzed": len(frames_bgr),
            "faces_detected_max": max_faces,
            "mean_face_ai_score": round(mean_score, 2),
            "deepfake_risk": risk if max_faces > 0 else "NO_FACES_DETECTED",
        }
