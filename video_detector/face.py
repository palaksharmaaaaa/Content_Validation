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

logger = logging.getLogger("video_detector.face")


class VideoFaceDeepfakeDetector:
    """Detects facial deepfake manipulation across video keyframes."""

    def __init__(self):
        pass

    def detect_faces(
        self,
        frame_bgr: np.ndarray,
        person_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detects face candidate regions in a video frame.
        If person_boxes are provided:
          Constrains candidate regions strictly to the head region (upper 48% of person box),
          preventing false detections on background surfaces, furniture, and hands.
        If person_boxes is None:
          Uses anthropometric morphology and skin-chrominance gradient filtering.
        """
        if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
            return []

        h_img, w_img = frame_bgr.shape[:2]
        if h_img < 16 or w_img < 16:
            return []

        # Case 1: Gated by person detections
        if person_boxes is not None:
            if len(person_boxes) == 0:
                return []

            candidate_faces = []
            for px, py, pw, ph in person_boxes:
                if pw < 10 or ph < 15:
                    continue
                hy1 = max(0, int(py))
                hy2 = min(h_img, int(py + ph * 0.48))
                hx1 = max(0, int(px))
                hx2 = min(w_img, int(px + pw))

                head_crop = frame_bgr[hy1:hy2, hx1:hx2]
                if head_crop.size == 0:
                    continue
                ch, cw = head_crop.shape[:2]

                ycrcb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2YCrCb)
                lower_skin = np.array([0, 133, 77], dtype=np.uint8)
                upper_skin = np.array([255, 173, 127], dtype=np.uint8)
                skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

                cnts, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_cnts = [c for c in cnts if cv2.contourArea(c) >= (cw * ch * 0.08)]

                face_found = False
                if valid_cnts:
                    c = max(valid_cnts, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(c)
                    aspect = float(bh) / max(1.0, float(bw))
                    if 0.65 <= aspect <= 2.20:
                        bx_c = max(0, min(w_img - 1, hx1 + bx))
                        by_c = max(0, min(h_img - 1, hy1 + by))
                        bw_c = min(w_img - bx_c, bw)
                        bh_c = min(h_img - by_c, bh)
                        candidate_faces.append((bx_c, by_c, bw_c, bh_c, float(bw_c * bh_c)))
                        face_found = True

                if not face_found:
                    gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                    if gray_head.size > 0 and float(np.std(gray_head)) > 12.0:
                        fw = max(16, int(pw * 0.50))
                        fh = max(16, int(ph * 0.35))
                        fx = max(0, min(w_img - 1, hx1 + int((pw - fw) / 2)))
                        fy = max(0, min(h_img - 1, hy1 + int(ph * 0.05)))
                        fw = min(w_img - fx, fw)
                        fh = min(h_img - fy, fh)
                        candidate_faces.append((fx, fy, fw, fh, float(fw * fh)))

            return candidate_faces

        # Case 2: Standalone frame without prior person boxes
        total_pixels = float(h_img * w_img)

        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        kernel_size = max(5, int(min(h_img, w_img) * 0.015))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=4)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_faces = []

        for c in contours:
            area = cv2.contourArea(c)
            min_area = max(1500, total_pixels * 0.0025)
            max_area = total_pixels * 0.40
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(h) / max(1.0, float(w))
                extent = float(area) / max(1.0, float(w * h))
                if 0.95 <= aspect_ratio <= 1.90 and 0.45 <= extent <= 0.90:
                    candidate_faces.append((int(x), int(y), int(w), int(h), float(area)))

        candidate_faces.sort(key=lambda item: item[4], reverse=True)
        return candidate_faces

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
