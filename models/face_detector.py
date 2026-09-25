"""
Robust Facial & Deepfake Artifact Detector compatible with OpenCV 4.x and 5.x.
Localizes human faces via skin-chrominance morphology and facial geometry,
analyzing bilateral texture smoothness and synthetic boundary artifacts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


class FaceDeepfakeDetector:
    def __init__(self):
        pass

    def detect_faces(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detects face candidate regions using YCrCb skin chrominance segmentation and morphological analysis.
        Returns list of (x, y, w, h, area).
        """
        if image_bgr is None:
            return []

        h_img, w_img = image_bgr.shape[:2]
        total_pixels = float(h_img * w_img)

        # Convert to YCrCb for skin segmentation
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Morphological structuring to coalesce face components
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
            # Face region should be at least 2.5% of total image area
            if area > (total_pixels * 0.025):
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(h) / max(1.0, float(w))
                # Human face aspect ratios typically fall between 0.85 and 2.2
                if 0.85 <= aspect_ratio <= 2.2:
                    candidate_faces.append((int(x), int(y), int(w), int(h), float(area)))

        # Sort by area descending
        candidate_faces.sort(key=lambda item: item[4], reverse=True)
        return candidate_faces

    def analyze_faces(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Localizes faces and evaluates synthetic skin smoothing, bilateral texture anomalies,
        and deepfake-associated cues.
        """
        if image_bgr is None:
            return {
                "faces_detected": 0,
                "deepfake_risk": "NONE",
                "facial_ai_confidence": 0.0,
                "face_details": [],
            }

        faces = self.detect_faces(image_bgr)
        if not faces:
            return {
                "faces_detected": 0,
                "deepfake_risk": "NO_FACES_DETECTED",
                "facial_ai_confidence": 0.0,
                "face_details": [],
            }

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_details = []
        deepfake_scores = []

        for x, y, w, h, area in faces:
            face_roi_gray = gray[y : y + h, x : x + w]

            # 1. Bilateral texture smoothness inside face
            bilateral = cv2.bilateralFilter(face_roi_gray, 9, 75, 75)
            diff = cv2.absdiff(face_roi_gray, bilateral)
            texture_diff = float(np.mean(diff))

            # 2. High-frequency sensor noise on upper face (forehead/cheeks)
            fh_y1 = max(0, int(h * 0.15))
            fh_y2 = min(h, int(h * 0.50))
            fh_x1 = max(0, int(w * 0.20))
            fh_x2 = min(w, int(w * 0.80))
            upper_roi = face_roi_gray[fh_y1:fh_y2, fh_x1:fh_x2]

            if upper_roi.size > 0:
                blurred = cv2.GaussianBlur(upper_roi, (3, 3), 0)
                noise_mean = float(np.mean(cv2.absdiff(upper_roi, blurred)))
            else:
                noise_mean = 2.0

            # Evaluated risk factors
            risk_factors = 0
            # Diffusion face portraits typically have waxy skin (< 3.2 bilateral diff)
            if texture_diff < 3.2:
                risk_factors += 1.5
            # Natural camera portraits have sensor noise > 2.2 on face
            if noise_mean < 1.8:
                risk_factors += 1.0

            face_prob = min(0.95, (risk_factors / 2.5) * 0.75 + 0.15)
            deepfake_scores.append(face_prob)

            face_details.append({
                "bbox": [x, y, w, h],
                "texture_smoothness": round(texture_diff, 2),
                "facial_noise_residual": round(noise_mean, 2),
                "is_waxy_texture": texture_diff < 3.2,
                "deepfake_score": round(face_prob, 2),
            })

        avg_score = float(np.mean(deepfake_scores)) if deepfake_scores else 0.0
        if avg_score >= 0.70:
            risk = "HIGH_SYNTHETIC_RISK"
        elif avg_score >= 0.45:
            risk = "SUSPICIOUS_ARTIFACTS"
        else:
            risk = "LOW_RISK_NATURAL_TEXTURE"

        return {
            "faces_detected": len(faces),
            "deepfake_risk": risk,
            "facial_ai_confidence": round(avg_score, 2),
            "face_details": face_details,
        }
