"""
image_detector.face: face detection (YuNet, see core.face_detection) and facial texture-artifact scoring.
The texture score (waxy skin, missing sensor noise) is a heuristic cue, not a trained deepfake detector.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from core.face_detection import get_face_finder
from core.perception.face_attributes import get_face_attributes, group_same_person

logger = logging.getLogger("image_detector.face")


class FaceDeepfakeDetector:
    """
    Independent facial deepfake artifact detector.
    Analyzes skin smoothing, bilateral texture anomalies, and sensor noise absence.
    """

    def __init__(self):
        pass

    def detect_faces(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Faces found by the YuNet detector, as (x, y, w, h, confidence)."""
        return get_face_finder().find(image_bgr)

    def analyze_faces(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Localizes faces and evaluates synthetic skin smoothing, bilateral texture anomalies,
        and deepfake-associated cues.
        """
        if image_bgr is None or not hasattr(image_bgr, "shape"):
            return {
                "faces_detected": 0,
                "deepfake_risk": "NONE",
                "facial_ai_confidence": 0.0,
                "face_details": [],
                "face_boxes": [],
            }

        detailed = get_face_finder().find_detailed(image_bgr)
        faces = [(*d["box"], d["score"]) for d in detailed]
        if not faces:
            return {
                "faces_detected": 0,
                "deepfake_risk": "NO_FACES_DETECTED",
                "facial_ai_confidence": 0.0,
                "face_details": [],
                "face_boxes": [],
            }

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_details = []
        deepfake_scores = []

        attributes = get_face_attributes()
        embeddings = []
        for (x, y, w, h, area), found in zip(faces, detailed):
            face_roi_gray = gray[y : y + h, x : x + w]
            if face_roi_gray.size == 0:
                continue

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
            risk_factors = 0.0
            if texture_diff < 3.2:
                risk_factors += 1.5
            if noise_mean < 1.8:
                risk_factors += 1.0

            face_prob = min(0.95, (risk_factors / 2.5) * 0.75 + 0.15)
            deepfake_scores.append(face_prob)

            embedding = attributes.embedding(image_bgr, found["landmarks"])
            embeddings.append(embedding)
            face_details.append({
                "bbox": [x, y, w, h],
                "detector_confidence": round(float(found["score"]), 3),
                "expression": attributes.expression(image_bgr, found["landmarks"]),
                "embedding": None if embedding is None else [round(float(v), 5) for v in embedding],
                "texture_smoothness": round(texture_diff, 2),
                "facial_noise_residual": round(noise_mean, 2),
                "is_waxy_texture": texture_diff < 3.2,
                "deepfake_score": round(face_prob, 2),
            })

        groups = group_same_person(embeddings)
        for detail, group in zip(face_details, groups):
            detail["same_person_group"] = group
        avg_score = float(np.mean(deepfake_scores)) if deepfake_scores else 0.0
        if avg_score >= 0.70:
            risk = "HIGH_SYNTHETIC_RISK"
        elif avg_score >= 0.45:
            risk = "SUSPICIOUS_ARTIFACTS"
        else:
            risk = "LOW_RISK_NATURAL_TEXTURE"

        face_boxes = [
            {"x": d["bbox"][0], "y": d["bbox"][1], "width": d["bbox"][2], "height": d["bbox"][3]}
            for d in face_details
        ]

        return {
            "faces_detected": len(faces),
            "deepfake_risk": risk,
            "facial_ai_confidence": round(avg_score, 2),
            "face_details": face_details,
            "face_boxes": face_boxes,
            "expressions": [d["expression"]["label"] for d in face_details if d.get("expression")],
            "distinct_people": len({g for g in groups if g >= 0}),
        }
