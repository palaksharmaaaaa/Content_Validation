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
from core.perception.face_texture import risk_label, texture_cues
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

            cues = texture_cues(face_roi_gray)
            face_prob = cues["score"]
            deepfake_scores.append(face_prob)

            embedding = attributes.embedding(image_bgr, found["landmarks"])
            embeddings.append(embedding)
            face_details.append({
                "bbox": [x, y, w, h],
                "detector_confidence": round(float(found["score"]), 3),
                "expression": attributes.expression(image_bgr, found["landmarks"]),
                "embedding": None if embedding is None else [round(float(v), 5) for v in embedding],
                "texture_smoothness": round(cues["texture_smoothness"], 2),
                "facial_noise_residual": None if cues["facial_noise_residual"] is None else round(cues["facial_noise_residual"], 2),
                "is_waxy_texture": cues["is_waxy_texture"],
                "deepfake_score": round(face_prob, 2),
            })

        groups = group_same_person(embeddings)
        for detail, group in zip(face_details, groups):
            detail["same_person_group"] = group
        avg_score = float(np.mean(deepfake_scores)) if deepfake_scores else 0.0
        risk = risk_label(avg_score)

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
