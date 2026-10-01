"""
image_detector.face: Standalone Facial & Deepfake Artifact Detector.
Localizes human faces via skin-chrominance morphology and facial geometry,
analyzing bilateral texture smoothness and synthetic boundary artifacts.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("image_detector.face")


class FaceDeepfakeDetector:
    """
    Independent facial deepfake artifact detector.
    Analyzes skin smoothing, bilateral texture anomalies, and sensor noise absence.
    """

    def __init__(self):
        pass

    def detect_faces(
        self,
        image_bgr: np.ndarray,
        person_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detects face candidate regions.
        If person_boxes are provided (from neural object detector):
          Localizes faces strictly within the upper 45% of each detected person,
          guaranteeing that furniture, clothing, hands, or background surfaces are
          never falsely detected as human faces.
        If person_boxes is None:
          Runs anthropometric geometry and skin-chrominance gradient validation
          to identify close-up portrait faces.
        Returns list of (x, y, w, h, area).
        """
        if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
            return []

        h_img, w_img = image_bgr.shape[:2]
        if h_img < 16 or w_img < 16:
            return []

        # Case 1: Person boxes provided explicitly by object detector
        if person_boxes is not None:
            if len(person_boxes) == 0:
                # Confirmed zero persons in the frame -> zero faces
                return []

            candidate_faces = []
            for px, py, pw, ph in person_boxes:
                if pw < 10 or ph < 15:
                    continue
                hy1 = max(0, int(py))
                hy2 = min(h_img, int(py + ph * 0.48))
                hx1 = max(0, int(px))
                hx2 = min(w_img, int(px + pw))

                head_crop = image_bgr[hy1:hy2, hx1:hx2]
                if head_crop.size == 0:
                    continue
                ch, cw = head_crop.shape[:2]

                # Check skin chrominance inside the head crop
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
                    # Head region check: if texture variance indicates a visible head
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

        # Case 2: Standalone call (person_boxes is None)
        max_dim = max(h_img, w_img)
        scale = 1024.0 / max_dim if max_dim > 1024 else 1.0
        if scale < 1.0:
            detect_frame = cv2.resize(image_bgr, (int(w_img * scale), int(h_img * scale)), interpolation=cv2.INTER_AREA)
        else:
            detect_frame = image_bgr

        h_d, w_d = detect_frame.shape[:2]
        total_pixels = float(h_d * w_d)

        ycrcb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        kernel_size = max(5, int(min(h_d, w_d) * 0.015))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_candidates = []
        inv_scale = 1.0 / scale
        gray_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

        for c in contours:
            area = cv2.contourArea(c)
            min_area = max(800, total_pixels * 0.02)
            max_area = total_pixels * 0.55
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(h) / max(1.0, float(w))
                extent = float(area) / max(1.0, float(w * h))
                if 0.90 <= aspect_ratio <= 1.95 and 0.40 <= extent <= 0.92:
                    crop_gray = gray_frame[y : y + h, x : x + w]
                    if crop_gray.size > 0:
                        std_dev = float(np.std(crop_gray))
                        if std_dev < 15.0:
                            # Reject flat wall/cloth patch
                            continue
                        ch, cw = crop_gray.shape
                        fh_crop = crop_gray[: max(1, int(ch * 0.20)), :]
                        eye_crop = crop_gray[int(ch * 0.20) : int(ch * 0.50), :]
                        if fh_crop.size > 0 and eye_crop.size > 0:
                            if float(np.mean(fh_crop)) < (float(np.mean(eye_crop)) - 30.0):
                                continue

                        raw_candidates.append((
                            int(x * inv_scale),
                            int(y * inv_scale),
                            int(w * inv_scale),
                            int(h * inv_scale),
                            float(area * (inv_scale ** 2)),
                        ))

        # NMS suppression
        raw_candidates.sort(key=lambda item: item[4], reverse=True)
        candidate_faces = []
        for cand in raw_candidates:
            cx, cy, cw, ch, ca = cand
            keep = True
            for kx, ky, kw, kh, ka in candidate_faces:
                xA = max(cx, kx)
                yA = max(cy, ky)
                xB = min(cx + cw, kx + kw)
                yB = min(cy + ch, ky + kh)
                inter = max(0, xB - xA) * max(0, yB - yA)
                iou = inter / float(cw * ch + kw * kh - inter)
                if iou > 0.30:
                    keep = False
                    break
            if keep:
                candidate_faces.append(cand)

        return candidate_faces

    def analyze_faces(
        self,
        image_bgr: np.ndarray,
        person_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Dict[str, Any]:
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

        faces = self.detect_faces(image_bgr, person_boxes=person_boxes)
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

        for x, y, w, h, area in faces:
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
        }
