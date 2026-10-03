"""
image_detector.content: Comprehensive Scene, Content, Environment & Tone Intelligence for Images.
Analyzes:
1. Objects & Items present (electronics, accessories, furniture, apparel).
2. Living entities: Humans (count, faces, boxes) and Animals.
3. Vehicles: Automobiles, bikes, aircraft, watercraft.
4. Environment: Indoor vs Outdoor, Urban, Nature, Studio, Domestic.
5. Daytime & Lighting: Daylight, Sunset/Golden Hour, Low-Light/Night, Studio Strobe.
6. Tone & Mood: Warm, Cool, Neutral, High-Contrast, Soft/Muted.
7. Purpose & Depiction: Portrait, Landscape, Commercial/Product, Editorial.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from image_detector.face import FaceDeepfakeDetector

logger = logging.getLogger("image_detector.content")

COCO_ANIMALS = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
}

COCO_VEHICLES = {
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
}

COCO_ITEMS = {
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

# Module-level shared model cache to avoid reloading weights across instances
_SHARED_VISION_MODEL = None
_VISION_INIT_LOCK = threading.Lock()
_SHARED_CATEGORIES = None


class ImageContentAnalyzer:
    """Scene, object inventory, lighting, and environmental intelligence analyzer for images."""

    def __init__(self):
        self.face_detector = FaceDeepfakeDetector()
        self.vision_model = None
        self.categories: List[str] = []
        self._init_vision_backbone()

    def _init_vision_backbone(self) -> None:
        """Loads a pretrained neural object detector (SSDLite320 MobileNet V3 Large) for real-time bounding box detection."""
        with _VISION_INIT_LOCK:  # one load per process even when several sessions initialise concurrently
            self._init_vision_backbone_locked()

    def _init_vision_backbone_locked(self) -> None:
        global _SHARED_VISION_MODEL, _SHARED_CATEGORIES
        if _SHARED_VISION_MODEL is not None and _SHARED_CATEGORIES is not None:
            self.vision_model = _SHARED_VISION_MODEL
            self.categories = _SHARED_CATEGORIES
            return

        try:
            import torch
            from torchvision.models.detection import (
                ssdlite320_mobilenet_v3_large,
                SSDLite320_MobileNet_V3_Large_Weights,
            )
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.vision_model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
            if torch.cuda.is_available():
                self.vision_model = self.vision_model.to("cuda")
            self.categories = weights.meta.get("categories", [])
            _SHARED_VISION_MODEL = self.vision_model
            _SHARED_CATEGORIES = self.categories
            logger.info("ImageContentAnalyzer neural object detector backbone loaded successfully.")
        except Exception as e:
            logger.debug("Neural detector backbone not initialized, running fallback: %s", e)
            self.vision_model = None
            self.categories = []

    def analyze_image_content(self, image_path: str | Path | np.ndarray) -> Dict[str, Any]:
        """Runs multi-dimensional content analysis on an image."""
        if isinstance(image_path, (str, Path)):
            path = Path(image_path)
            if not path.is_file():
                return self._empty_result()
            try:
                img_bgr = cv2.imread(str(path))
            except Exception:
                img_bgr = None
        elif isinstance(image_path, np.ndarray):
            img_bgr = image_path
        else:
            return self._empty_result()

        if img_bgr is None:
            return self._empty_result()

        h, w = img_bgr.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1280:
            scale = 1280.0 / max_dim
            sample_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            sample_bgr = img_bgr

        # 1. Neural Object Detection (SSDLite MobileNet V3)
        (
            person_boxes,
            detected_animals,
            animal_details,
            detected_vehicles,
            vehicle_details,
            detected_items,
            item_details,
        ) = self._detect_objects(sample_bgr)

        # 2. Face & Human / Character Detection Anchored to Persons
        face_info = self.face_detector.analyze_faces(sample_bgr, person_boxes=person_boxes)
        faces_detected = face_info["faces_detected"]

        human_count, faces_detected, single_character_detected = self._count_humans(sample_bgr, person_boxes, faces_detected)

        # 3. Lighting & Daytime Analysis
        lighting_info = self._analyze_lighting(sample_bgr)

        # 4. Tone & Mood Analysis
        tone_info = self._analyze_tone_and_color(sample_bgr)

        # 5. Environment & Surroundings
        environment = self._infer_environment(sample_bgr, detected_items, detected_vehicles, detected_animals)

        text_regions_count = self._count_text_regions(img_bgr)

        # 6. Depiction & Purpose
        purpose = self._infer_purpose(human_count, h, w, detected_items, text_regions_count=text_regions_count, is_character=single_character_detected)

        return self._assemble_result(
            human_count, faces_detected, single_character_detected, person_boxes, face_info,
            (detected_animals, animal_details), (detected_vehicles, vehicle_details), (detected_items, item_details),
            text_regions_count, environment, lighting_info, tone_info, purpose,
        )

    @staticmethod
    def _assemble_result(
        human_count, faces_detected, single_character_detected, person_boxes, face_info,
        animals, vehicles, items, text_regions_count, environment, lighting_info, tone_info, purpose,
    ) -> Dict[str, Any]:
        """Builds the nested content-inventory dict (with the flat/legacy aliases consumers read)."""
        detected_animals, animal_details = animals
        detected_vehicles, vehicle_details = vehicles
        detected_items, item_details = items
        entities_dict = {
            "humans": {
                "count": human_count,
                "faces": faces_detected,
                "persons_count": human_count,
                "faces_count": faces_detected,
                "is_stylized_character": single_character_detected,
                "bounding_boxes": face_info.get("face_boxes", []),
                "person_boxes": [
                    {"x": b[0], "y": b[1], "width": b[2], "height": b[3]} for b in person_boxes
                ],
                "deepfake_analysis": {
                    "risk": face_info.get("deepfake_risk", "NONE"),
                    "confidence": face_info.get("facial_ai_confidence", 0.0),
                    "details": face_info.get("face_details", []),
                },
            },
            "animals": {
                "detected": len(detected_animals) > 0,
                "count": len(detected_animals),
                "types": detected_animals,
                "animal_types": detected_animals,
                "details": animal_details,
            },
        }

        vehicles_dict = {
            "detected": len(detected_vehicles) > 0,
            "count": len(detected_vehicles),
            "types": detected_vehicles,
            "vehicle_types": detected_vehicles,
            "details": vehicle_details,
        }

        items_dict = {
            "detected": len(detected_items) > 0,
            "count": len(detected_items),
            "items": detected_items,
            "identified_items": detected_items,
            "details": item_details,
            "text_regions_count": text_regions_count,
        }

        env_dict = {
            "setting": environment.get("setting", "Unknown"),
            "setting_type": environment.get("setting", "Unknown"),
            "vegetation_ratio": environment.get("vegetation_ratio", 0.0),
            "sky_water_ratio": environment.get("sky_water_ratio", 0.0),
        }

        light_dict = {
            "daytime": lighting_info.get("daytime", "Daylight"),
            "estimated_daytime": lighting_info.get("daytime", "Daylight"),
            "lighting_quality": lighting_info.get("lighting_quality", "Ambient"),
            "lighting_style": lighting_info.get("lighting_quality", "Ambient"),
            "mean_luminance": lighting_info.get("mean_luminance", 128.0),
            "luminance_dynamic_range": lighting_info.get("luminance_dynamic_range", 30.0),
        }

        tone_dict = {
            "color_temperature": tone_info.get("color_temperature", "Neutral"),
            "color_tone": tone_info.get("color_temperature", "Neutral"),
            "contrast_profile": tone_info.get("contrast_profile", "Balanced"),
            "atmospheric_mood": tone_info.get("contrast_profile", "Balanced"),
            "channel_means": tone_info.get("channel_means", {}),
        }

        purpose_dict = {
            "primary_genre": purpose.get("primary_genre", "General Scene"),
            "photographic_purpose": purpose.get("primary_genre", "General Scene"),
            "document_layout": purpose.get("document_layout", "None (Standard Visual Content)"),
            "aspect_ratio": purpose.get("aspect_ratio", 1.0),
        }

        return {
            "living_entities": entities_dict,
            "entities": entities_dict,
            "vehicles": vehicles_dict,
            "items_and_objects": items_dict,
            "contents_and_items": items_dict,
            "environment": env_dict,
            "environment_and_surroundings": env_dict,
            "lighting_and_daytime": light_dict,
            "tone_and_mood": tone_dict,
            "purpose_and_depiction": purpose_dict,
            "persons_count": human_count,
            "faces_count": faces_detected,
            "scene_type": env_dict["setting"],
        }

    @staticmethod
    def _count_humans(sample_bgr: np.ndarray, person_boxes: list, faces_detected: int) -> Tuple[int, int, bool]:
        """Returns (human_count, faces_detected, single_character_detected) reconciling person boxes with faces."""
        human_count = len(person_boxes)
        single_character_detected = False
        if human_count == 0:
            if faces_detected > 0:
                human_count = faces_detected
            else:
                # Isolated upright character / figure silhouette (masked character, hooded figure, anime character)
                gray_s = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2GRAY)
                sh, sw = gray_s.shape[:2]
                corners = [gray_s[:25, :25], gray_s[:25, -25:], gray_s[-25:, :25], gray_s[-25:, -25:]]
                if sum(float(np.std(c)) < 8.0 for c in corners) >= 3:
                    bg_val = float(np.median([float(np.mean(c)) for c in corners]))
                    fg_mask = (np.abs(gray_s.astype(float) - bg_val) > 18).astype(np.uint8)
                    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_bg)
                    cnts_bg, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    major_bodies = [c for c in cnts_bg if cv2.contourArea(c) > (sh * sw * 0.08)]
                    if len(major_bodies) == 1:
                        _bx, _by, bw, bh = cv2.boundingRect(major_bodies[0])
                        if bh > bw * 0.8:  # Upright character / figure
                            human_count = 1
                            single_character_detected = True
        elif faces_detected > human_count:
            faces_detected = human_count  # Physical invariant: each person has at most 1 face
        return human_count, faces_detected, single_character_detected

    @staticmethod
    def _count_text_regions(img_bgr: np.ndarray) -> int:
        """Counts text-like regions via a morphological-gradient / Otsu / closing pipeline."""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gh, gw = gray.shape[:2]
            if max(gh, gw) > 800:
                sc = 800.0 / max(gh, gw)
                s_gray = cv2.resize(gray, (int(gw * sc), int(gh * sc)), interpolation=cv2.INTER_AREA)
            else:
                s_gray = gray
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            m_grad = cv2.morphologyEx(s_gray, cv2.MORPH_GRADIENT, kernel)
            _, bw = cv2.threshold(m_grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3)))
            cnts, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for c in cnts:
                _x, _y, cw, ch = cv2.boundingRect(c)
                aspect = float(cw) / max(1.0, float(ch))
                if 2.0 <= aspect <= 30.0 and 120 <= (cw * ch) <= 50000:
                    count += 1
            return count
        except Exception as exc:
            logger.debug("_count_text_regions: ignored %s: %s", type(exc).__name__, exc)
            return 0

    def _detect_objects(
        self, img_bgr: np.ndarray
    ) -> Tuple[
        List[Tuple[int, int, int, int]],
        List[str],
        List[Dict[str, Any]],
        List[str],
        List[Dict[str, Any]],
        List[str],
        List[Dict[str, Any]],
    ]:
        """Detects person boxes, animals, vehicles, and items via SSDLite MobileNet V3 with strict confidence filtering."""
        if self.vision_model is None or not self.categories:
            return [], [], [], [], [], [], []

        try:
            import torch
            import torchvision.transforms.functional as TF
            h_img, w_img = img_bgr.shape[:2]
            max_d = max(h_img, w_img)
            scale = 640.0 / max_d if max_d > 640 else 1.0
            sw, sh = int(w_img * scale), int(h_img * scale)
            inv_scale = 1.0 / scale

            resized_bgr = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            tensor = TF.to_tensor(rgb).unsqueeze(0)

            with torch.no_grad():
                model_device = next(self.vision_model.parameters()).device
                preds = self.vision_model(tensor.to(model_device))[0]

            boxes = preds["boxes"]
            labels = preds["labels"]
            scores = preds["scores"]

            keep = scores >= 0.35
            k_boxes = boxes[keep].tolist()
            k_labels = labels[keep].tolist()
            k_scores = scores[keep].tolist()

            del tensor, preds, boxes, labels, scores

            person_boxes: List[Tuple[int, int, int, int]] = []
            detected_animals: List[str] = []
            animal_details: List[Dict[str, Any]] = []
            detected_vehicles: List[str] = []
            vehicle_details: List[Dict[str, Any]] = []
            detected_items: List[str] = []
            item_details: List[Dict[str, Any]] = []

            for box, label_idx, score in zip(k_boxes, k_labels, k_scores):
                if label_idx >= len(self.categories):
                    continue
                label_name = self.categories[label_idx].lower().strip()
                if label_name in ("__background__", "n/a"):
                    continue

                bx1, by1, bx2, by2 = box
                orig_x = max(0, int(bx1 * inv_scale))
                orig_y = max(0, int(by1 * inv_scale))
                orig_w = min(w_img - orig_x, int((bx2 - bx1) * inv_scale))
                orig_h = min(h_img - orig_y, int((by2 - by1) * inv_scale))
                bbox_dict = {"x": orig_x, "y": orig_y, "width": orig_w, "height": orig_h, "confidence": round(score, 2)}

                if label_name == "person":
                    if score >= 0.40:
                        person_boxes.append((orig_x, orig_y, orig_w, orig_h))
                elif label_name in COCO_ANIMALS:
                    if score >= 0.40:
                        detected_animals.append(label_name)
                        animal_details.append({"name": label_name, "score": round(score, 2), "bbox": bbox_dict})
                elif label_name in COCO_VEHICLES:
                    if score >= 0.40:
                        detected_vehicles.append(label_name)
                        vehicle_details.append({"name": label_name, "score": round(score, 2), "bbox": bbox_dict})
                else:
                    if label_name not in detected_items:
                        detected_items.append(label_name)
                    item_details.append({"name": label_name, "score": round(score, 2), "bbox": bbox_dict})

            return (
                person_boxes,
                detected_animals,
                animal_details,
                detected_vehicles,
                vehicle_details,
                detected_items,
                item_details,
            )
        except Exception as e:
            logger.warning("Neural object detection failed: %s", e)
            return [], [], [], [], [], [], []

    def _classify_semantic_categories(
        self, img_bgr: np.ndarray
    ) -> Tuple[List[str], List[str], List[str]]:
        """Classifies objects into items, animals, and vehicles (backwards compatible)."""
        _, animals, _, vehicles, _, items, _ = self._detect_objects(img_bgr)
        return items, animals, vehicles

    def _analyze_lighting(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Infers lighting conditions from luminance distribution."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))

        if mean_lum < 55.0:
            daytime = "Night / Low-Light"
            quality = "Low-Light / Ambient"
        elif mean_lum > 195.0:
            daytime = "Daylight / Overexposed"
            quality = "Direct Hard Light"
        else:
            # Check warm cast in upper quadrants for sunset/golden hour
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            hue_mean = float(np.mean(hsv[:, :, 0]))
            sat_mean = float(np.mean(hsv[:, :, 1]))
            if (hue_mean < 25 or hue_mean > 165) and sat_mean > 70:
                daytime = "Golden Hour / Sunset"
                quality = "Warm Directional Light"
            else:
                daytime = "Daylight"
                quality = "Diffuse Daylight" if std_lum < 50 else "Direct Daylight"

        return {
            "daytime": daytime,
            "lighting_quality": quality,
            "mean_luminance": round(mean_lum, 1),
            "luminance_dynamic_range": round(std_lum, 1),
        }

    def _analyze_tone_and_color(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Evaluates color temperature, contrast, and atmospheric tone."""
        b, g, r = cv2.split(img_bgr)
        mean_b, mean_g, mean_r = float(np.mean(b)), float(np.mean(g)), float(np.mean(r))

        temp = "Neutral"
        if mean_r > mean_b + 12:
            temp = "Warm"
        elif mean_b > mean_r + 12:
            temp = "Cool"

        std_lum = float(np.std(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)))
        if std_lum > 65:
            contrast = "High Contrast / Dramatic"
        elif std_lum < 35:
            contrast = "Low Contrast / Soft"
        else:
            contrast = "Balanced Contrast"

        return {
            "color_temperature": temp,
            "contrast_profile": contrast,
            "channel_means": {"red": round(mean_r, 1), "green": round(mean_g, 1), "blue": round(mean_b, 1)},
        }

    def _infer_environment(
        self, img_bgr: np.ndarray, items: List[str], vehicles: List[str], animals: List[str]
    ) -> Dict[str, Any]:
        """Infers setting type (indoor, outdoor, nature, studio)."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        blue_mask = cv2.inRange(hsv, (90, 40, 40), (130, 255, 255))

        green_ratio = float(np.sum(green_mask > 0)) / float(img_bgr.shape[0] * img_bgr.shape[1])
        blue_ratio = float(np.sum(blue_mask > 0)) / float(img_bgr.shape[0] * img_bgr.shape[1])

        if green_ratio > 0.20 or (blue_ratio > 0.25 and len(animals) > 0):
            env_type = "Outdoor / Nature"
        elif len(vehicles) > 0:
            env_type = "Outdoor / Urban"
        elif len(items) > 0:
            env_type = "Indoor / Domestic"
        else:
            env_type = "Studio / Isolated"

        return {
            "setting": env_type,
            "vegetation_ratio": round(green_ratio, 3),
            "sky_water_ratio": round(blue_ratio, 3),
        }

    def _infer_purpose(
        self, human_count: int, h: int, w: int, items: List[str], text_regions_count: int = 0, is_character: bool = False
    ) -> Dict[str, Any]:
        """Infers photographic purpose and genre covering the NIST/forensic catalog."""
        aspect = float(w) / max(1.0, float(h))
        
        # Check for Document / Identity / Layout
        document_layout = "None (Standard Visual Content)"
        if text_regions_count >= 8:
            if aspect > 1.3 and aspect < 1.8:
                document_layout = "Identity Card / Financial Cheque Layout"
                genre = "Document / Identity Verification"
            elif aspect < 0.9:
                document_layout = "Tax Invoice / Structured Receipt Layout"
                genre = "Document / Financial Invoice"
            else:
                document_layout = "Text Document / Official Record"
                genre = "Document / Text Record"
        elif human_count == 1 or is_character:
            if aspect <= 0.85:
                genre = "Individual Character / Portrait (Vertical)" if is_character else "Individual Portrait (Vertical Close-up)"
            else:
                genre = "Individual Character / Portrait" if is_character else "Individual Portrait"
        elif human_count > 1:
            genre = "Group / Social Scene"
        elif aspect >= 1.4:
            genre = "Landscape / Wide Vista Scene"
        elif len(items) > 0:
            genre = "Commercial / Product Photography"
        else:
            genre = "General Scene"

        return {
            "primary_genre": genre,
            "document_layout": document_layout,
            "aspect_ratio": round(aspect, 2),
        }

    def _empty_result(self) -> Dict[str, Any]:
        entities = {
            "humans": {"count": 0, "faces": 0, "persons_count": 0, "faces_count": 0, "bounding_boxes": []},
            "animals": {"detected": False, "count": 0, "types": [], "animal_types": []},
        }
        items = {"detected": False, "count": 0, "items": [], "identified_items": [], "text_regions_count": 0}
        env = {"setting": "Unknown", "setting_type": "Unknown", "vegetation_ratio": 0.0, "sky_water_ratio": 0.0}
        light = {"daytime": "Unknown", "estimated_daytime": "Unknown", "lighting_quality": "Unknown", "lighting_style": "Unknown"}
        tone = {"color_temperature": "Unknown", "color_tone": "Unknown", "contrast_profile": "Unknown", "atmospheric_mood": "Unknown"}
        purpose = {"primary_genre": "Unknown", "photographic_purpose": "Unknown", "document_layout": "None (Standard Visual Content)"}
        return {
            "living_entities": entities,
            "entities": entities,
            "vehicles": {"detected": False, "count": 0, "types": [], "vehicle_types": []},
            "items_and_objects": items,
            "contents_and_items": items,
            "environment": env,
            "environment_and_surroundings": env,
            "lighting_and_daytime": light,
            "tone_and_mood": tone,
            "purpose_and_depiction": purpose,
            "persons_count": 0,
            "faces_count": 0,
            "scene_type": "Unknown",
        }
