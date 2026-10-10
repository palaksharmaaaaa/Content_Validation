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
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from core.imageio import imread
import numpy as np

from core.perception.age import get_age_estimator
from core.perception.colors import dominant_colors
from core.perception.detector import ANIMALS, VEHICLES, get_object_detector
from core.perception.enrich import describe_scene, recognize_details
from image_detector.face import FaceDeepfakeDetector

logger = logging.getLogger("image_detector.content")

COCO_ANIMALS = ANIMALS
COCO_VEHICLES = VEHICLES

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

def _scaled_box(box, factor: float):
    """An (x, y, width, height) box with every number multiplied by ``factor`` and rounded to whole pixels."""
    return tuple(int(round(v * factor)) for v in box)


class ImageContentAnalyzer:
    """Scene, object inventory, faces, lighting and environmental intelligence analyzer for images.

    Detection uses RF-DETR, recognition (scene, species, vehicle type, genre, time of day) uses SigLIP 2, expression and
    same-person matching use the OpenCV zoo models (see core.perception). Every model degrades to "not available".
    """

    def __init__(self):
        self.face_detector = FaceDeepfakeDetector()
        self.detector = get_object_detector()

    def analyze_image_content(self, image_path: str | Path | np.ndarray) -> Dict[str, Any]:
        """Runs multi-dimensional content analysis on an image."""
        if isinstance(image_path, (str, Path)):
            path = Path(image_path)
            if not path.is_file():
                return self._empty_result()
            try:
                img_bgr = imread(str(path))
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
        scale = 1.0
        if max_dim > 1280:
            scale = 1280.0 / max_dim
            sample_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            sample_bgr = img_bgr
        up = 1.0 / scale                      # factor that maps a box found in ``sample_bgr`` back onto the original picture

        # 1. Neural object detection (RF-DETR Small, core.perception.detector)
        (
            person_boxes,
            detected_animals,
            animal_details,
            detected_vehicles,
            vehicle_details,
            detected_items,
            item_details,
        ) = self._detect_objects(sample_bgr)

        # 1b. What kind of animal / vehicle (zero-shot recognition on each box)
        recognize_details(sample_bgr, animal_details, "animal")
        recognize_details(sample_bgr, vehicle_details, "vehicle")

        # 2. Faces: detection, texture risk, expression, same-person groups
        face_info = self.face_detector.analyze_faces(sample_bgr)
        faces_detected = face_info["faces_detected"]

        human_count, faces_detected, single_character_detected = self._count_humans(sample_bgr, person_boxes, faces_detected)

        # 2b. Age and minor screening for everyone found (recall-first; see core.perception.age)
        # The age screen looks at the ORIGINAL pixels: a small face that survives in the full picture can be lost when it is shrunk to 1280 px.
        age_info = get_age_estimator().assess(
            img_bgr, persons=[_scaled_box(b, up) for b in person_boxes], persons_available=get_object_detector().available)

        # 3. Lighting & Daytime Analysis
        scene = describe_scene(sample_bgr)
        lighting_info = self._analyze_lighting(sample_bgr)
        if scene.get("time_of_day"):
            lighting_info["daytime"] = scene["time_of_day"].capitalize()

        # 4. Tone & Mood Analysis
        tone_info = self._analyze_tone_and_color(sample_bgr)

        # 5. Environment & Surroundings
        environment = self._infer_environment(sample_bgr, detected_items, detected_vehicles, detected_animals)
        if scene.get("scene"):
            environment["setting"] = scene["scene"].capitalize()
        environment.update({k: scene[k] for k in ("scene_confidence", "scene_candidates", "indoor") if k in scene})

        text_regions_count = self._count_text_regions(img_bgr)

        # 6. Depiction & Purpose
        purpose = self._infer_purpose(human_count, h, w, detected_items, text_regions_count=text_regions_count, is_character=single_character_detected)
        if "genre" in scene:
            # The trained recogniser decides the genre; the text-region heuristic over-fires on busy photos, so a
            # document layout is only kept when the recogniser also says "document or screenshot".
            genre = scene["genre"]
            purpose["primary_genre"] = genre.capitalize() if genre else "General scene"
            if genre != "document or screenshot":
                purpose["document_layout"] = "None (Standard Visual Content)"

        # Boxes are reported in the coordinates of the picture the user supplied, not of the shrunken copy the models looked at.
        person_boxes = [_scaled_box(b, up) for b in person_boxes]
        face_info = {**face_info, "face_boxes": [
            {"x": int(round(b["x"] * up)), "y": int(round(b["y"] * up)), "width": int(round(b["width"] * up)), "height": int(round(b["height"] * up))}
            for b in face_info.get("face_boxes", [])]}
        result = self._assemble_result(
            human_count, faces_detected, single_character_detected, person_boxes, face_info,
            (detected_animals, animal_details), (detected_vehicles, vehicle_details), (detected_items, item_details),
            text_regions_count, environment, lighting_info, tone_info, purpose,
        )
        result["entities"]["humans"]["age_estimation"] = age_info
        result["minors"] = {k: age_info[k] for k in ("status", "contains_minor", "contains_possible_minor", "review_required", "youngest_age", "n_subjects")}
        result["colors"] = dominant_colors(sample_bgr)
        return result

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
                "expressions": face_info.get("expressions", []),
                "distinct_people_in_faces": face_info.get("distinct_people", 0),
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
            "scene_confidence": environment.get("scene_confidence"),
            "scene_candidates": environment.get("scene_candidates", []),
            "indoor": environment.get("indoor"),
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
            human_count = faces_detected  # the person detector missed people whose faces were found
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
        """Detects person boxes, animals, vehicles, and items with RF-DETR."""
        person_boxes: List[Tuple[int, int, int, int]] = []
        detected_animals: List[str] = []
        animal_details: List[Dict[str, Any]] = []
        detected_vehicles: List[str] = []
        vehicle_details: List[Dict[str, Any]] = []
        detected_items: List[str] = []
        item_details: List[Dict[str, Any]] = []
        try:
            detections = self.detector.detect(img_bgr)
        except Exception as e:
            logger.warning("Object detection failed: %s", e)
            detections = []
        for det in detections:
            label, score = det["label"], det["score"]
            x, y, w, h = det["box"]
            bbox_dict = {"x": x, "y": y, "width": w, "height": h, "confidence": round(score, 2)}
            if label == "person":
                person_boxes.append((x, y, w, h))
            elif label in COCO_ANIMALS:
                detected_animals.append(label)
                animal_details.append({"name": label, "score": round(score, 2), "bbox": bbox_dict})
            elif label in COCO_VEHICLES:
                detected_vehicles.append(label)
                vehicle_details.append({"name": label, "score": round(score, 2), "bbox": bbox_dict})
            else:
                if label not in detected_items:
                    detected_items.append(label)
                item_details.append({"name": label, "score": round(score, 2), "bbox": bbox_dict})
        return person_boxes, detected_animals, animal_details, detected_vehicles, vehicle_details, detected_items, item_details

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
        """Infers photographic purpose and genre covering the photographic genres."""
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
