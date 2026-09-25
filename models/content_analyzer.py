"""
Comprehensive Scene, Content, Environment & Tone Intelligence Engine.
Analyzes:
1. Contents, items, and objects present (electronics, furniture, accessories, apparel, tools).
2. Living entities: Humans (count, faces, boxes) and Animals (detected species/types).
3. Vehicles: Automobiles, bikes, aircraft, watercraft, public transport.
4. Surroundings & Environment: Indoor vs Outdoor, Urban, Nature, Studio, Domestic, Office, Coastal.
5. Daytime & Lighting: Daylight, Golden Hour/Sunset, Night/Low-light, Dawn/Dusk, Studio Strobe.
6. Tone & Mood / Atmosphere: Warm & Golden, Cool/Cinematic, Neutral, Dramatic/High-Contrast, Soft/Muted.
7. Purpose & Depiction: Portrait, Landscape, Commercial/Product, Editorial/News, Concept Art.
8. Audio Scene & Acoustic Tone: Setting, delivery style, speech vs music vs environmental sound.

Runs 100% locally and offline without external paid APIs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

from models.face_detector import FaceDeepfakeDetector
from utils.logging_utils import get_logger

logger = get_logger("content_analyzer")

# Semantic category keywords for ImageNet-1K classification
ANIMAL_KEYWORDS = [
    "dog", "cat", "bird", "horse", "lion", "tiger", "bear", "elephant", "zebra", "giraffe",
    "monkey", "ape", "wolf", "fox", "rabbit", "deer", "cow", "ox", "sheep", "goat", "pig",
    "fish", "shark", "whale", "dolphin", "snake", "lizard", "turtle", "frog", "eagle", "owl",
    "parrot", "duck", "goose", "swan", "penguin", "butterfly", "bee", "spider", "crab", "lobster"
]

VEHICLE_KEYWORDS = [
    "car", "automobile", "cab", "taxi", "convertible", "coupe", "jeep", "limousine", "minivan",
    "van", "truck", "pickup", "trailer", "bus", "train", "locomotive", "streetcar", "tram",
    "bicycle", "bike", "motorcycle", "scooter", "airplane", "airliner", "aircraft", "helicopter",
    "boat", "speedboat", "canoe", "yacht", "ship", "submarine", "cart", "wagon"
]

ITEM_KEYWORDS = [
    "phone", "laptop", "computer", "screen", "monitor", "television", "camera", "clock", "watch",
    "chair", "table", "desk", "sofa", "couch", "bed", "lamp", "book", "bottle", "cup", "mug",
    "plate", "bowl", "fork", "knife", "spoon", "guitar", "piano", "violin", "drum", "tie", "suit",
    "coat", "jacket", "dress", "hat", "glasses", "sunglasses", "backpack", "bag", "purse", "umbrella",
    "shoe", "boot", "sneaker", "ring", "necklace", "vase", "mirror", "pillow", "blanket"
]


class ContentAnalyzer:
    """Deep Content, Environmental Surroundings, Lighting & Tone Analyzer."""

    def __init__(self):
        self.face_detector = FaceDeepfakeDetector()
        self.vision_model = None
        self.categories = []
        self.transform = None
        self._init_vision_backbone()

    def _init_vision_backbone(self) -> None:
        """Loads cached ResNet18 ImageNet weights for zero-cost local object recognition."""
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.vision_model = models.resnet18(weights=weights).eval()
            self.categories = weights.meta.get("categories", [])
            self.transform = weights.transforms()
            logger.info("ContentAnalyzer initialized with local ImageNet backbone.")
        except Exception as exc:
            logger.warning("Could not initialize ImageNet backbone: %s", exc)

    def analyze_image_content(self, image_path: str | Path) -> Dict[str, Any]:
        """
        Comprehensive visual analysis of items, surroundings, daytime, tone,
        entities, vehicles, and purpose.
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return {"error": "Could not read image file."}

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 1. Living Entities (Humans & Animals)
        face_res = self.face_detector.analyze_faces(img_bgr)
        faces_detected = face_res.get("faces_detected", 0)
        face_boxes = face_res.get("face_boxes", [])

        # HOG person detection
        persons = self._detect_persons_hog(img_bgr)
        persons_count = max(len(persons), faces_detected)

        # 2. Local Deep Vision Object & Item Recognition (Top-10 ImageNet)
        detected_items, detected_animals, detected_vehicles, top_classes = self._classify_objects_and_entities(image_path)

        # 3. Surroundings & Environmental Setting
        environment_info = self._analyze_environment(img_bgr, hsv, faces_detected, top_classes)

        # 4. Lighting & Daytime Estimation
        lighting_info = self._analyze_lighting_and_daytime(img_bgr, hsv)

        # 5. Tone & Atmosphere
        tone_info = self._analyze_tone_and_mood(img_bgr, hsv)

        # 6. Text & Typography
        text_boxes = self._detect_text_regions(gray)
        text_count = len(text_boxes)

        # 7. Purpose & Depiction Classification
        depiction_info = self._determine_depiction_purpose(
            faces_count=faces_detected,
            persons_count=persons_count,
            text_count=text_count,
            env_type=environment_info["setting_type"],
            top_classes=top_classes,
            img_shape=(h, w),
            face_boxes=face_boxes,
        )

        return {
            "entities": {
                "humans": {
                    "persons_count": persons_count,
                    "faces_count": faces_detected,
                    "face_boxes": face_boxes,
                    "has_humans": (persons_count > 0 or faces_detected > 0),
                },
                "animals": {
                    "animals_detected": len(detected_animals) > 0,
                    "animal_types": detected_animals,
                },
            },
            "vehicles": {
                "vehicles_detected": len(detected_vehicles) > 0,
                "vehicle_types": detected_vehicles,
            },
            "contents_and_items": {
                "identified_items": detected_items,
                "text_regions_count": text_count,
                "has_text": text_count > 0,
                "top_recognitions": top_classes[:5],
            },
            "environment_and_surroundings": environment_info,
            "lighting_and_daytime": lighting_info,
            "tone_and_mood": tone_info,
            "purpose_and_depiction": depiction_info,
        }

    def _classify_objects_and_entities(
        self, image_path: str | Path
    ) -> Tuple[List[str], List[str], List[str], List[Dict[str, Any]]]:
        """Runs ResNet18 and parses ImageNet predictions into items, animals, and vehicles."""
        if self.vision_model is None or not self.categories:
            return [], [], [], []

        try:
            with Image.open(image_path) as p_img:
                tensor_img = self.transform(p_img.convert("RGB")).unsqueeze(0)

            with torch.no_grad():
                logits = self.vision_model(tensor_img)
                probs = torch.softmax(logits, dim=1)[0]
                topk = torch.topk(probs, 8)

            top_classes = []
            detected_items = []
            detected_animals = []
            detected_vehicles = []

            for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
                cat_name = self.categories[idx].replace("_", " ").lower()
                prob_pct = round(prob * 100.0, 1)

                if prob >= 0.04:
                    top_classes.append({"label": cat_name.title(), "confidence_pct": prob_pct})

                    # Categorize into animals, vehicles, items
                    is_animal = any(kw in cat_name for kw in ANIMAL_KEYWORDS)
                    is_vehicle = any(kw in cat_name for kw in VEHICLE_KEYWORDS)

                    if is_animal:
                        detected_animals.append(cat_name.title())
                    elif is_vehicle:
                        detected_vehicles.append(cat_name.title())
                    else:
                        detected_items.append(cat_name.title())

            return detected_items[:6], detected_animals[:4], detected_vehicles[:4], top_classes
        except Exception as exc:
            logger.warning("Object classification error: %s", exc)
            return [], [], [], []

    def _analyze_environment(
        self, img_bgr: np.ndarray, hsv: np.ndarray, faces_count: int, top_classes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Classifies indoor vs outdoor, specific setting typology, and surroundings."""
        h, w = img_bgr.shape[:2]
        total_px = float(h * w)

        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        # Green vegetation mask (Hue ~ 35 - 85)
        green_mask = (h_ch >= 35) & (h_ch <= 85) & (s_ch > 45)
        green_ratio = np.sum(green_mask) / total_px

        # Sky / Water blue mask (Hue ~ 90 - 130)
        blue_mask = (h_ch >= 90) & (h_ch <= 130) & (s_ch > 40)
        blue_ratio = np.sum(blue_mask) / total_px

        # Edge complexity
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_px

        is_outdoor = (green_ratio > 0.15) or (blue_ratio > 0.20) or (green_ratio + blue_ratio > 0.25)
        location = "Outdoor" if is_outdoor else "Indoor"

        # Determine detailed setting typology
        if faces_count > 0 and not is_outdoor and edge_density < 0.05:
            setting = "Studio / Controlled Portrait Setting"
        elif green_ratio > 0.25:
            setting = "Forest / Nature / Garden Environment"
        elif blue_ratio > 0.25:
            setting = "Coastal / Beach / Open Sky Setting"
        elif is_outdoor and edge_density > 0.08:
            setting = "Urban / City Street / Architecture"
        elif not is_outdoor and edge_density > 0.08:
            setting = "Office / Structured Workspace"
        elif not is_outdoor:
            setting = "Domestic Room / Home Interior"
        else:
            setting = "Natural Open Landscape"

        return {
            "location_context": location,
            "setting_type": setting,
            "vegetation_coverage_pct": round(green_ratio * 100.0, 1),
            "sky_water_coverage_pct": round(blue_ratio * 100.0, 1),
            "visual_complexity": "High / Detailed" if edge_density > 0.08 else ("Moderate" if edge_density > 0.04 else "Minimalist / Clean"),
        }

    def _analyze_lighting_and_daytime(self, img_bgr: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Determines daytime (daylight, golden hour, night, dawn/dusk) and lighting style."""
        brightness = float(np.mean(hsv[:, :, 2]))
        b_mean, g_mean, r_mean = [float(np.mean(img_bgr[:, :, i])) for i in range(3)]

        # Color Temperature: Red vs Blue
        is_warm = (r_mean > b_mean + 12)
        is_cool = (b_mean > r_mean + 10)

        # Highlight distribution (top 10% brightest pixels)
        upper_luma = float(np.percentile(hsv[:, :, 2], 90))

        if brightness < 45:
            daytime = "Night / Low-Light Environment"
            style = "Low-Key / Moody / Chiaroscuro"
        elif brightness < 80 and is_warm:
            daytime = "Golden Hour / Sunset / Twilight"
            style = "Warm Directional Sunlight"
        elif brightness < 85:
            daytime = "Dawn / Dusk / Dim Ambient"
            style = "Subdued Ambient Lighting"
        elif brightness > 185:
            daytime = "Bright Daylight / High-Key"
            style = "Intense Direct Sunlight / High Exposure"
        elif is_warm and upper_luma > 200:
            daytime = "Late Afternoon / Golden Sunlight"
            style = "Warm Directional Light"
        elif is_cool:
            daytime = "Daylight / Overcast Sky"
            style = "Cool Diffused Daylight"
        else:
            daytime = "Daylight / Balanced Ambient"
            style = "Balanced Natural / Studio Illumination"

        color_temp = "Warm / Golden" if is_warm else ("Cool / Blue-shifted" if is_cool else "Neutral / 5500K Balanced")

        return {
            "estimated_daytime": daytime,
            "lighting_style": style,
            "color_temperature": color_temp,
            "mean_luminance": round(brightness, 1),
            "highlight_intensity": round(upper_luma, 1),
        }

    def _analyze_tone_and_mood(self, img_bgr: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Evaluates color saturation, contrast, and atmospheric mood."""
        sat_mean = float(np.mean(hsv[:, :, 1]))
        contrast_std = float(np.std(hsv[:, :, 2]))

        if sat_mean < 35:
            color_tone = "Muted / Desaturated / Subtle"
        elif sat_mean > 120:
            color_tone = "Vibrant / Highly Saturated / Vivid"
        else:
            color_tone = "Natural / Moderate Saturation"

        if contrast_std > 65:
            mood = "Dramatic & Cinematic (High Dynamic Contrast)"
        elif contrast_std < 35:
            mood = "Soft, Dreamy & Atmospheric (Low Contrast)"
        elif sat_mean > 110:
            mood = "Lively & Energetic"
        else:
            mood = "Balanced, Calm & Realistic"

        return {
            "color_tone": color_tone,
            "atmospheric_mood": mood,
            "mean_saturation": round(sat_mean, 1),
            "contrast_index": round(contrast_std, 1),
        }

    def _determine_depiction_purpose(
        self,
        faces_count: int,
        persons_count: int,
        text_count: int,
        env_type: str,
        top_classes: List[Dict[str, Any]],
        img_shape: Tuple[int, int],
        face_boxes: List[Dict[str, int]],
    ) -> Dict[str, Any]:
        """Infers the intended depiction genre and photographic purpose."""
        h, w = img_shape
        total_area = float(h * w)

        # Check face area ratio
        total_face_area = sum(fb["width"] * fb["height"] for fb in face_boxes) if face_boxes else 0
        face_area_pct = (total_face_area / total_area) * 100.0

        if face_area_pct > 12.0 or (faces_count == 1 and face_area_pct > 6.0):
            purpose = "Portrait / Character & Identity Focus"
            depiction = "Individual human portrait with primary facial subject"
        elif text_count >= 3:
            purpose = "Infographic / Editorial / Marketing Graphic"
            depiction = "Information-rich composition with prominent typography"
        elif persons_count >= 2:
            purpose = "Social Interaction / Group Subject"
            depiction = "Multiple human subjects interacting in scene"
        elif "Nature" in env_type or "Coastal" in env_type or "Landscape" in env_type:
            purpose = "Landscape / Scenic Environmental Photography"
            depiction = "Expansive natural environment or scenic landscape"
        elif "Urban" in env_type:
            purpose = "Street / Architectural Photography"
            depiction = "Cityscape, street scene, or architectural structures"
        elif any(c["confidence_pct"] > 30.0 for c in top_classes):
            top_name = top_classes[0]["label"]
            purpose = f"Object Showcase / Still Life ({top_name})"
            depiction = f"Close inspection or showcase of {top_name.lower()}"
        else:
            purpose = "General Photographic / Artistic Depiction"
            depiction = "Standard environmental or creative scene"

        return {
            "photographic_purpose": purpose,
            "depiction_summary": depiction,
            "primary_subject_focus": "Human / Face" if faces_count > 0 else ("Typographic / Document" if text_count >= 3 else "Environment / Object"),
        }

    def _detect_text_regions(self, gray_img: np.ndarray) -> List[Dict[str, int]]:
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            grad = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel)
            _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, h_kernel)
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_boxes = []
            h_img, w_img = gray_img.shape[:2]
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / max(1, h)
                area = w * h
                if aspect > 1.4 and 100 < area < (w_img * h_img * 0.4) and h > 8:
                    text_boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
            return text_boxes[:20]
        except Exception:
            return []

    def _detect_persons_hog(self, img_bgr: np.ndarray) -> List[Dict[str, int]]:
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            h, w = img_bgr.shape[:2]
            scale = min(1.0, 640.0 / max(w, h))
            resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr
            boxes, weights = hog.detectMultiScale(resized, winStride=(8, 8), padding=(4, 4), scale=1.05)
            person_boxes = []
            for (bx, by, bw, bh), weight in zip(boxes, weights):
                if weight > 0.2:
                    person_boxes.append({"x": int(bx / scale), "y": int(by / scale), "width": int(bw / scale), "height": int(bh / scale)})
            return person_boxes[:15]
        except Exception:
            return []

    def analyze_audio_content(self, samples: np.ndarray, sample_rate: int, duration_sec: float) -> Dict[str, Any]:
        """
        Comprehensive acoustic analysis of speech, music, ambient setting,
        and audio delivery tone.
        """
        if samples is None or len(samples) < 1000 or sample_rate <= 0:
            return {
                "dominant_audio_type": "No Audio Signal",
                "acoustic_environment": "N/A",
                "vocal_tone_and_delivery": "N/A",
                "estimated_speakers": 0,
                "audio_purpose": "N/A",
            }

        frame_len = int(sample_rate * 0.03)
        hop = frame_len // 2
        num_frames = (len(samples) - frame_len) // hop + 1

        energies = []
        zero_crossings = []
        for i in range(num_frames):
            frame = samples[i * hop : i * hop + frame_len]
            energies.append(float(np.sqrt(np.mean(frame ** 2))))
            zero_crossings.append(float(np.mean(np.abs(np.diff(np.signbit(frame))))))

        energies = np.array(energies)
        zero_crossings = np.array(zero_crossings)

        silence_mask = energies < 0.015
        silence_ratio = float(np.mean(silence_mask))
        active_energies = energies[~silence_mask] if np.sum(~silence_mask) > 0 else np.array([0.01])
        energy_dyn_range = float(np.max(active_energies) / max(1e-4, np.percentile(active_energies, 10)))

        mean_zcr = float(np.mean(zero_crossings[~silence_mask])) if np.sum(~silence_mask) > 0 else 0.05
        speech_detected = (silence_ratio < 0.85) and (0.02 < mean_zcr < 0.40)

        # Harmonic detection for music
        fft_mag = np.abs(np.fft.rfft(samples[: min(len(samples), 32768)]))
        peaks = np.sort(fft_mag)[-10:]
        harmonic_ratio = float(np.sum(peaks) / max(1e-6, np.sum(fft_mag)))
        music_detected = (harmonic_ratio > 0.35) and not speech_detected

        # Acoustic Setting Determination
        if silence_ratio > 0.20:
            setting = "Studio / Anechoic (Dry Audio with Zero Room Reverb)"
        elif energy_dyn_range > 8.0:
            setting = "Domestic / Office Room with Moderate Reflection"
        elif silence_ratio < 0.05:
            setting = "Outdoor / Ambient Environment (Continuous Noise Floor)"
        else:
            setting = "Indoor Controlled Acoustic Space"

        # Tone & Delivery Style
        if speech_detected and energy_dyn_range > 7.0:
            tone = "Energetic / Dynamic & Expressive"
        elif speech_detected:
            tone = "Conversational / Calm & Measured"
        elif music_detected:
            tone = "Harmonic / Musical Score"
        else:
            tone = "Ambient Soundscape / Low-Energy"

        # Purpose / Genre
        if speech_detected and silence_ratio > 0.15:
            purpose = "Voiceover / Podcast / Studio Dialogue"
        elif speech_detected:
            purpose = "Live Interview / Casual Conversation"
        elif music_detected:
            purpose = "Musical Track / Background Audio"
        else:
            purpose = "Environmental Tone / Field Recording"

        return {
            "dominant_audio_type": "Human Speech" if speech_detected else ("Music / Melodic" if music_detected else "Ambient Environmental"),
            "acoustic_environment": setting,
            "vocal_tone_and_delivery": tone,
            "audio_purpose": purpose,
            "estimated_speakers": 1 if speech_detected else 0,
            "silence_ratio_pct": round(silence_ratio * 100.0, 1),
            "dynamic_range_index": round(energy_dyn_range, 2),
        }
