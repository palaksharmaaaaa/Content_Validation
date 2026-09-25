"""
Multi-Signal Forensic AI Image Detector.
Combines sensor noise profiling, bilateral surface texture, 2D FFT, Error Level Analysis (ELA),
EXIF provenance inspection, and spatial manipulation localization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image

from config.settings import AI_DETECTOR_MODEL
from models.face_detector import FaceDeepfakeDetector
from schemas.result_schema import ModalityScore
from utils.image_utils import (
    calculate_sensor_noise_profile,
    calculate_surface_smoothness,
    compute_error_level_analysis,
    extract_image_metadata,
    generate_manipulation_heatmap,
)
from utils.logging_utils import get_logger

logger = get_logger("ai_image_detector")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "ai_detector.pt"
IMAGE_SIZE = 224


class AIImageDetector:
    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH, hf_model_name: str = AI_DETECTOR_MODEL):
        self.checkpoint_path = Path(checkpoint_path)
        self.hf_model_name = hf_model_name
        self.is_available = True
        self.backend = "multimodal_forensics_ensemble"
        self.status_message = "Ready"
        self.model: Any = None
        self.face_detector = FaceDeepfakeDetector()
        self.class_to_idx: dict[str, int] = {}
        self.transform = None

    def load(self) -> bool:
        """Loads optional neural network weights if available, maintaining graceful fallback."""
        if self.model is not None:
            return True

        if self.checkpoint_path.is_file():
            try:
                import torch
                from torch import nn
                from torchvision import models, transforms

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 2)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)
                model.eval()

                self.model = model
                self.device = device
                self.class_to_idx = checkpoint.get("class_to_idx", {"ai_generated": 0, "real": 1})
                self.transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                logger.info("Loaded local PyTorch classifier: %s", self.checkpoint_path.name)
            except Exception as exc:
                logger.warning("Could not load local weights: %s", exc)

        return True

    def predict(self, image_path: str | Path) -> Dict[str, Any]:
        """
        Deep forensic evaluation across:
        1. Sensor noise residual (PRNU / Poisson shot noise vs latent denoising)
        2. Bilateral surface smoothness (waxy synthetic textures)
        3. Generative aspect ratio & dimension profiling (1024x1024 square)
        4. EXIF provenance & AI software signatures
        5. Facial deepfake and skin texture metrics
        6. Error Level Analysis (ELA)
        7. Spatial manipulation anomaly heatmap & area percentage
        """
        self.load()

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return {
                "is_available": False,
                "error": "Could not read image file.",
                "ai_percentage": 0.0,
                "real_percentage": 0.0,
                "undecided_percentage": 100.0,
                "confidence": 0.0,
                "prediction": "UNDECIDED",
            }

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Forensic Signals
        noise_mean, noise_std = calculate_sensor_noise_profile(gray)
        smoothness = calculate_surface_smoothness(gray)
        metadata = extract_image_metadata(image_path)
        ela = compute_error_level_analysis(image_path)
        face_analysis = self.face_detector.analyze_faces(img_bgr)
        heatmap_res = generate_manipulation_heatmap(image_path)

        # 2. Evidence Fusion Weights
        ai_evidence = 0.05
        real_evidence = 0.05
        cues_detected = []

        # (a) Sensor Noise: Real cameras have physical photon shot noise (> 2.5)
        # Latent diffusion models (Gemini, SD, Midjourney) denoise smoothly (< 1.3)
        if noise_mean < 1.15:
            ai_evidence += 0.35
            cues_detected.append(f"Synthetic latent space denoising detected (noise residual: {noise_mean:.2f})")
        elif noise_mean < 1.70:
            ai_evidence += 0.20
            cues_detected.append(f"Abnormally low sensor shot noise (noise residual: {noise_mean:.2f})")
        elif noise_mean > 3.20:
            real_evidence += 0.35
            cues_detected.append(f"Natural optical sensor shot noise detected ({noise_mean:.2f})")

        # (b) Bilateral Surface Texture: Human skin/fabrics have micro-pore texture (> 4.0)
        if smoothness < 2.30:
            ai_evidence += 0.30
            cues_detected.append(f"Synthetic bilateral surface over-smoothing detected (index: {smoothness:.2f})")
        elif smoothness > 4.50:
            real_evidence += 0.30
            cues_detected.append(f"Natural fine-grained surface micro-textures preserved ({smoothness:.2f})")

        # (c) Standard Generative Dimensions
        is_square_gen = (w in (512, 768, 1024, 1536, 2048) and h in (512, 768, 1024, 1536, 2048))
        if is_square_gen and not metadata.get("has_exif"):
            ai_evidence += 0.15
            cues_detected.append(f"Characteristic generative square geometry ({w}x{h}) without camera hardware EXIF")
        elif metadata.get("camera_make") and metadata.get("camera_model"):
            real_evidence += 0.25
            cues_detected.append(f"Authentic camera hardware provenance ({metadata.get('camera_make')} {metadata.get('camera_model')})")

        # (d) Explicit AI Generator Signature in Metadata
        if metadata.get("ai_signature_found"):
            ai_evidence += 0.45
            cues_detected.append(f"Provenance match: {metadata.get('signature_details')}")

        # (e) Facial Deepfake Inspection
        if face_analysis.get("faces_detected", 0) > 0:
            face_score = face_analysis.get("facial_ai_confidence", 0.0)
            if face_score >= 0.70:
                ai_evidence += 0.25
                cues_detected.append(f"Face analysis indicates synthetic waxy skin shading ({face_analysis.get('deepfake_risk')})")
            elif face_score <= 0.30:
                real_evidence += 0.20
                cues_detected.append("Face presents natural pore grain and optical boundary gradients")

        # (f) Optional Neural Classifier Vote (if loaded)
        if self.model is not None:
            try:
                import torch
                pil_img = Image.open(image_path).convert("RGB")
                inputs = self.transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    probs = torch.softmax(self.model(inputs), dim=1)[0]
                ai_idx = self.class_to_idx.get("ai_generated", 0)
                neural_ai_p = float(probs[ai_idx].item())
                # Add moderate weight to neural model
                if neural_ai_p > 0.75:
                    ai_evidence += 0.15
                elif neural_ai_p < 0.25:
                    real_evidence += 0.15
            except Exception:
                pass

        # 3. Probability Normalization & Uncertainty Estimation
        total_ev = ai_evidence + real_evidence
        raw_ai = ai_evidence / total_ev if total_ev > 0 else 0.5
        raw_real = real_evidence / total_ev if total_ev > 0 else 0.5

        # Uncertainty is proportional to ambiguity (close gap between AI and Real)
        gap = abs(raw_ai - raw_real)
        undecided_pct = max(3.0, (1.0 - gap) * 20.0)
        remaining = 100.0 - undecided_pct

        ai_pct = round((raw_ai / (raw_ai + raw_real)) * remaining, 1)
        real_pct = round((raw_real / (raw_ai + raw_real)) * remaining, 1)
        undecided_pct = round(100.0 - (ai_pct + real_pct), 1)

        if ai_pct >= 58.0:
            prediction = "LIKELY AI-GENERATED"
        elif real_pct >= 58.0:
            prediction = "LIKELY REAL"
        else:
            prediction = "UNDECIDED"

        return {
            "is_available": True,
            "backend": self.backend,
            "prediction": prediction,
            "confidence": round(max(raw_ai, raw_real), 2),
            "ai_percentage": ai_pct,
            "real_percentage": real_pct,
            "undecided_percentage": undecided_pct,
            "ai_spatial_area_pct": heatmap_res.get("ai_spatial_area_pct", 0.0),
            "heatmap_rgb": heatmap_res.get("heatmap_rgb"),
            "forensic_cues": cues_detected,
            "forensic_metrics": {
                "noise_residual_mean": round(noise_mean, 3),
                "surface_smoothness": round(smoothness, 3),
                "is_square_gen": is_square_gen,
                "ela_mean_error": ela.get("mean_error", 0.0),
            },
            "facial_analysis": face_analysis,
            "metadata_forensics": metadata,
        }

    def predict_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Fast prediction for a single BGR video frame."""
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            noise_mean, _ = calculate_sensor_noise_profile(gray)
            smoothness = calculate_surface_smoothness(gray)

            score = 0.5
            if noise_mean < 1.30 and smoothness < 2.50:
                score = 0.85
            elif noise_mean > 3.00 and smoothness > 4.00:
                score = 0.15

            label = "LIKELY AI-GENERATED" if score >= 0.65 else ("LIKELY REAL" if score <= 0.35 else "UNDECIDED")
            return {"label": label, "ai_prob": score, "real_prob": 1.0 - score}
        except Exception:
            return {"label": "UNDECIDED", "ai_prob": 0.5, "real_prob": 0.5}
