"""
Multi-Signal Forensic AI Image & Video Detector.
Combines sensor noise profiling (PRNU), bilateral surface texture, 2D FFT Radial Power Spectrum,
Error Level Analysis (ELA), EXIF provenance inspection, and spatial manipulation localization.
Supports configurable sensitivity modes (Balanced, High, Aggressive).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image

from config.settings import AI_DETECTOR_MODEL
from learning.forensic_memory import ForensicMemory
from models.face_detector import FaceDeepfakeDetector
from schemas.result_schema import ModalityScore
from utils.image_utils import (
    analyze_frequency_domain,
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
        self.memory = ForensicMemory()
        self.class_to_idx: dict[str, int] = {}
        self.transform = None

    def reload(self) -> bool:
        """Forces reload of model weights and calibration from disk."""
        self.model = None
        return self.load()

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

    def predict(self, image_path: str | Path, sensitivity: str = "high") -> Dict[str, Any]:
        """
        Deep forensic evaluation across:
        1. Sensor noise residual (PRNU / Poisson shot noise vs latent denoising)
        2. Bilateral surface smoothness (waxy synthetic textures)
        3. 2D FFT Radial Power Spectrum decay (1/f^alpha law deviation)
        4. Error Level Analysis (ELA) compression discrepancy
        5. Aspect ratio & dimension profiling
        6. EXIF provenance & AI software signatures
        7. Facial deepfake and skin texture metrics
        8. Spatial manipulation anomaly heatmap & area percentage
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

        # 1. Forensic Feature Extraction
        noise_mean, noise_std = calculate_sensor_noise_profile(gray)
        smoothness = calculate_surface_smoothness(gray)
        metadata = extract_image_metadata(image_path)
        ela = compute_error_level_analysis(image_path)
        fft_res = analyze_frequency_domain(image_path)
        face_analysis = self.face_detector.analyze_faces(img_bgr)
        heatmap_res = generate_manipulation_heatmap(image_path)

        # Load dynamic calibration and sensitivity offsets
        calib = self.memory.load_calibration()
        weights = calib.get("feature_weights", {})
        offsets = calib.get("sensitivity_offsets", {})

        w_noise = weights.get("noise_residual", 0.35)
        w_smooth = weights.get("surface_smoothness", 0.30)
        w_fft = weights.get("fft_decay", 0.20)
        w_face = weights.get("facial_shading", 0.25)

        noise_offset = offsets.get("noise_center_offset", 0.0)
        smooth_offset = offsets.get("smooth_center_offset", 0.0)

        # Sensitivity tuning centers with learned offsets
        if sensitivity == "aggressive":
            noise_center = 2.6 + noise_offset
            smooth_center = 3.8 + smooth_offset
            base_ai_bias = 0.15
        elif sensitivity == "high":
            noise_center = 2.3 + noise_offset
            smooth_center = 3.4 + smooth_offset
            base_ai_bias = 0.10
        else:  # balanced
            noise_center = 2.0 + noise_offset
            smooth_center = 3.0 + smooth_offset
            base_ai_bias = 0.05

        # 2. Continuous Probabilistic Signals
        # Sigmoid curve: lower noise residual = higher AI probability
        p_noise_ai = float(1.0 / (1.0 + np.exp((noise_mean - noise_center) * 2.2)))
        # Sigmoid curve: lower bilateral texture diff = higher AI probability
        p_smooth_ai = float(1.0 / (1.0 + np.exp((smoothness - smooth_center) * 1.4)))
        p_fft_ai = float(fft_res.get("p_fft_ai", 0.5))

        cues_detected = []
        ai_evidence = base_ai_bias
        real_evidence = 0.05

        # (a) Sensor Noise Residual
        if p_noise_ai > 0.65:
            ai_evidence += (p_noise_ai * w_noise)
            cues_detected.append(f"Synthetic latent space denoising detected (sensor noise residual: {noise_mean:.2f})")
        elif p_noise_ai < 0.30:
            real_evidence += ((1.0 - p_noise_ai) * w_noise)
            cues_detected.append(f"Natural optical sensor shot noise preserved ({noise_mean:.2f})")

        # (b) Surface Texture Smoothness
        if p_smooth_ai > 0.65:
            ai_evidence += (p_smooth_ai * w_smooth)
            cues_detected.append(f"Synthetic bilateral surface over-smoothing detected (index: {smoothness:.2f})")
        elif p_smooth_ai < 0.30:
            real_evidence += ((1.0 - p_smooth_ai) * w_smooth)
            cues_detected.append(f"Natural fine-grained surface micro-textures preserved ({smoothness:.2f})")

        # (c) FFT Radial Power Spectrum
        alpha = fft_res.get("spectral_decay_alpha", 2.0)
        if fft_res.get("is_anomalous_decay"):
            ai_evidence += w_fft
            cues_detected.append(f"2D Fourier power spectrum anomaly (alpha: {alpha:.2f}, departs from natural 1/f^2 distribution)")
        else:
            real_evidence += (w_fft * 0.75)
            cues_detected.append(f"Natural optical frequency decay (alpha: {alpha:.2f})")

        # (d) Generative Dimensions & Metadata
        is_square_gen = (w in (512, 768, 1024, 1536, 2048) and h in (512, 768, 1024, 1536, 2048))
        if is_square_gen and not metadata.get("has_exif"):
            ai_evidence += 0.15
            cues_detected.append(f"Characteristic generative square geometry ({w}x{h}) without camera hardware EXIF")
        elif metadata.get("camera_make") and metadata.get("camera_model"):
            real_evidence += 0.25
            cues_detected.append(f"Authentic camera hardware provenance ({metadata.get('camera_make')} {metadata.get('camera_model')})")

        # (e) Metadata AI Signatures
        if metadata.get("ai_signature_found"):
            ai_evidence += 0.50
            cues_detected.append(f"Provenance match: {metadata.get('signature_details')}")

        # (f) Facial Deepfake Inspection
        if face_analysis.get("faces_detected", 0) > 0:
            face_score = face_analysis.get("facial_ai_confidence", 0.0)
            if face_score >= 0.65:
                ai_evidence += w_face
                cues_detected.append(f"Facial inspection indicates synthetic skin shading ({face_analysis.get('deepfake_risk')})")
            elif face_score <= 0.30:
                real_evidence += (w_face * 0.8)
                cues_detected.append("Face presents natural optical pore grain and physical lighting gradients")

        # (g) Forensic Memory Bank Retrieval & Learned Prior Injection
        forensic_snapshot = {
            "forensic_metrics": {
                "noise_residual_mean": noise_mean,
                "surface_smoothness": smoothness,
                "spectral_decay_alpha": alpha,
                "ela_mean_error": ela.get("mean_error", 0.0),
            }
        }
        mem_res = self.memory.query_similar_media(image_path, modality="image", forensic_data=forensic_snapshot)
        if mem_res.get("has_matches"):
            ai_boost = mem_res["prior_adjustment"]["ai_boost"]
            real_boost = mem_res["prior_adjustment"]["real_boost"]
            if ai_boost > 0:
                ai_evidence += ai_boost
            if real_boost > 0:
                real_evidence += real_boost
            cues_detected.append(f"🧠 {mem_res['explanation']}")

        # 3. Probability Normalization & Uncertainty Estimation
        total_ev = ai_evidence + real_evidence
        raw_ai = ai_evidence / total_ev if total_ev > 0 else 0.5
        raw_real = real_evidence / total_ev if total_ev > 0 else 0.5

        gap = abs(raw_ai - raw_real)
        undecided_pct = max(3.0, (1.0 - gap) * 20.0)
        remaining = 100.0 - undecided_pct

        ai_pct = round((raw_ai / (raw_ai + raw_real)) * remaining, 1)
        real_pct = round((raw_real / (raw_ai + raw_real)) * remaining, 1)
        undecided_pct = round(100.0 - (ai_pct + real_pct), 1)

        threshold = 50.0 if sensitivity in ("high", "aggressive") else 58.0
        if ai_pct >= threshold:
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
                "spectral_decay_alpha": round(alpha, 3),
                "is_square_gen": is_square_gen,
                "ela_mean_error": ela.get("mean_error", 0.0),
            },
            "facial_analysis": face_analysis,
            "metadata_forensics": metadata,
            "memory_match": mem_res,
        }

    def predict_frame(self, frame_bgr: np.ndarray, sensitivity: str = "high") -> Dict[str, Any]:
        """
        Fast prediction for a single video frame, calibrated for MP4/H264 video compression.
        Video compression adds ~0.8 baseline high-frequency noise, which is compensated for.
        """
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            noise_mean, _ = calculate_sensor_noise_profile(gray)
            smoothness = calculate_surface_smoothness(gray)

            # In compressed video, compensate for macroblock quantization noise
            comp_noise = max(0.2, noise_mean - 0.70)
            noise_thresh = 2.4 if sensitivity in ("high", "aggressive") else 2.0
            smooth_thresh = 3.6 if sensitivity in ("high", "aggressive") else 3.1

            p_noise_ai = float(1.0 / (1.0 + np.exp((comp_noise - noise_thresh) * 2.0)))
            p_smooth_ai = float(1.0 / (1.0 + np.exp((smoothness - smooth_thresh) * 1.3)))
            score = (p_noise_ai * 0.55) + (p_smooth_ai * 0.45)

            thresh = 0.50 if sensitivity in ("high", "aggressive") else 0.60
            label = "LIKELY AI-GENERATED" if score >= thresh else ("LIKELY REAL" if score <= 0.35 else "UNDECIDED")
            return {"label": label, "ai_prob": round(score, 3), "real_prob": round(1.0 - score, 3)}
        except Exception:
            return {"label": "UNDECIDED", "ai_prob": 0.5, "real_prob": 0.5}
