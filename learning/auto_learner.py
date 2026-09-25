"""
End-to-End Online Auto-Learner & Continual Optimizer.
Dynamically recalibrates algorithmic feature weights, sensitivity thresholds,
and triggers online fine-tuning of neural classifier checkpoints based on user feedback.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from utils.logging_utils import get_logger

logger = get_logger("auto_learner")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = PROJECT_ROOT / "data" / "forensic_memory"
CALIBRATION_FILE = MEMORY_DIR / "dynamic_calibration.json"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "ai_detector.pt"
STARTER_DATASET = PROJECT_ROOT / "dataset"


class AutoLearner:
    """Dynamically adapts forensic thresholds and updates model weights end-to-end."""

    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    def load_calibration(self) -> Dict[str, Any]:
        if CALIBRATION_FILE.is_file():
            try:
                with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "feedback_count": 0,
            "feature_weights": {
                "noise_residual": 0.35,
                "surface_smoothness": 0.30,
                "fft_decay": 0.20,
                "facial_shading": 0.25,
                "vocoder_cutoff": 0.40,
                "spectral_flatness": 0.30,
                "silence_ratio": 0.25,
            },
            "sensitivity_offsets": {
                "noise_center_offset": 0.0,
                "smooth_center_offset": 0.0,
                "audio_ai_offset": 0.0,
            },
        }

    def save_calibration(self, calib: Dict[str, Any]) -> None:
        calib["last_updated"] = datetime.now().isoformat()
        try:
            with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
                json.dump(calib, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save dynamic calibration: %s", exc)

    def on_feedback_received(self, record: Dict[str, Any], all_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes new user feedback, updates algorithmic calibration,
        and triggers online gradient updates if applicable.
        """
        calib = self.load_calibration()
        calib["feedback_count"] = len(all_records)

        weights = calib.get("feature_weights", {})
        offsets = calib.get("sensitivity_offsets", {})
        ratings = record.get("ratings", {})
        modality = record.get("modality", "image")

        synth_score = float(ratings.get("synthetic_score", 5.0))
        real_score = float(ratings.get("real_score", 5.0))
        undetected_score = float(ratings.get("undetected_score", 0.0))
        f_metrics = record.get("forensic_metrics", {})

        modifications = []

        # 1. Image & Video Visual Forensics Calibration
        if modality in ("image", "video"):
            if synth_score >= 7.0 or undetected_score >= 6.0:
                # Feedback says it is synthetic or artifacts were undetected/missed
                noise = float(f_metrics.get("noise_residual_mean", 2.2))
                smooth = float(f_metrics.get("surface_smoothness", 3.2))
                alpha = float(f_metrics.get("spectral_decay_alpha", 2.0))

                # Tighten noise residual sensitivity
                if noise < 2.5:
                    weights["noise_residual"] = min(0.60, weights.get("noise_residual", 0.35) + 0.03)
                    offsets["noise_center_offset"] = max(-0.40, offsets.get("noise_center_offset", 0.0) - 0.04)
                    modifications.append("Boosted noise residual sensitivity (latent denoising filter)")

                # Tighten bilateral smoothness
                if smooth < 3.8:
                    weights["surface_smoothness"] = min(0.55, weights.get("surface_smoothness", 0.30) + 0.03)
                    offsets["smooth_center_offset"] = max(-0.50, offsets.get("smooth_center_offset", 0.0) - 0.04)
                    modifications.append("Boosted surface smoothness weight (waxy texture detection)")

                # Tighten 2D FFT spectral decay
                if alpha > 2.3 or alpha < 1.75:
                    weights["fft_decay"] = min(0.50, weights.get("fft_decay", 0.20) + 0.04)
                    modifications.append("Boosted 2D Fourier power slope weight")

            elif real_score >= 7.0:
                # User confirmed media is authentic camera capture -> avoid false positives
                offsets["noise_center_offset"] = min(0.30, offsets.get("noise_center_offset", 0.0) + 0.03)
                offsets["smooth_center_offset"] = min(0.30, offsets.get("smooth_center_offset", 0.0) + 0.03)
                modifications.append("Calibrated baseline noise thresholds to protect authentic camera shots")

        # 2. Audio Acoustic Forensics Calibration
        elif modality == "audio":
            if synth_score >= 7.0:
                has_cutoff = f_metrics.get("has_vocoder_cutoff", False)
                if has_cutoff:
                    weights["vocoder_cutoff"] = min(0.65, weights.get("vocoder_cutoff", 0.40) + 0.04)
                    modifications.append("Strengthened neural vocoder high-frequency cutoff weight")

                flatness = float(f_metrics.get("spectral_flatness", 0.02))
                if flatness < 0.005:
                    weights["spectral_flatness"] = min(0.55, weights.get("spectral_flatness", 0.30) + 0.03)
                    modifications.append("Strengthened Wiener spectral flatness weight")

                offsets["audio_ai_offset"] = min(0.15, offsets.get("audio_ai_offset", 0.0) + 0.02)
            elif real_score >= 7.0:
                offsets["audio_ai_offset"] = max(-0.15, offsets.get("audio_ai_offset", 0.0) - 0.02)
                modifications.append("Relaxed audio synthetic prior for natural speech")

        calib["feature_weights"] = weights
        calib["sensitivity_offsets"] = offsets
        self.save_calibration(calib)

        # 3. Online Neural Checkpoint Fine-Tuning
        retrain_info = self._attempt_online_finetune(all_records)

        return {
            "calibration_updated": True,
            "modifications": modifications,
            "active_weights": weights,
            "sensitivity_offsets": offsets,
            "online_finetune": retrain_info,
        }

    def _attempt_online_finetune(self, all_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Performs quick online fine-tuning on the PyTorch neural network checkpoint
        using feedback samples. Runs non-blocking or lightweight so inference stays ultra-responsive.
        """
        image_records = [
            r for r in all_records
            if r.get("modality") == "image" and (r.get("ground_truth") in ("AI-Generated", "Real / Camera Authentic") or r.get("ratings", {}).get("synthetic_score", 5) >= 8 or r.get("ratings", {}).get("real_score", 5) >= 8)
        ]

        if len(image_records) < 1:
            return {"status": "skipped", "reason": "Insufficient labeled feedback samples for gradient step."}

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model
            from torchvision import models, transforms
            from PIL import Image

            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)

            if CHECKPOINT_PATH.is_file():
                ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])

            model.to(device)
            model.train()

            # Freeze early layers, fine-tune layer4 + fc
            for name, param in model.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Prepare feedback batch
            batch_images = []
            batch_labels = []

            for r in image_records[-8:]:  # train on recent feedback samples
                rel_path = r.get("archived_sample_path")
                if not rel_path:
                    continue
                full_path = PROJECT_ROOT / rel_path
                if not full_path.is_file():
                    continue

                try:
                    with Image.open(full_path) as p_img:
                        t_img = transform(p_img.convert("RGB"))
                    gt = r.get("ground_truth", "")
                    synth_s = r.get("ratings", {}).get("synthetic_score", 5.0)
                    is_ai = (gt == "AI-Generated" or synth_s >= 6.5)
                    label = 0 if is_ai else 1  # 0: ai_generated, 1: real

                    batch_images.append(t_img)
                    batch_labels.append(label)
                except Exception:
                    continue

            if not batch_images:
                return {"status": "skipped", "reason": "No accessible image files in recent feedback."}

            tensor_x = torch.stack(batch_images).to(device)
            tensor_y = torch.tensor(batch_labels, dtype=torch.long).to(device)

            # 2 fast online training steps
            losses = []
            for _ in range(2):
                optimizer.zero_grad()
                outputs = model(tensor_x)
                loss = criterion(outputs, tensor_y)
                loss.backward()
                optimizer.step()
                losses.append(round(loss.item(), 4))

            # Save updated checkpoint
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": {"ai_generated": 0, "real": 1},
                    "architecture": "resnet18",
                    "image_size": 224,
                    "last_online_update": datetime.now().isoformat(),
                    "feedback_samples_trained": len(batch_images),
                },
                CHECKPOINT_PATH,
            )

            logger.info("Online neural checkpoint updated. Loss: %s", losses)
            return {
                "status": "success",
                "samples_trained": len(batch_images),
                "losses": losses,
                "checkpoint": str(CHECKPOINT_PATH.name),
            }

        except Exception as exc:
            logger.warning("Online fine-tune failed gracefully: %s", exc)
            return {"status": "failed", "error": str(exc)}
