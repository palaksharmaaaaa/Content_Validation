"""
image_detector.detector: Complete, self-contained Image AI Detection engine.
Combines:
1. Local PyTorch neural classifier backbone (models/ai_detector.pt).
2. Physical camera sensor noise residual analysis (PRNU / Poisson shot noise).
3. Bilateral filter surface smoothness & plastic skin texture detection.
4. 2D FFT Radial Power Spectrum decay (1/f^alpha field law).
5. Error Level Analysis (ELA) compression footprint discrepancy.
6. Localized spatial manipulation heatmaps.
7. Adaptive online self-improver feedback calibration.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from image_detector.config import (
    AI_NOISE_MU,
    AI_NOISE_SIGMA,
    AI_SMOOTH_MU,
    AI_SMOOTH_SIGMA,
    CANONICAL_RESOLUTIONS,
    DEFAULT_CHECKPOINT,
    FFT_DECAY_ALPHA_JPEG,
    FFT_DECAY_ALPHA_OPTICAL,
    IMAGE_SIZE,
    REAL_NOISE_MU,
    REAL_NOISE_SIGMA,
    REAL_SMOOTH_MU,
    REAL_SMOOTH_SIGMA,
)
from image_detector.features import (
    analyze_fft_radial_power_spectrum,
    calculate_sensor_noise_profile,
    calculate_surface_smoothness,
    compute_ela,
    detect_ai_watermark,
    detect_background_cutout,
    detect_digital_art_and_painting,
    detect_face_swap_artifacts,
    detect_inpainting_and_manipulation,
    detect_scanned_photo,
    detect_screen_rephotography_moire,
    detect_screenshot,
    detect_spectral_modality,
    extract_image_metadata,
    generate_spatial_manipulation_heatmap,
)
from image_detector.learner import ImageSelfImprover
from image_detector.models.backbone import build_image_classifier
from image_detector.schemas import ImageForensicResult, ImageTaxonomyState
from image_detector.scoring import (
    calculate_epistemic_uncertainty,
    evaluate_image_decision,
    evaluate_taxonomy_classification,
    normalize_percentages,
    pool_bayesian_log_odds,
)

logger = logging.getLogger("image_detector.detector")


class ImageAIDetector:
    """
    Completely independent, self-contained, and self-improving Image AI Detector.
    Evaluates physical sensor noise, bilateral smoothness, FFT spectral decay,
    and neural latent fingerprints within a calibrated Bayesian log-odds framework.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        self_improver: Optional[ImageSelfImprover] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self.self_improver = self_improver or ImageSelfImprover()
        self.model = None
        self.device = None
        self.transform = None
        self.backend = "heuristic_and_statistical"
        self._is_loaded = False

    def load(self) -> bool:
        """Loads neural weights if checkpoint exists, otherwise falls back to pure forensic analysis."""
        if self._is_loaded:
            return True

        if self.checkpoint_path.is_file():
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=True)

                model = build_image_classifier(architecture="resnet18", pretrained=False)
                state_dict = checkpoint.get("model_state_dict", checkpoint)

                # Accommodate both sequential head (fc.1, fc.3) and linear head (fc.weight)
                if any("fc.1" in k for k in state_dict.keys()):
                    model.fc = torch.nn.Sequential(
                        torch.nn.Dropout(p=0.2),
                        torch.nn.Linear(512, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, 2),
                    )
                else:
                    model.fc = torch.nn.Linear(512, 2)

                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()

                self.model = model
                self.device = device
                self.transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                self.backend = "hybrid_deep_learning_and_statistical"
                logger.info("ImageAIDetector loaded local PyTorch classifier: %s", self.checkpoint_path.name)
            except Exception as exc:
                logger.warning("Could not load neural checkpoint, using forensic heuristics: %s", exc)

        self._is_loaded = True
        return True

    def predict(
        self,
        image_path: str | Path,
        sensitivity: str = "high",
        face_boxes: Optional[List[Dict[str, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Deep forensic evaluation across physical, spectral, and neural modalities.
        """
        self.load()
        path = Path(image_path)

        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            return ImageForensicResult(
                is_available=False,
                backend=self.backend,
                prediction="UNDECIDED",
                label="UNDECIDED",
                confidence=0.0,
                ai_percentage=0.0,
                real_percentage=0.0,
                undecided_percentage=100.0,
                error="Could not read image file.",
            ).to_dict()

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Forensic Feature & Anomaly Extraction
        noise_mean, noise_std = calculate_sensor_noise_profile(gray)
        smoothness = calculate_surface_smoothness(gray)
        fft_res = analyze_fft_radial_power_spectrum(gray)
        ela_mean, ela_map = compute_ela(path)
        meta = extract_image_metadata(path)
        watermark_res = detect_ai_watermark(path, img_bgr)
        cutout_res = detect_background_cutout(path, img_bgr)
        scanned_res = detect_scanned_photo(path, img_bgr, meta)
        face_swap_res = detect_face_swap_artifacts(path, img_bgr)
        art_res = detect_digital_art_and_painting(path, img_bgr, metadata=meta)
        screenshot_res = detect_screenshot(path, img_bgr, meta)
        inpainting_res = detect_inpainting_and_manipulation(path, img_bgr, ela_map)
        screen_recapture_res = detect_screen_rephotography_moire(path, img_bgr, meta)
        spectral_res = detect_spectral_modality(path, img_bgr)
        heatmap_res = generate_spatial_manipulation_heatmap(img_bgr)

        # 2. Dynamic Calibration Weights from ImageSelfImprover
        calib = self.self_improver.load_calibration()
        weights = calib.get("feature_weights", {})
        offsets = calib.get("sensitivity_offsets", {})

        w_noise = weights.get("noise_residual", 0.35)
        w_smooth = weights.get("surface_smoothness", 0.30)
        w_fft = weights.get("fft_decay", 0.20)
        w_ela = weights.get("ela_discrepancy", 0.15)
        total_w = w_noise + w_smooth + w_fft + w_ela
        if total_w > 0:
            w_noise /= total_w
            w_smooth /= total_w
            w_fft /= total_w
            w_ela /= total_w

        log_lrs: Dict[str, float] = {}
        cues_detected: List[str] = []

        # (a) Hardware EXIF cue & Provenance
        has_camera_hardware = bool(meta.get("camera_make") and meta.get("camera_model"))
        if meta.get("ai_signature_found"):
            log_lrs["exif_ai_signature"] = 3.2
            cues_detected.append(meta.get("signature_details", "AI signature in metadata"))
        elif meta.get("ai_enhancer_signature_found"):
            log_lrs["ai_enhancer_signature"] = 1.2
            cues_detected.append(meta.get("signature_details", "AI enhancement signature"))
        elif has_camera_hardware:
            log_lrs["exif_hardware"] = -1.4
            cues_detected.append(f"Genuine camera hardware detected ({meta['camera_make']} {meta['camera_model']})")

        if meta.get("graphic_editor_signature_found"):
            cues_detected.append(meta.get("signature_details", "Graphic editing tool detected"))

        # (b) Watermark & Emblem Detection (Gemini sparkle, etc.)
        if watermark_res.get("watermark_detected"):
            log_lrs["ai_watermark"] = 3.5
            cues_detected.append(watermark_res["details"])

        # (c) Scanned Physical Print Detection
        if scanned_res.get("is_scanned"):
            log_lrs["scanned_print"] = -2.2
            cues_detected.append(scanned_res["details"])

        # (d) Face Swap & Neural Manipulation Detection
        if face_swap_res.get("is_face_swap"):
            log_lrs["face_swap"] = 2.2
            cues_detected.append(face_swap_res["details"])

        # (d2) Localized Inpainting & Splicing
        if inpainting_res.get("is_manipulated"):
            log_lrs["inpainting_splicing"] = 1.8 * float(inpainting_res.get("confidence", 0.7))
            cues_detected.append(inpainting_res["details"])

        # (d3) Screenshot Detection
        if screenshot_res.get("is_screenshot"):
            cues_detected.append(screenshot_res["details"])

        # (d4) Digital Art & Synthetic Illustration Detection
        if art_res.get("is_digital_art") and not has_camera_hardware and not scanned_res.get("is_scanned"):
            log_lrs["digital_art_synthesis"] = 2.4 * float(art_res.get("confidence", 0.8))
            cues_detected.append(art_res["details"])

        # (e) Background Cutout Detection
        if cutout_res.get("is_cutout"):
            cues_detected.append(cutout_res["details"])

        # (f) PRNU Sensor Noise Residual
        mu_real_noise, s_real_noise = REAL_NOISE_MU, REAL_NOISE_SIGMA
        mu_ai_noise, s_ai_noise = AI_NOISE_MU + offsets.get("noise_center_offset", 0.0), AI_NOISE_SIGMA
        effective_noise = art_res.get("flat_noise_mean", noise_mean) if art_res.get("is_digital_art") else noise_mean
        lr_noise = (np.log10(s_real_noise / s_ai_noise)
                    - ((effective_noise - mu_ai_noise) ** 2) / (2 * (s_ai_noise ** 2) * np.log(10))
                    + ((effective_noise - mu_real_noise) ** 2) / (2 * (s_real_noise ** 2) * np.log(10)))
        lr_noise = float(np.clip(lr_noise, -3.0, 3.0))
        if has_camera_hardware or scanned_res.get("is_scanned"):
            lr_noise = min(0.15, lr_noise)
        if art_res.get("is_digital_art"):
            lr_noise = max(0.15, lr_noise)

        log_lrs["sensor_noise"] = lr_noise * w_noise
        if lr_noise > 0.3:
            cues_detected.append(f"Synthetic latent space denoising detected (PRNU residual: {noise_mean:.2f})")
        elif lr_noise < -0.3 and not art_res.get("is_digital_art"):
            cues_detected.append(f"Natural optical camera sensor shot noise preserved ({noise_mean:.2f})")

        # (g) Surface Texture Smoothness
        mu_real_sm, s_real_sm = REAL_SMOOTH_MU, REAL_SMOOTH_SIGMA
        mu_ai_sm, s_ai_sm = AI_SMOOTH_MU + offsets.get("smooth_center_offset", 0.0), AI_SMOOTH_SIGMA
        lr_smooth = (np.log10(s_real_sm / s_ai_sm)
                     - ((smoothness - mu_ai_sm) ** 2) / (2 * (s_ai_sm ** 2) * np.log(10))
                     + ((smoothness - mu_real_sm) ** 2) / (2 * (s_real_sm ** 2) * np.log(10)))
        lr_smooth = float(np.clip(lr_smooth, -3.0, 3.0))
        if has_camera_hardware or scanned_res.get("is_scanned"):
            lr_smooth = min(0.10, lr_smooth)
        if art_res.get("is_digital_art"):
            lr_smooth = max(0.20, lr_smooth)

        log_lrs["surface_smoothness"] = lr_smooth * w_smooth
        if lr_smooth > 0.3:
            cues_detected.append(f"Synthetic bilateral surface over-smoothing detected (index: {smoothness:.2f})")
        elif lr_smooth < -0.3 and not art_res.get("is_digital_art"):
            cues_detected.append(f"Natural fine-grained surface micro-textures preserved ({smoothness:.2f})")

        # (h) 2D FFT Radial Power Spectrum Decay
        alpha = float(fft_res.get("spectral_decay_alpha", 2.05))
        is_jpeg = str(path).lower().endswith((".jpg", ".jpeg"))
        if has_camera_hardware or is_jpeg or scanned_res.get("is_scanned"):
            mu_alpha, sigma_alpha = FFT_DECAY_ALPHA_JPEG, 0.35
            z_fft = (alpha - mu_alpha) / sigma_alpha
            is_anomaly = (alpha > 3.45 or alpha < 1.65)
        else:
            mu_alpha, sigma_alpha = FFT_DECAY_ALPHA_OPTICAL, 0.25
            z_fft = (alpha - mu_alpha) / sigma_alpha
            is_anomaly = fft_res.get("is_anomalous_decay") or abs(z_fft) > 2.0

        if is_anomaly:
            lr_fft = float(np.clip(abs(z_fft) * 0.4, 0.2, 1.5))
            cues_detected.append(f"Abnormal Fourier spectral decay alpha ({alpha:.2f})")
        else:
            lr_fft = -0.4
        log_lrs["fft_power_decay"] = lr_fft * w_fft

        # (i) Aspect Ratio & Canonical Dimensions
        is_canonical_gen = (w, h) in CANONICAL_RESOLUTIONS
        is_square_gen = (w == h and w in (512, 768, 1024, 1536, 2048))
        if (is_canonical_gen or is_square_gen) and not has_camera_hardware and not scanned_res.get("is_scanned"):
            log_lrs["canonical_dimensions"] = 0.45
            cues_detected.append(f"Canonical AI generative canvas geometry: {w}x{h}")

        # (j) Neural Model Inference (if available)
        if self.model is not None and self.transform is not None:
            try:
                with Image.open(path) as p_img:
                    tensor = self.transform(p_img.convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    p_ai_neural = float(probs[0].item())
                lr_neural = float(np.clip(np.log10(p_ai_neural / max(1e-4, 1.0 - p_ai_neural)), -2.0, 2.0))
                log_lrs["neural_backbone"] = lr_neural * 0.50
            except Exception as e:
                logger.debug("Neural inference bypassed: %s", e)

        # 3. Bayesian Evidence Pooling & Uncertainty
        total_log_odds, prob_ai_raw, prob_real_raw = pool_bayesian_log_odds(sensitivity, log_lrs)
        uncertainty = calculate_epistemic_uncertainty(prob_ai_raw)
        target_undecided = max(3.0, min(24.0, uncertainty * 20.0))

        ai_pct, real_pct, undecided_pct = normalize_percentages(
            ai_val=prob_ai_raw * 100.0,
            real_val=prob_real_raw * 100.0,
            undecided_val=target_undecided,
            min_undecided=3.0,
            decimals=1,
        )

        # 4. Multi-State Forensic Taxonomy Classification
        tax_state, tax_label, tax_desc, tax_reasons = evaluate_taxonomy_classification(
            ai_pct=ai_pct,
            real_pct=real_pct,
            metadata=meta,
            watermark_data=watermark_res,
            cutout_data=cutout_res,
            scanned_data=scanned_res,
            face_swap_data=face_swap_res,
            art_data=art_res,
            screenshot_data=screenshot_res,
            inpainting_data=inpainting_res,
            screen_recapture_data=screen_recapture_res,
            noise_mean=noise_mean,
            smoothness=smoothness,
            is_square_gen=is_square_gen,
            is_canonical_gen=is_canonical_gen,
        )

        # Align primary decision prediction
        if tax_state == ImageTaxonomyState.FULLY_AI_GENERATED:
            prediction = "LIKELY AI-GENERATED"
        elif tax_state == ImageTaxonomyState.PROCEDURAL_CGI_SYNTHETIC:
            prediction = "PROCEDURAL CGI SYNTHETIC"
        elif tax_state == ImageTaxonomyState.ADVERSARIAL_SPOOF_SYNTHETIC:
            prediction = "ADVERSARIAL SPOOF"
        elif tax_state == ImageTaxonomyState.AI_ENHANCED_COMPOSITE:
            prediction = "AI-ENHANCED / COMPOSITE"
        elif tax_state == ImageTaxonomyState.AUTHENTIC_RECAPTURED_SCREEN:
            prediction = "AUTHENTIC (RECAPTURED SCREEN)"
        elif tax_state == ImageTaxonomyState.AUTHENTIC_EDITED:
            prediction = "AUTHENTIC (CONVENTIONALLY EDITED)"
        elif tax_state == ImageTaxonomyState.AUTHENTIC_SCREENSHOT:
            prediction = "AUTHENTIC SCREENSHOT"
        elif tax_state == ImageTaxonomyState.AI_ENHANCED_SCREENSHOT:
            prediction = "AI-ENHANCED SCREENSHOT"
        elif tax_state == ImageTaxonomyState.AI_GENERATED_SCREENSHOT:
            prediction = "AI-GENERATED SCREENSHOT"
        else:
            prediction = "LIKELY REAL"

        # Determine visual medium & sensor spectrum
        visual_medium = art_res.get("visual_medium", "Photographic Capture")
        sensor_spectrum = spectral_res.get("sensor_spectrum", "Visible Spectrum (Bayer RGB)")

        res = ImageForensicResult(
            is_available=True,
            backend=self.backend,
            prediction=prediction,
            label=prediction,
            confidence=round(max(prob_ai_raw, prob_real_raw), 2),
            ai_percentage=ai_pct,
            real_percentage=real_pct,
            undecided_percentage=undecided_pct,
            taxonomy_state=tax_state,
            taxonomy_label=tax_label,
            taxonomy_description=tax_desc,
            taxonomy_reasons=tax_reasons,
            subject_genre="Unspecified General Scene",
            visual_medium=visual_medium,
            sensor_spectrum=sensor_spectrum,
            document_layout="None (Standard Visual Content)",
            watermark_detected=watermark_res["watermark_detected"],
            watermark_details=watermark_res.get("details"),
            background_cutout_detected=cutout_res["is_cutout"],
            scanned_photo_detected=scanned_res["is_scanned"],
            screen_recapture_detected=screen_recapture_res.get("is_screen_recapture", False),
            screen_recapture_details=screen_recapture_res,
            neural_enhancer_detected=meta.get("ai_enhancer_signature_found", False),
            face_swap_detected=face_swap_res["is_face_swap"],
            digital_art_detected=art_res.get("is_digital_art", False),
            screenshot_detected=screenshot_res.get("is_screenshot", False),
            screenshot_details=screenshot_res,
            inpainting_detected=inpainting_res.get("is_manipulated", False),
            inpainting_details=inpainting_res,
            log_likelihood_ratios={k: round(v, 3) for k, v in log_lrs.items()},
            posterior_log_odds=round(total_log_odds, 3),
            ai_spatial_area_pct=heatmap_res.get("ai_spatial_area_pct", 0.0),
            heatmap_rgb=heatmap_res.get("heatmap_rgb"),
            forensic_cues=cues_detected,
            forensic_metrics={
                "noise_residual_mean": round(noise_mean, 3),
                "surface_smoothness": round(smoothness, 3),
                "spectral_decay_alpha": round(alpha, 3),
                "is_canonical_gen": is_canonical_gen,
                "is_cutout": cutout_res["is_cutout"],
                "is_scanned": scanned_res["is_scanned"],
                "is_screen_recaptured": screen_recapture_res.get("is_screen_recapture", False),
                "sensor_spectrum": sensor_spectrum,
                "visual_medium": visual_medium,
                "is_face_swap": face_swap_res["is_face_swap"],
                "is_digital_art": art_res.get("is_digital_art", False),
                "is_screenshot": screenshot_res.get("is_screenshot", False),
                "screenshot_device": screenshot_res.get("device_type", "None"),
                "screenshot_orientation": screenshot_res.get("orientation", "None"),
                "is_inpainted_or_composite": inpainting_res.get("is_manipulated", False),
            },
        )
        return res.to_dict()

    predict_image = predict

    def predict_frame(self, frame_bgr: np.ndarray, sensitivity: str = "high") -> Dict[str, Any]:
        """Fast frame-level inference for video frames."""
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            noise_mean, _ = calculate_sensor_noise_profile(gray)
            smoothness = calculate_surface_smoothness(gray)

            comp_noise = max(0.2, noise_mean - 0.70)
            noise_thresh = 2.4 if sensitivity in ("high", "aggressive") else 2.0
            smooth_thresh = 3.6 if sensitivity in ("high", "aggressive") else 3.1

            p_noise_ai = float(1.0 / (1.0 + np.exp((comp_noise - noise_thresh) * 2.0)))
            p_smooth_ai = float(1.0 / (1.0 + np.exp((smoothness - smooth_thresh) * 1.3)))
            score = (p_noise_ai * 0.55) + (p_smooth_ai * 0.45)

            thresh = 0.50 if sensitivity in ("high", "aggressive") else 0.60
            label = "LIKELY AI-GENERATED" if score >= thresh else ("LIKELY REAL" if score <= 0.35 else "UNDECIDED")
            return {"label": label, "prediction": label, "ai_prob": round(score, 3), "real_prob": round(1.0 - score, 3)}
        except Exception as exc:
            logger.warning("predict_frame encountered exception: %s", exc)
            return {"label": "UNDECIDED", "prediction": "UNDECIDED", "ai_prob": 0.5, "real_prob": 0.5}
