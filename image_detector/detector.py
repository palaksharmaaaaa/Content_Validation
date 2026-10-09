"""
image_detector.detector: Complete, self-contained Image AI Detection engine.
Combines:
1. Local PyTorch neural classifier backbone (models/ai_detector.pt).
2. Physical camera sensor noise residual analysis (PRNU / Poisson shot noise).
3. Bilateral filter surface smoothness & plastic skin texture detection.
4. 2D FFT Radial Power Spectrum decay (1/f^alpha field law).
5. Error Level Analysis (ELA) compression footprint discrepancy.
6. Localized spatial manipulation heatmaps.
7. Rule-based feedback calibration (see learner.py -- adjusts scoring constants, not model weights).
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
from core.imageio import imread
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
    EXIF_CONTRADICTION_LR,
    SMALL_IMAGE_MAX_SIDE,
    DIGITAL_ART_LR_SCALE,
    EXIF_TRUSTED_CREDIT,
    EXIF_UNTRUSTED_CREDIT,
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
    calculate_image_epistemic_uncertainty,
    evaluate_taxonomy_classification,
    normalize_percentages,
    pool_bayesian_log_odds,
)

logger = logging.getLogger("image_detector.detector")


_PREDICTION_BY_TAXONOMY = {
    ImageTaxonomyState.FULLY_AI_GENERATED: "LIKELY AI-GENERATED",
    ImageTaxonomyState.PROCEDURAL_CGI_SYNTHETIC: "PROCEDURAL CGI SYNTHETIC",
    ImageTaxonomyState.AI_ENHANCED_COMPOSITE: "AI-ENHANCED / COMPOSITE",
    ImageTaxonomyState.AUTHENTIC_RECAPTURED_SCREEN: "AUTHENTIC (RECAPTURED SCREEN)",
    ImageTaxonomyState.AUTHENTIC_EDITED: "AUTHENTIC (CONVENTIONALLY EDITED)",
    ImageTaxonomyState.AUTHENTIC_SCREENSHOT: "AUTHENTIC SCREENSHOT",
    ImageTaxonomyState.AI_ENHANCED_SCREENSHOT: "AI-ENHANCED SCREENSHOT",
    ImageTaxonomyState.AI_GENERATED_SCREENSHOT: "AI-GENERATED SCREENSHOT",
}


@dataclass
class _Signals:
    """Raw outputs of every feature extractor for one image."""

    img_bgr: np.ndarray
    path: Path
    meta: Dict[str, Any]
    noise_mean: float
    smoothness: float
    fft: Dict[str, Any]
    watermark: Dict[str, Any]
    cutout: Dict[str, Any]
    scanned: Dict[str, Any]
    art: Dict[str, Any]
    screenshot: Dict[str, Any]
    inpainting: Dict[str, Any]
    screen_recapture: Dict[str, Any]
    spectral: Dict[str, Any]
    heatmap: Dict[str, Any]


@dataclass
class _Terms:
    """Accumulated base-10 log-odds terms and the human-readable cues explaining them."""

    lrs: Dict[str, float] = field(default_factory=dict)
    cues: List[str] = field(default_factory=list)


class ImageAIDetector:
    """
    Completely independent, self-contained Image AI Detector with rule-based feedback calibration (see learner.py).
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
        self._load_lock = threading.RLock()

    def load(self) -> bool:
        """Loads once; concurrent callers wait for the first load instead of loading the model twice."""
        with self._load_lock:
            return self._load_locked()

    def _load_locked(self) -> bool:
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

    # ------------------------------------------------------------------ input decoding
    @staticmethod
    def _decode_input(image_path: Any) -> Tuple[Optional[np.ndarray], Path, Any]:
        """Decodes any supported input into (BGR array, nominal path, source for metadata extraction)."""
        path = Path("in_memory.png")
        if isinstance(image_path, np.ndarray):
            if image_path.ndim == 2:
                img = cv2.cvtColor(image_path, cv2.COLOR_GRAY2BGR)
            elif image_path.shape[2] == 4:
                img = cv2.cvtColor(image_path, cv2.COLOR_BGRA2BGR)
            else:
                img = image_path.copy()
            return img, path, img
        if isinstance(image_path, Image.Image):
            rgb = np.array(image_path.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), path, image_path
        if isinstance(image_path, (bytes, bytearray)):
            img = cv2.imdecode(np.frombuffer(image_path, np.uint8), cv2.IMREAD_COLOR)
            return img, path, io.BytesIO(image_path)
        if hasattr(image_path, "read"):
            content = image_path.read()
            img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            return img, Path(getattr(image_path, "name", "in_memory.png")), io.BytesIO(content)
        path = Path(image_path)
        return imread(str(path)), path, path

    @staticmethod
    def _extract_signals(img_bgr: np.ndarray, path: Path, raw_meta_source: Any) -> _Signals:
        """Runs every forensic feature extractor once."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        noise_mean, _noise_std = calculate_sensor_noise_profile(gray)
        _ela_mean, ela_map = compute_ela(img_bgr)
        meta = extract_image_metadata(raw_meta_source)
        return _Signals(
            img_bgr=img_bgr, path=path, meta=meta,
            noise_mean=noise_mean,
            smoothness=calculate_surface_smoothness(gray),
            fft=analyze_fft_radial_power_spectrum(gray),
            watermark=detect_ai_watermark(path, img_bgr),
            cutout=detect_background_cutout(path, img_bgr),
            scanned=detect_scanned_photo(path, img_bgr, meta),
            art=detect_digital_art_and_painting(path, img_bgr, metadata=meta),
            screenshot=detect_screenshot(path, img_bgr, meta),
            inpainting=detect_inpainting_and_manipulation(path, img_bgr, ela_map),
            screen_recapture=detect_screen_rephotography_moire(path, img_bgr, meta),
            spectral=detect_spectral_modality(path, img_bgr),
            heatmap=generate_spatial_manipulation_heatmap(img_bgr),
        )

    # ------------------------------------------------------------------ evidence terms
    @staticmethod
    def _metadata_terms(sig: _Signals, provenance: Optional[Dict[str, Any]], ev: _Terms) -> Tuple[bool, bool]:
        """Camera/AI-signature/C2PA metadata terms. Returns (has_camera_hardware, metadata_absent)."""
        meta = sig.meta
        has_camera = bool(meta.get("camera_make") and meta.get("camera_model"))
        if meta.get("ai_signature_found"):
            ev.lrs["exif_ai_signature"] = 3.2
            ev.cues.append(meta.get("signature_details", "AI signature in metadata"))
        elif meta.get("ai_enhancer_signature_found"):
            ev.lrs["ai_enhancer_signature"] = 1.2
            ev.cues.append(meta.get("signature_details", "AI enhancement signature"))
        elif has_camera:
            # Credit is finalized in _pixel_terms, once the physical evidence is known (EXIF is forgeable).
            ev.lrs["exif_hardware"] = EXIF_TRUSTED_CREDIT

        if meta.get("graphic_editor_signature_found"):
            ev.cues.append(meta.get("signature_details", "Graphic editing tool detected"))

        # C2PA: only marker PRESENCE is detected (no certificate chain or hash binding is validated), and markers can
        # be copied into any file, so presence is reported but never scored as protective evidence.
        c2pa_present = bool(provenance and provenance.get("c2pa_present"))
        if c2pa_present:
            ev.cues.append("C2PA Content Credentials markers present (presence only; not cryptographically verified, no score credit).")

        # Explicit neutrality for the most common real-world case (downloaded / re-shared / stripped images): zero weight,
        # but flagged so the taxonomy classifier requires corroborating pixel evidence before routing to FULLY_AI_GENERATED.
        metadata_absent = (
            not has_camera and not meta.get("ai_signature_found")
            and not meta.get("ai_enhancer_signature_found") and not c2pa_present
        )
        if metadata_absent:
            ev.lrs["metadata_absent"] = 0.0
            ev.cues.append(
                "No camera hardware tags, AI/enhancer signature, or C2PA manifest found -- "
                "metadata is absent or stripped. Treated as neutral, not as evidence of synthesis "
                "(this is the normal state for downloaded, re-shared, or platform-processed images)."
            )
        return has_camera, metadata_absent

    @staticmethod
    def _detector_flag_terms(sig: _Signals, has_camera: bool, ev: _Terms) -> None:
        """Terms from the discrete detectors (watermark, scan, inpainting, screenshot, art, cutout)."""
        scanned = sig.scanned.get("is_scanned")
        if sig.watermark.get("watermark_detected"):
            ev.lrs["ai_watermark"] = 3.5
            ev.cues.append(sig.watermark["details"])
        if scanned:
            ev.lrs["scanned_print"] = -2.2
            ev.cues.append(sig.scanned["details"])
        # Uneven local noise is a statistic of the sensor grain, which a thumbnail no longer has: on the real small-photo sets the
        # finding fires on 6-20 % of photographs but on only 4 % of AI faces, so below SMALL_IMAGE_MAX_SIDE it is not evidence.
        if sig.inpainting.get("is_manipulated") and max(sig.img_bgr.shape[:2]) > SMALL_IMAGE_MAX_SIDE:
            ev.lrs["inpainting_splicing"] = 1.8 * float(sig.inpainting.get("confidence", 0.7))
            ev.cues.append(sig.inpainting["details"])
        if sig.screenshot.get("is_screenshot"):
            ev.cues.append(sig.screenshot["details"])
        if sig.art.get("is_digital_art") and not has_camera and not scanned:
            ev.lrs["digital_art_synthesis"] = DIGITAL_ART_LR_SCALE * float(sig.art.get("confidence", 0.8))
            ev.cues.append(sig.art["details"])
        if sig.cutout.get("is_cutout"):
            ev.cues.append(sig.cutout["details"])

    @staticmethod
    def _gaussian_llr(x: float, mu_real: float, s_real: float, mu_ai: float, s_ai: float) -> float:
        """Base-10 log-likelihood ratio (AI vs real) under two Gaussians, clipped to +/-3."""
        lr = (np.log10(s_real / s_ai)
              - ((x - mu_ai) ** 2) / (2 * (s_ai ** 2) * np.log(10))
              + ((x - mu_real) ** 2) / (2 * (s_real ** 2) * np.log(10)))
        return float(np.clip(lr, -3.0, 3.0))

    def _pixel_terms(self, sig: _Signals, has_camera: bool, weights: Dict[str, float], offsets: Dict[str, float], ev: _Terms) -> Tuple[float, float]:
        """Sensor-noise and surface-smoothness terms, including the unauthenticated-EXIF trust policy. Returns raw (lr_noise, lr_smooth)."""
        w_noise, w_smooth = weights["noise"], weights["smooth"]
        meta, scanned, art = sig.meta, sig.scanned.get("is_scanned"), sig.art.get("is_digital_art")

        effective_noise = sig.art.get("flat_noise_mean", sig.noise_mean) if art else sig.noise_mean
        lr_noise = self._gaussian_llr(effective_noise, REAL_NOISE_MU, REAL_NOISE_SIGMA,
                                      AI_NOISE_MU + offsets.get("noise_center_offset", 0.0), AI_NOISE_SIGMA)
        # Raw smoothness evidence is computed up-front so camera EXIF can be checked against the physical evidence
        # before it is allowed to discount it.
        lr_smooth = self._gaussian_llr(sig.smoothness, REAL_SMOOTH_MU, REAL_SMOOTH_SIGMA,
                                       AI_SMOOTH_MU + offsets.get("smooth_center_offset", 0.0), AI_SMOOTH_SIGMA)

        # Downscaled images lose their sensor grain whatever made them, so at thumbnail size "no noise" and "very
        # smooth" say nothing about synthesis: they may still argue for a real photo, but never for AI.
        h_px, w_px = sig.img_bgr.shape[:2]
        if max(h_px, w_px) <= SMALL_IMAGE_MAX_SIDE and (lr_noise > 0.0 or lr_smooth > 0.0):
            lr_noise, lr_smooth = min(lr_noise, 0.0), min(lr_smooth, 0.0)
            ev.cues.append(f"Image is only {w_px}x{h_px}px: noise and smoothness are not reliable at this size and are not counted towards AI")

        exif_untrusted = bool(has_camera and not scanned and (max(lr_noise, 0.0) + max(lr_smooth, 0.0)) >= EXIF_CONTRADICTION_LR)
        if has_camera and "exif_hardware" in ev.lrs:
            if exif_untrusted:
                ev.lrs["exif_hardware"] = EXIF_UNTRUSTED_CREDIT
                ev.cues.append(
                    f"Camera EXIF present ({meta['camera_make']} {meta['camera_model']}) but CONTRADICTED by strong synthetic "
                    "pixel evidence; unauthenticated metadata is treated as untrusted (possible forged EXIF)."
                )
            else:
                ev.cues.append(f"Genuine camera hardware detected ({meta['camera_make']} {meta['camera_model']})")
        discount = (has_camera and not exif_untrusted) or scanned

        if discount:
            lr_noise = min(0.15, lr_noise)
        if art:
            lr_noise = max(0.15, lr_noise)
        ev.lrs["sensor_noise"] = lr_noise * w_noise
        if lr_noise > 0.3:
            ev.cues.append(f"Synthetic latent space denoising detected (PRNU residual: {sig.noise_mean:.2f})")
        elif lr_noise < -0.3 and not art:
            ev.cues.append(f"Natural optical camera sensor shot noise preserved ({sig.noise_mean:.2f})")

        if discount:
            lr_smooth = min(0.10, lr_smooth)
        if art:
            lr_smooth = max(0.20, lr_smooth)
        ev.lrs["surface_smoothness"] = lr_smooth * w_smooth
        if lr_smooth > 0.3:
            ev.cues.append(f"Synthetic bilateral surface over-smoothing detected (index: {sig.smoothness:.2f})")
        elif lr_smooth < -0.3 and not art:
            ev.cues.append(f"Natural fine-grained surface micro-textures preserved ({sig.smoothness:.2f})")
        return lr_noise, lr_smooth

    @staticmethod
    def _fft_term(sig: _Signals, has_camera: bool, w_fft: float, ev: _Terms) -> Tuple[Optional[float], bool]:
        """2D FFT radial power-spectrum decay term. Returns (alpha, is_anomaly); alpha is None when no spectrum was measured."""
        alpha = sig.fft.get("spectral_decay_alpha")
        if alpha is None:
            return None, False
        alpha = float(alpha)
        is_jpeg = str(sig.path).lower().endswith((".jpg", ".jpeg"))
        if has_camera or is_jpeg or sig.scanned.get("is_scanned"):
            z_fft = (alpha - FFT_DECAY_ALPHA_JPEG) / 0.35
            is_anomaly = (alpha > 3.45 or alpha < 1.65)
        else:
            z_fft = (alpha - FFT_DECAY_ALPHA_OPTICAL) / 0.25
            is_anomaly = sig.fft.get("is_anomalous_decay") or abs(z_fft) > 2.0
        if is_anomaly:
            lr_fft = float(np.clip(abs(z_fft) * 0.4, 0.2, 1.5))
            ev.cues.append(f"Abnormal Fourier spectral decay alpha ({alpha:.2f})")
        else:
            lr_fft = -0.4
        ev.lrs["fft_power_decay"] = lr_fft * w_fft
        return alpha, bool(is_anomaly)

    @staticmethod
    def _canonical_term(sig: _Signals, has_camera: bool, ev: _Terms) -> Tuple[bool, bool]:
        """Aspect ratio / canonical generator canvas term. Returns (is_canonical_gen, is_square_gen)."""
        h, w = sig.img_bgr.shape[:2]
        is_canonical = (w, h) in CANONICAL_RESOLUTIONS
        is_square = (w == h and w in (512, 768, 1024, 1536, 2048))
        if (is_canonical or is_square) and not has_camera and not sig.scanned.get("is_scanned"):
            ev.lrs["canonical_dimensions"] = 0.45
            ev.cues.append(f"Canonical AI generative canvas geometry: {w}x{h}")
        return is_canonical, is_square

    def _neural_term(self, img_bgr: np.ndarray, ev: _Terms) -> Optional[float]:
        """Neural backbone log-odds term (None when no checkpoint is loaded or inference fails)."""
        if self.model is None or self.transform is None:
            return None
        try:
            p_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            tensor = self.transform(p_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(tensor), dim=1)[0]
                p_ai_neural = float(probs[0].item())
            lr_neural = float(np.clip(np.log10(p_ai_neural / max(1e-4, 1.0 - p_ai_neural)), -2.0, 2.0))
            ev.lrs["neural_backbone"] = lr_neural * 0.50
            return lr_neural
        except Exception as e:
            logger.debug("Neural inference bypassed: %s", e)
            return None

    @staticmethod
    def _dimension_terms(extra_log_lrs: Optional[Dict[str, float]], ev: _Terms) -> None:
        """Capped dimension-check terms (already clamped by core.forensics.registry.build_report)."""
        for dim_key, dim_val in (extra_log_lrs or {}).items():
            if dim_val:
                ev.lrs[f"dim_{dim_key}"] = float(dim_val)
                ev.cues.append(f"Dimension check '{dim_key}' adjusted log-odds by {float(dim_val):+.2f}")

    @staticmethod
    def _calibrated_weights(calib: Dict[str, Any]) -> Dict[str, float]:
        w = calib.get("feature_weights", {})
        parts = {
            "noise": w.get("noise_residual", 0.35), "smooth": w.get("surface_smoothness", 0.30),
            "fft": w.get("fft_decay", 0.20), "ela": w.get("ela_discrepancy", 0.15),
        }
        total = sum(parts.values())
        return {k: v / total for k, v in parts.items()} if total > 0 else parts

    # ------------------------------------------------------------------ public API
    def predict(
        self,
        image_path: Union[str, Path, bytes, bytearray, io.BytesIO, np.ndarray, Image.Image],
        sensitivity: str = "balanced",
        face_boxes: Optional[List[Dict[str, int]]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        extra_log_lrs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Deep forensic evaluation across physical, spectral, and neural modalities.
        ``extra_log_lrs``: optional capped base-10 log-odds terms from the dimension checks
        (see image_detector.dimension_checks); None/empty leaves behavior unchanged.
        Natively accepts file paths, raw bytes, io.BytesIO, numpy arrays, or PIL Images in RAM.
        """
        self.load()
        img_bgr, path, raw_meta_source = self._decode_input(image_path)
        if img_bgr is None or img_bgr.size == 0:
            return ImageForensicResult(
                is_available=False, backend=self.backend, prediction="UNDECIDED", label="UNDECIDED", confidence=0.0,
                ai_percentage=0.0, real_percentage=0.0, undecided_percentage=100.0,
                error="Could not read image file or in-memory stream.",
            ).to_dict()

        sig = self._extract_signals(img_bgr, path, raw_meta_source)
        calib = self.self_improver.load_calibration()
        weights = self._calibrated_weights(calib)
        offsets = calib.get("sensitivity_offsets", {})

        ev = _Terms()
        has_camera, metadata_absent = self._metadata_terms(sig, provenance, ev)
        self._detector_flag_terms(sig, has_camera, ev)
        lr_noise, lr_smooth = self._pixel_terms(sig, has_camera, weights, offsets, ev)
        alpha, fft_anomaly = self._fft_term(sig, has_camera, weights["fft"], ev)
        is_canonical_gen, is_square_gen = self._canonical_term(sig, has_camera, ev)
        lr_neural = self._neural_term(img_bgr, ev)

        # Independent pixel signals that individually lean synthetic: the taxonomy classifier's corroboration gate
        # for metadata_absent images (a single ambiguous signal must not route an EXIF-stripped real photo to FULLY_AI).
        synthetic_signal_count = sum([lr_noise > 0.3, lr_smooth > 0.3, fft_anomaly, bool(lr_neural is not None and lr_neural > 0.3)])
        self._dimension_terms(extra_log_lrs, ev)

        total_log_odds, prob_ai_raw, prob_real_raw = pool_bayesian_log_odds(sensitivity, ev.lrs)
        target_undecided = max(3.0, min(24.0, calculate_image_epistemic_uncertainty(prob_ai_raw) * 20.0))
        ai_pct, real_pct, undecided_pct = normalize_percentages(
            ai_val=prob_ai_raw * 100.0, real_val=prob_real_raw * 100.0,
            undecided_val=target_undecided, min_undecided=3.0, decimals=1,
        )

        tax_state, tax_label, tax_desc, tax_reasons = evaluate_taxonomy_classification(
            ai_pct=ai_pct, real_pct=real_pct, metadata=sig.meta,
            watermark_data=sig.watermark, cutout_data=sig.cutout, scanned_data=sig.scanned,
            art_data=sig.art, screenshot_data=sig.screenshot,
            inpainting_data=sig.inpainting, screen_recapture_data=sig.screen_recapture,
            noise_mean=sig.noise_mean, smoothness=sig.smoothness,
            is_square_gen=is_square_gen, is_canonical_gen=is_canonical_gen,
            metadata_absent=metadata_absent, synthetic_signal_count=synthetic_signal_count,
        )
        prediction = _PREDICTION_BY_TAXONOMY.get(tax_state, "LIKELY REAL")
        visual_medium = sig.art.get("visual_medium", "Photographic Capture")
        sensor_spectrum = sig.spectral.get("sensor_spectrum", "Visible light (colour RGB)")

        return ImageForensicResult(
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
            subject_genre=None,                       # the genre comes from the content analysis, not from the detector
            visual_medium=visual_medium,
            sensor_spectrum=sensor_spectrum,
            document_layout="None (Standard Visual Content)",
            watermark_detected=sig.watermark["watermark_detected"],
            watermark_details=sig.watermark.get("details"),
            background_cutout_detected=sig.cutout["is_cutout"],
            scanned_photo_detected=sig.scanned["is_scanned"],
            screen_recapture_detected=sig.screen_recapture.get("is_screen_recapture", False),
            screen_recapture_details=sig.screen_recapture,
            neural_enhancer_detected=sig.meta.get("ai_enhancer_signature_found", False),
            digital_art_detected=sig.art.get("is_digital_art", False),
            screenshot_detected=sig.screenshot.get("is_screenshot", False),
            screenshot_details=sig.screenshot,
            inpainting_detected=sig.inpainting.get("is_manipulated", False),
            inpainting_details=sig.inpainting,
            log_likelihood_ratios={k: round(v, 3) for k, v in ev.lrs.items()},
            posterior_log_odds=round(total_log_odds, 3),
            ai_spatial_area_pct=sig.heatmap.get("ai_spatial_area_pct", 0.0),
            heatmap_rgb=sig.heatmap.get("heatmap_rgb"),
            forensic_cues=ev.cues,
            forensic_metrics={
                "noise_residual_mean": round(sig.noise_mean, 3),
                "surface_smoothness": round(sig.smoothness, 3),
                "spectral_decay_alpha": None if alpha is None else round(alpha, 3),
                "is_canonical_gen": is_canonical_gen,
                "is_cutout": sig.cutout["is_cutout"],
                "is_scanned": sig.scanned["is_scanned"],
                "is_screen_recaptured": sig.screen_recapture.get("is_screen_recapture", False),
                "sensor_spectrum": sensor_spectrum,
                "visual_medium": visual_medium,
                "is_digital_art": sig.art.get("is_digital_art", False),
                "is_screenshot": sig.screenshot.get("is_screenshot", False),
                "screenshot_device": sig.screenshot.get("device_type", "None"),
                "screenshot_orientation": sig.screenshot.get("orientation", "None"),
                "is_inpainted_or_composite": sig.inpainting.get("is_manipulated", False),
            },
        ).to_dict()

    predict_image = predict

    def predict_frame(self, frame_bgr: np.ndarray, sensitivity: str = "balanced") -> Dict[str, Any]:
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
            return {"label": label, "prediction": label, "ai_prob": round(score, 3), "real_prob": round(1.0 - score, 3),
                    "frame_noise": round(float(noise_mean), 3)}
        except Exception as exc:
            logger.warning("predict_frame encountered exception: %s", exc)
            return {"label": "UNDECIDED", "prediction": "UNDECIDED", "ai_prob": 0.5, "real_prob": 0.5}
