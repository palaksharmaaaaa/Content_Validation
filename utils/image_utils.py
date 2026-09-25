"""
Forensic image utilities: EXIF metadata inspection, Error Level Analysis (ELA),
2D FFT Radial Power Spectrum analysis, sensor noise residual profiling, and spatial manipulation heatmaps.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS


KNOWN_AI_SOFTWARE_SIGNATURES = [
    "midjourney",
    "stable diffusion",
    "dall-e",
    "novelai",
    "adobe firefly",
    "comfyui",
    "civitai",
    "automatic1111",
    "invokeai",
    "bing image creator",
    "imagen",
    "gemini",
    "flux",
    "runway",
    "kling",
    "sora",
    "pika",
    "luma",
]


def extract_image_metadata(image_path: str | Path) -> Dict[str, Any]:
    """
    Extracts EXIF metadata, camera hardware traces, and searches for AI generator tags.
    """
    metadata_info = {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "date_time": None,
        "ai_signature_found": False,
        "signature_details": None,
        "raw_tags": {},
    }

    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return metadata_info

            metadata_info["has_exif"] = True
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                tag_val_str = str(value)
                metadata_info["raw_tags"][tag_name] = tag_val_str

                if tag_name == "Make":
                    metadata_info["camera_make"] = tag_val_str
                elif tag_name == "Model":
                    metadata_info["camera_model"] = tag_val_str
                elif tag_name == "Software":
                    metadata_info["software"] = tag_val_str
                elif tag_name == "DateTime":
                    metadata_info["date_time"] = tag_val_str

                val_lower = tag_val_str.lower()
                for sig in KNOWN_AI_SOFTWARE_SIGNATURES:
                    if sig in val_lower:
                        metadata_info["ai_signature_found"] = True
                        metadata_info["signature_details"] = f"Found '{sig}' in {tag_name}"

    except Exception:
        pass

    return metadata_info


def compute_error_level_analysis(
    image_path: str | Path, quality: int = 90, scale: int = 10
) -> Dict[str, Any]:
    """
    Error Level Analysis (ELA):
    Re-compresses the image at a known JPEG quality and measures compression delta.
    Uniform low error or localized compression anomalies indicate synthetic generation.
    """
    try:
        with Image.open(image_path) as original:
            original = original.convert("RGB")

            buffer = io.BytesIO()
            original.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            resaved = Image.open(buffer)

            diff = ImageChops.difference(original, resaved)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema]) if extrema else 0
            scale_factor = 255.0 / max_diff if max_diff > 0 else 1.0

            diff_enhanced = ImageEnhance.Brightness(diff).enhance(scale_factor)
            diff_array = np.array(diff, dtype=np.float32)
            mean_error = float(np.mean(diff_array))
            std_error = float(np.std(diff_array))

            return {
                "success": True,
                "mean_error": round(mean_error, 2),
                "std_error": round(std_error, 2),
                "max_diff": int(max_diff),
                "ela_image": diff_enhanced,
            }
    except Exception as exc:
        return {"success": False, "error": str(exc), "mean_error": 0.0, "std_error": 0.0}


def analyze_frequency_domain(image_path: str | Path) -> Dict[str, Any]:
    """
    Performs 2D Fast Fourier Transform (FFT) analysis and computes the radial power spectrum decay (alpha).
    Natural images exhibit power spectral density P(f) ~ 1 / f^alpha where alpha is typically in [1.8, 2.3].
    Generative diffusion models depart from this physical distribution due to convolutional upsampling and latent decoding.
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"success": False, "error": "Unable to read image."}

        h, w = img.shape
        dft = np.fft.fft2(img)
        dft_shift = np.fft.fftshift(dft)
        psd2D = np.abs(dft_shift) ** 2

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        r_max = min(cy, cx)

        radial_prof = np.bincount(r.ravel(), psd2D.ravel())[:r_max] / np.maximum(1, np.bincount(r.ravel())[:r_max])
        freqs = np.arange(1, len(radial_prof))

        # Fit log(P) = -alpha * log(f) + c
        log_f = np.log(freqs)
        log_p = np.log(radial_prof[1:] + 1e-12)
        fit_len = min(len(log_f), r_max // 2)
        if fit_len > 15:
            slope, _ = np.polyfit(log_f[5:fit_len], log_p[5:fit_len], 1)
            alpha = float(-slope)
        else:
            alpha = 2.0

        is_anomalous_decay = (alpha < 1.75 or alpha > 2.55)
        p_fft_ai = 0.85 if is_anomalous_decay else 0.20

        return {
            "success": True,
            "spectral_decay_alpha": round(alpha, 3),
            "is_anomalous_decay": is_anomalous_decay,
            "p_fft_ai": round(p_fft_ai, 2),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "spectral_decay_alpha": 2.0, "is_anomalous_decay": False, "p_fft_ai": 0.5}


def calculate_sensor_noise_profile(gray_img: np.ndarray) -> Tuple[float, float]:
    """
    Measures sensor shot noise residuals. Real optical cameras produce physical Poisson/Gaussian noise.
    Diffusion models denoise in latent space, yielding abnormally low high-frequency residuals.
    """
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    noise_residual = cv2.absdiff(gray_img, blurred)
    return float(np.mean(noise_residual)), float(np.std(noise_residual))


def calculate_surface_smoothness(gray_img: np.ndarray) -> float:
    """
    Measures bilateral texture preservation. Diffusion models produce unnatural waxy skin and plastic surfaces.
    """
    bilateral = cv2.bilateralFilter(gray_img, 9, 75, 75)
    diff = cv2.absdiff(gray_img, bilateral)
    return float(np.mean(diff))


def generate_manipulation_heatmap(image_path: str | Path) -> Dict[str, Any]:
    """
    Produces a spatial anomaly heatmap visualizing where generative artifacts and noise discrepancies occur.
    Also calculates the spatial proportion (% of pixels identified as synthetic).
    """
    try:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return {"success": False, "error": "Unable to read image"}

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_diff = cv2.absdiff(gray, blurred)

        norm_noise = cv2.normalize(noise_diff, None, 0, 255, cv2.NORM_MINMAX)
        inverted = 255 - norm_noise
        heatmap_color = cv2.applyColorMap(inverted, cv2.COLORMAP_INFERNO)
        overlay_bgr = cv2.addWeighted(img_bgr, 0.55, heatmap_color, 0.45, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        anomaly_mask = noise_diff < 3
        ai_spatial_area_pct = float((np.sum(anomaly_mask) / float(gray.size)) * 100.0)

        return {
            "success": True,
            "ai_spatial_area_pct": round(ai_spatial_area_pct, 1),
            "heatmap_rgb": overlay_rgb,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "ai_spatial_area_pct": 0.0, "heatmap_rgb": None}
