"""
image_detector.validator: Standalone image validation & quality assurance module.
Performs:
1. Container and format integrity validation.
2. Dimensions, aspect ratio, and color space evaluation.
3. Visual quality assessment (Laplacian sharpness, exposure, dynamic range, contrast).
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging

from pathlib import Path
from typing import Any, Dict

import cv2
from core.imageio import imread
import numpy as np
from PIL import Image, ImageOps

from core.hashing import file_sha256
from image_detector.config import MAX_FILE_SIZE_MB, MIN_RESOLUTION, SUPPORTED_EXTENSIONS
from image_detector.schemas import ImageValidationResult

logger = logging.getLogger("image_detector.validator")

BLANK_STD = 1.0


def _get_file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except Exception as exc:
        logger.debug("_get_file_size_mb: ignored %s: %s", type(exc).__name__, exc)
        return 0.0


def _calculate_file_hash(path: Path) -> str:
    try:
        return file_sha256(path)
    except Exception as exc:  # unreadable file: report an empty hash, but leave a trace
        logger.warning("Could not hash %s: %s", path, exc)
        return ""


class ImageValidator:
    """Independent validator for image integrity, dimensions, and visual quality."""

    def __init__(
        self,
        min_resolution: int = MIN_RESOLUTION,
        max_file_size_mb: float = MAX_FILE_SIZE_MB,
    ):
        self.min_resolution = min_resolution
        self.max_file_size_mb = max_file_size_mb

    def validate(self, image_path: str | Path) -> ImageValidationResult:
        """
        Runs comprehensive image file and quality validation.
        """
        path = Path(image_path)
        if not path.is_file():
            return ImageValidationResult(
                valid=False,
                error=f"Image file not found: {path}",
            )

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return ImageValidationResult(
                valid=False,
                filename=path.name,
                error=f"Unsupported image file extension: {path.suffix}",
            )

        size_mb = _get_file_size_mb(path)
        if size_mb > self.max_file_size_mb:
            return ImageValidationResult(
                valid=False,
                filename=path.name,
                file_size_mb=round(size_mb, 3),
                error=f"Image file size ({size_mb:.2f}MB) exceeds limit of {self.max_file_size_mb}MB",
            )

        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                w, h = img.size
                format_name = img.format or path.suffix.replace(".", "").upper()
                mode = img.mode
        except Exception as exc:
            return ImageValidationResult(
                valid=False,
                filename=path.name,
                error=f"Corrupted or unreadable image file: {exc}",
            )

        if w < self.min_resolution or h < self.min_resolution:
            return ImageValidationResult(
                valid=False,
                filename=path.name,
                width=w,
                height=h,
                error=f"Image dimensions ({w}x{h}) below minimum {self.min_resolution}px threshold.",
            )

        # Quality analysis
        img_bgr = imread(str(path))
        if img_bgr is None:
            return ImageValidationResult(
                valid=False,
                filename=path.name,
                width=w,
                height=h,
                error="Failed to decode image raster with OpenCV.",
            )

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(np.mean(gray))
        lum_std = float(np.std(gray))

        # A picture whose channels barely vary (a single flat colour) has nothing to analyse; below ordinary sensor noise (std < 1 of 255).
        is_blank = float(np.std(img_bgr)) < BLANK_STD
        is_blurry = laplacian_var < 80.0
        is_underexposed = mean_lum < 35.0
        is_overexposed = mean_lum > 225.0
        is_low_contrast = lum_std < 25.0

        quality_score = 1.0
        if is_blurry:
            quality_score -= 0.30
        if is_underexposed or is_overexposed:
            quality_score -= 0.25
        if is_low_contrast:
            quality_score -= 0.20
        quality_score = max(0.1, min(1.0, quality_score))

        return ImageValidationResult(
            valid=True,
            filename=path.name,
            width=w,
            height=h,
            aspect_ratio=round(float(w) / max(1, h), 3),
            format=format_name,
            color_mode=mode,
            file_size_mb=round(size_mb, 3),
            file_hash_sha256=_calculate_file_hash(path),
            quality={
                "sharpness_laplacian": round(laplacian_var, 2),
                "mean_luminance": round(mean_lum, 2),
                "contrast_std": round(lum_std, 2),
                "is_blank": is_blank,
                "is_blurry": is_blurry,
                "is_underexposed": is_underexposed,
                "is_overexposed": is_overexposed,
                "quality_score": round(quality_score, 2),
            },
        )



def analyze_image(image_path: str | Path) -> Dict[str, Any]:
    """Inspects visual quality and dimensions of an image, returning flat dictionary."""
    res = ImageValidator().validate(image_path)
    d = res.to_dict()
    q = d.get("quality", {})
    if isinstance(q, dict):
        d.update(q)
    return d
