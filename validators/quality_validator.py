"""
Quality validation utilities for image and frame inspection.
"""
from typing import Dict, Any
import cv2
import numpy as np

from config.settings import (
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
    BLANK_VARIANCE_THRESHOLD,
    BLANK_EXTREME_PIXEL_RATIO,
    BLUR_THRESHOLD,
    DARK_MEAN_THRESHOLD,
    BRIGHT_MEAN_THRESHOLD,
)


def evaluate_visual_quality(gray_img: np.ndarray, width: int, height: int) -> Dict[str, Any]:
    """
    Evaluates visual quality of a grayscale image or frame.
    """
    variance = float(np.var(gray_img))
    mean_val = float(np.mean(gray_img))
    blur_score = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

    dark_pixels = np.sum(gray_img <= 5)
    bright_pixels = np.sum(gray_img >= 250)
    total_pixels = gray_img.size
    extreme_ratio = float((dark_pixels + bright_pixels) / total_pixels) if total_pixels > 0 else 1.0

    is_blank = (
        variance <= BLANK_VARIANCE_THRESHOLD
        or extreme_ratio >= BLANK_EXTREME_PIXEL_RATIO
    )
    is_low_res = width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT
    is_blurry = blur_score < BLUR_THRESHOLD
    is_too_dark = mean_val < DARK_MEAN_THRESHOLD
    is_too_bright = mean_val > BRIGHT_MEAN_THRESHOLD

    if is_blank:
        status = "INVALID"
    elif is_low_res or is_blurry or is_too_dark or is_too_bright:
        status = "POOR"
    else:
        status = "GOOD"

    return {
        "status": status,
        "is_blank": is_blank,
        "is_blurry": is_blurry,
        "is_low_resolution": is_low_res,
        "is_too_dark": is_too_dark,
        "is_too_bright": is_too_bright,
        "blur_score": round(blur_score, 2),
        "brightness": round(mean_val, 2),
        "variance": round(variance, 2),
        "extreme_pixel_ratio": round(extreme_ratio, 4),
    }
