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


def calculate_blur_score(gray_image):
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def calculate_brightness(gray_image):
    return float(np.mean(gray_image))


def calculate_extreme_pixel_ratio(gray_image):
    dark_pixels = np.sum(gray_image <= 5)
    bright_pixels = np.sum(gray_image >= 250)

    total_pixels = gray_image.size

    extreme_pixels = dark_pixels + bright_pixels

    return float(extreme_pixels / total_pixels)


def analyze_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return {
            "valid": False,
            "error": "Image could not be read."
        }

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness = calculate_brightness(gray)

    variance = float(np.var(gray))

    extreme_ratio = calculate_extreme_pixel_ratio(gray)

    blur_score = calculate_blur_score(gray)

    is_blank = (
        variance <= BLANK_VARIANCE_THRESHOLD
        or extreme_ratio >= BLANK_EXTREME_PIXEL_RATIO
    )

    is_low_resolution = (
        width < MIN_IMAGE_WIDTH
        or height < MIN_IMAGE_HEIGHT
    )

    is_blurry = blur_score < BLUR_THRESHOLD

    is_too_dark = brightness < DARK_MEAN_THRESHOLD

    is_too_bright = brightness > BRIGHT_MEAN_THRESHOLD

    if is_blank:
        quality = "INVALID"

    elif is_low_resolution:
        quality = "POOR"

    elif is_blurry:
        quality = "POOR"

    elif is_too_dark or is_too_bright:
        quality = "POOR"

    else:
        quality = "GOOD"

    return {
        "valid": not is_blank,
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "variance": round(variance, 2),
        "extreme_pixel_ratio": round(extreme_ratio, 4),
        "blur_score": round(blur_score, 2),
        "is_blank": is_blank,
        "is_blurry": is_blurry,
        "is_low_resolution": is_low_resolution,
        "is_too_dark": is_too_dark,
        "is_too_bright": is_too_bright,
        "quality": quality,
    }