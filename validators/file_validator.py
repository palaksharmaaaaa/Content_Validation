from pathlib import Path
from PIL import Image
import cv2

from config.settings import (
    MAX_FILE_SIZE_MB,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from utils.file_utils import (
    get_file_extension,
    get_file_size_mb,
)
from utils.audio_utils import extract_audio_samples


def detect_media_type(file_path: str):
    extension = get_file_extension(file_path)

    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"

    if extension in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"

    if extension in SUPPORTED_AUDIO_EXTENSIONS:
        return "audio"

    return "unknown"


def validate_image_file(file_path: str):
    result = {
        "readable": False,
        "format_valid": False,
        "size_valid": False,
        "error": None,
    }

    try:
        size_mb = get_file_size_mb(file_path)

        if size_mb > MAX_FILE_SIZE_MB:
            result["error"] = (
                f"File size {size_mb:.2f} MB exceeds "
                f"{MAX_FILE_SIZE_MB} MB limit."
            )
            return result

        result["size_valid"] = True

        with Image.open(file_path) as image:
            image.verify()

        result["readable"] = True
        result["format_valid"] = True

    except Exception as exc:
        result["error"] = str(exc)

    return result


def validate_video_file(file_path: str):
    result = {
        "readable": False,
        "format_valid": False,
        "size_valid": False,
        "error": None,
    }

    try:
        size_mb = get_file_size_mb(file_path)

        if size_mb > MAX_FILE_SIZE_MB:
            result["error"] = (
                f"File size {size_mb:.2f} MB exceeds "
                f"{MAX_FILE_SIZE_MB} MB limit."
            )
            return result

        result["size_valid"] = True

        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            result["error"] = "Unable to open video."
            return result

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            result["error"] = "Video contains no readable frames."
            return result

        result["readable"] = True
        result["format_valid"] = True

    except Exception as exc:
        result["error"] = str(exc)

    return result


def validate_audio_file(file_path: str):
    result = {
        "readable": False,
        "format_valid": False,
        "size_valid": False,
        "error": None,
    }

    try:
        size_mb = get_file_size_mb(file_path)

        if size_mb > MAX_FILE_SIZE_MB:
            result["error"] = (
                f"File size {size_mb:.2f} MB exceeds "
                f"{MAX_FILE_SIZE_MB} MB limit."
            )
            return result

        result["size_valid"] = True

        samples, sr, dur = extract_audio_samples(file_path)
        if samples is None or len(samples) == 0:
            result["error"] = "Unable to decode audio stream."
            return result

        result["readable"] = True
        result["format_valid"] = True
        result["duration_seconds"] = dur
        result["sample_rate"] = sr

    except Exception as exc:
        result["error"] = str(exc)

    return result


def validate_file(file_path: str):
    media_type = detect_media_type(file_path)

    if media_type == "image":
        validation = validate_image_file(file_path)
    elif media_type == "video":
        validation = validate_video_file(file_path)
    elif media_type == "audio":
        validation = validate_audio_file(file_path)
    else:
        validation = {
            "readable": False,
            "format_valid": False,
            "size_valid": False,
            "error": "Unsupported media format.",
        }

    return {
        "media_type": media_type,
        **validation,
    }