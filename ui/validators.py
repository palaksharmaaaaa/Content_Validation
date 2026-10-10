"""
ui.validators: Unified multi-modal validation and URL ingestion for Streamlit UI.
Provides:
1. File format, container, and size validation across Image, Video, and Audio.
2. Provenance dispatch to the per-modality validators (C2PA marker presence only; no cryptographic verification).
3. Safe URL validation, platform detection, and remote streaming media ingestion.
Completely self-contained with zero external directory dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from PIL import Image

import audio_detector.config as audio_cfg
import image_detector.config as image_cfg
import video_detector.config as video_cfg
from audio_detector import AudioValidator
from core.security import SAFE_MAX_IMAGE_PIXELS, SecureUrlFetcher, validate_secure_url
from image_detector.validator import ImageValidator
from video_detector.validator import VideoValidator

logger = logging.getLogger("ui.validators")

Image.MAX_IMAGE_PIXELS = SAFE_MAX_IMAGE_PIXELS

# Supported extensions and size limits come from each package's own config: one definition, so the app can never accept a file
# its engine rejects (or reject one the documentation says is supported).
SUPPORTED_IMAGE_EXTENSIONS = set(image_cfg.SUPPORTED_EXTENSIONS)
SUPPORTED_VIDEO_EXTENSIONS = set(video_cfg.SUPPORTED_EXTENSIONS)
SUPPORTED_AUDIO_EXTENSIONS = set(audio_cfg.SUPPORTED_EXTENSIONS)
MAX_FILE_SIZE_MB_BY_TYPE = {"image": image_cfg.MAX_FILE_SIZE_MB, "video": video_cfg.MAX_FILE_SIZE_MB, "audio": audio_cfg.MAX_FILE_SIZE_MB}

PLATFORM_DOMAINS = {
    "Instagram": [
        "instagram.com",
        "www.instagram.com",
    ],
    "YouTube": [
        "youtube.com",
        "www.youtube.com",
        "youtu.be",
        "www.youtu.be",
    ],
    "Facebook": [
        "facebook.com",
        "www.facebook.com",
        "fb.watch",
    ],
    "TikTok": [
        "tiktok.com",
        "www.tiktok.com",
    ],
    "X": [
        "x.com",
        "www.x.com",
        "twitter.com",
        "www.twitter.com",
    ],
}


def get_file_extension(filename: str | Path) -> str:
    """Lower-case extension including the dot."""
    return Path(filename).suffix.lower()


def detect_media_type(file_path: str | Path) -> str:
    """'image', 'video' or 'audio' from the extension, or 'unknown'."""
    extension = get_file_extension(file_path)
    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if extension in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    if extension in SUPPORTED_AUDIO_EXTENSIONS:
        return "audio"
    return "unknown"


def _validation_result(validator_factory: Callable[[], Any], limit_mb: float, file_path: str | Path, kind: str) -> Dict[str, Any]:
    """Run one package validator and package its verdict for the UI; any failure is a clear "not readable" result."""
    try:
        val_res = validator_factory().validate(file_path)
        return {
            "readable": val_res.valid,
            "format_valid": val_res.valid,
            "size_valid": val_res.file_size_mb <= limit_mb,
            "error": val_res.error,
            "details": val_res.to_dict(),
        }
    except Exception as exc:
        logger.debug("validate_%s_file error: %s", kind, exc)
        return {"readable": False, "format_valid": False, "size_valid": False, "error": str(exc)}


def validate_image_file(file_path: str | Path) -> Dict[str, Any]:
    """Check an image file with the image validator (limit from ``image_detector.config``)."""
    limit = MAX_FILE_SIZE_MB_BY_TYPE["image"]
    return _validation_result(lambda: ImageValidator(max_file_size_mb=limit), limit, file_path, "image")


def validate_video_file(file_path: str | Path) -> Dict[str, Any]:
    """Check a video file with the video validator (limit from ``video_detector.config``)."""
    limit = MAX_FILE_SIZE_MB_BY_TYPE["video"]
    return _validation_result(lambda: VideoValidator(max_file_size_mb=limit), limit, file_path, "video")


def validate_audio_file(file_path: str | Path) -> Dict[str, Any]:
    """Check an audio file with the audio validator (limit from ``audio_detector.config``)."""
    limit = MAX_FILE_SIZE_MB_BY_TYPE["audio"]
    return _validation_result(lambda: AudioValidator(max_size_mb=limit), limit, file_path, "audio")


def validate_file(file_path: str | Path) -> Dict[str, Any]:
    """Detect the media type and validate with the matching package validator."""
    media_type = detect_media_type(file_path)
    if media_type == "image":
        result = validate_image_file(file_path)
    elif media_type == "video":
        result = validate_video_file(file_path)
    elif media_type == "audio":
        result = validate_audio_file(file_path)
    else:
        result = {
            "readable": False,
            "format_valid": False,
            "size_valid": False,
            "error": "Unsupported file format.",
        }
    result["media_type"] = media_type
    return result


def analyze_provenance(file_path: str | Path) -> Dict[str, Any]:
    """Provenance for any supported media file, via the owning package's validator (single source of truth).

    The result carries the nested ``c2pa`` / ``exif`` / ``provenance_verdict`` view plus the package's flat keys.
    C2PA is marker presence only (never cryptographically verified).
    """
    media_type = detect_media_type(file_path)
    if media_type == "image":
        from image_detector.provenance import ImageProvenanceValidator as Validator
    elif media_type == "video":
        from video_detector.provenance import VideoProvenanceValidator as Validator
    else:
        from audio_detector.provenance import AudioProvenanceValidator as Validator
    return Validator().analyze_provenance(file_path)


def normalize_domain(domain: str) -> str:
    """Lower-case host without a leading ``www.``."""
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def detect_platform(url: str) -> str:
    """Which social platform a link belongs to (Instagram, YouTube, Facebook, TikTok, X), or 'Unknown'."""
    try:
        parsed = urlparse(url)
        domain = normalize_domain(parsed.netloc)
        if not domain:
            return "Unknown"
        for platform, domains in PLATFORM_DOMAINS.items():
            normalized_domains = [normalize_domain(d) for d in domains]
            if domain in normalized_domains:
                return platform
        return "Unknown"
    except Exception as exc:
        logger.debug("detect_platform: ignored %s: %s", type(exc).__name__, exc)
        return "Unknown"


def validate_url(url: str) -> Dict[str, Any]:
    """Validates URL using OWASP anti-SSRF protections and identifies platform domain."""
    result = {
        "valid_url": False,
        "platform": "Unknown",
        "message": "",
    }
    if not url:
        result["message"] = "URL is empty."
        return result

    valid, msg, _ = validate_secure_url(url)
    if not valid:
        result["message"] = msg
        return result

    platform = detect_platform(url)
    result["valid_url"] = True
    result["platform"] = platform
    result["message"] = f"Valid {platform} URL." if platform != "Unknown" else "Valid direct web link."
    return result


def validate_expected_platform(url: str, expected_platform: str) -> Dict[str, Any]:
    """Check that a link is well formed and, if a platform is expected, that it matches."""
    result = validate_url(url)
    if not result["valid_url"]:
        return {**result, "platform_match": False}

    detected_platform = result["platform"]
    if expected_platform == "Auto":
        return {**result, "platform_match": True}

    platform_match = (detected_platform == expected_platform)
    if platform_match:
        message = f"URL matches expected platform ({expected_platform})."
    else:
        message = f"Platform mismatch: expected {expected_platform}, but URL belongs to {detected_platform}."

    return {
        **result,
        "platform_match": platform_match,
        "message": message,
    }


def fetch_media_from_url(
    url: str, expected_type: str = "image", max_mb: Optional[int] = None, dest_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Downloads remote media via anti-SSRF SecureUrlFetcher, enforcing the size limit of the media type (or ``max_mb``).

    Pass the session's scratch directory as ``dest_dir`` so the sidebar wipe removes the download."""
    limit = float(max_mb if max_mb is not None else MAX_FILE_SIZE_MB_BY_TYPE.get(expected_type, image_cfg.MAX_FILE_SIZE_MB))
    fetcher = SecureUrlFetcher(max_mb=limit, timeout_seconds=120)
    fetch_res = fetcher.fetch(url, dest_dir=dest_dir, expected_type=expected_type)
    if not fetch_res.get("success"):
        return {"success": False, "error": fetch_res.get("error", "Download failed")}

    platform = detect_platform(url)
    return {
        "success": True,
        "file_path": fetch_res["file_path"],
        "filename": fetch_res["filename"],
        "size_mb": fetch_res["size_mb"],
        "content_type": fetch_res["content_type"],
        "platform": platform if platform != "Unknown" else "Direct Link",
    }
