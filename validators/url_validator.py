"""
URL validation and remote media fetching for social platforms and direct media links.
"""
from pathlib import Path
import tempfile
from typing import Any, Dict
from urllib.parse import urlparse
import requests

from config.settings import (
    MAX_FILE_SIZE_MB,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)


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


def normalize_domain(domain: str) -> str:
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def detect_platform(url: str) -> str:
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
    except Exception:
        return "Unknown"


def validate_url(url: str) -> Dict[str, Any]:
    result = {
        "valid_url": False,
        "platform": "Unknown",
        "message": "",
    }

    if not url:
        result["message"] = "URL is empty."
        return result

    url = url.strip()

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            result["message"] = "URL must start with http:// or https://."
            return result

        if not parsed.netloc:
            result["message"] = "Invalid URL hostname."
            return result

    except Exception:
        result["message"] = "Malformed URL structure."
        return result

    platform = detect_platform(url)
    result["valid_url"] = True
    result["platform"] = platform
    result["message"] = f"Valid {platform} URL." if platform != "Unknown" else "Valid direct web link."
    return result


def validate_expected_platform(url: str, expected_platform: str) -> Dict[str, Any]:
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
    url: str, expected_type: str = "image", max_mb: int = MAX_FILE_SIZE_MB
) -> Dict[str, Any]:
    """
    Downloads remote media from direct URL, enforcing size limits and format checks.
    Saves to a temporary file and returns path and metadata.
    """
    val_res = validate_url(url)
    if not val_res["valid_url"]:
        return {"success": False, "error": val_res["message"]}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
    }

    parsed = urlparse(url)
    path_suffix = Path(parsed.path).suffix.lower()

    # Determine default extension based on expected type
    if expected_type == "image":
        allowed_exts = SUPPORTED_IMAGE_EXTENSIONS
        default_suffix = ".jpg"
    elif expected_type == "video":
        allowed_exts = SUPPORTED_VIDEO_EXTENSIONS
        default_suffix = ".mp4"
    else:
        allowed_exts = SUPPORTED_AUDIO_EXTENSIONS
        default_suffix = ".mp3"

    suffix = path_suffix if path_suffix in allowed_exts else default_suffix

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=15)
        response.raise_for_status()

        # Check content length header if provided
        cl = response.headers.get("content-length")
        if cl:
            size_mb = int(cl) / (1024 * 1024)
            if size_mb > max_mb:
                return {
                    "success": False,
                    "error": f"File size ({size_mb:.1f} MB) exceeds maximum limit of {max_mb} MB.",
                }

        content_type = response.headers.get("content-type", "").lower()

        # Write to temporary file in chunks
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_path = tmp_file.name
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    if (downloaded / (1024 * 1024)) > max_mb:
                        Path(temp_path).unlink(missing_ok=True)
                        return {
                            "success": False,
                            "error": f"Download aborted: size exceeded {max_mb} MB limit.",
                        }

        final_size_mb = Path(temp_path).stat().st_size / (1024 * 1024)
        return {
            "success": True,
            "file_path": temp_path,
            "filename": Path(parsed.path).name or f"downloaded_{expected_type}{suffix}",
            "size_mb": round(final_size_mb, 2),
            "content_type": content_type,
            "platform": val_res.get("platform", "Direct Link"),
        }

    except Exception as exc:
        return {"success": False, "error": f"Failed to download media: {str(exc)}"}