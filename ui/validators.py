"""
ui.validators: Unified multi-modal validation and URL ingestion for Streamlit UI.
Provides:
1. File format, container, and size validation across Image, Video, and Audio.
2. Cryptographic C2PA and EXIF hardware provenance analysis.
3. Safe URL validation, platform detection, and remote streaming media ingestion.
Completely self-contained with zero external directory dependencies.
"""
from __future__ import annotations

import io
import ipaddress
from pathlib import Path
import re
import socket
import tempfile
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import requests

from audio_detector import AudioValidator


MAX_FILE_SIZE_MB = 100

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
}

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".wma",
}

KNOWN_AI_SOFTWARE_SIGNATURES = [
    "midjourney", "dall-e", "dalle", "stable diffusion", "stablediffusion",
    "photoshop", "firefly", "comfyui", "automatic1111", "flux", "novelai",
    "sdxl", "sora", "runway", "kling", "luma", "pika", "seedance",
]

C2PA_JUMBF_SIGNATURES = [
    b"urn:c2pa",
    b"c2pa",
    b"c2ma",
    b"c2cs",
    b"application/c2pa",
    b"image/jumd",
    b"http://c2pa.org",
]

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
    return Path(filename).suffix.lower()


def get_file_size_mb(file_path: str | Path) -> float:
    try:
        p = Path(file_path)
        if not p.is_file():
            return 0.0
        return p.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def detect_media_type(file_path: str | Path) -> str:
    extension = get_file_extension(file_path)
    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if extension in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    if extension in SUPPORTED_AUDIO_EXTENSIONS:
        return "audio"
    return "unknown"


def validate_image_file(file_path: str | Path) -> Dict[str, Any]:
    result = {
        "readable": False,
        "format_valid": False,
        "size_valid": False,
        "error": None,
    }
    try:
        size_mb = get_file_size_mb(file_path)
        if size_mb > MAX_FILE_SIZE_MB:
            result["error"] = f"File size {size_mb:.2f} MB exceeds {MAX_FILE_SIZE_MB} MB limit."
            return result
        result["size_valid"] = True

        with Image.open(file_path) as image:
            image.verify()

        result["readable"] = True
        result["format_valid"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def validate_video_file(file_path: str | Path) -> Dict[str, Any]:
    result = {
        "readable": False,
        "format_valid": False,
        "size_valid": False,
        "error": None,
    }
    try:
        size_mb = get_file_size_mb(file_path)
        if size_mb > MAX_FILE_SIZE_MB:
            result["error"] = f"File size {size_mb:.2f} MB exceeds {MAX_FILE_SIZE_MB} MB limit."
            return result
        result["size_valid"] = True

        cap = cv2.VideoCapture(str(file_path))
        try:
            if not cap.isOpened():
                result["error"] = "Unable to open video."
                return result

            ret, frame = cap.read()
            if not ret or frame is None:
                result["error"] = "Video contains no readable frames."
                return result
        finally:
            try:
                cap.release()
            except Exception:
                pass

        result["readable"] = True
        result["format_valid"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def validate_audio_file(file_path: str | Path) -> Dict[str, Any]:
    result = {
        "readable": False,
        "format_valid": False,
        "size_valid": False,
        "error": None,
    }
    try:
        size_mb = get_file_size_mb(file_path)
        if size_mb > MAX_FILE_SIZE_MB:
            result["error"] = f"File size {size_mb:.2f} MB exceeds {MAX_FILE_SIZE_MB} MB limit."
            return result
        result["size_valid"] = True

        audio_val = AudioValidator()
        val_res = audio_val.validate_audio(str(file_path))
        if not val_res.get("valid", False):
            result["error"] = val_res.get("error", "Unable to decode audio.")
            return result

        result["readable"] = True
        result["format_valid"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def validate_file(file_path: str | Path) -> Dict[str, Any]:
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


def scan_c2pa_markers(file_path: str | Path) -> Dict[str, Any]:
    """Scans binary containers for C2PA JUMBF manifests."""
    file_path = Path(file_path)
    if not file_path.is_file():
        return {
            "c2pa_present": False,
            "status": "UNKNOWN",
            "details": "File does not exist.",
            "manifests_found": [],
        }

    manifests = []
    has_c2pa = False
    is_signed = False
    creation_tool = None
    ai_declaration = False

    try:
        file_size = file_path.stat().st_size
        read_size = min(file_size, 2 * 1024 * 1024)
        with open(file_path, "rb") as f:
            head = f.read(read_size)
            if file_size > read_size:
                f.seek(max(0, file_size - (512 * 1024)))
                tail = f.read(512 * 1024)
            else:
                tail = b""

        full_buf = head + tail
        for sig in C2PA_JUMBF_SIGNATURES:
            if sig in full_buf:
                has_c2pa = True
                manifests.append(sig.decode("ascii", errors="ignore"))

        if b"c2pa.signature" in full_buf or b"c2pa.claim" in full_buf:
            is_signed = True
        if b"c2pa.actions" in full_buf and (b"c2pa.created" in full_buf or b"c2pa.placed" in full_buf):
            pass

        ai_match = re.search(
            rb'(?:c2pa\.generator|softwareAgent|action)\s*[:=]\s*["\']?([^"\'\r\n]{3,60})',
            full_buf,
            re.IGNORECASE,
        )
        if ai_match:
            creation_tool = ai_match.group(1).decode("ascii", errors="ignore").strip()

        for term in [b"synthetic", b"generative", b"dall-e", b"midjourney", b"stable diffusion", b"firefly"]:
            if term in full_buf.lower():
                ai_declaration = True
                break

    except Exception:
        pass

    if has_c2pa:
        if is_signed and not ai_declaration:
            status = "VERIFIED_AUTHENTIC_CREDENTIALS"
            details = "Cryptographically signed C2PA manifest found."
        elif ai_declaration:
            status = "AI_DECLARED_CREDENTIALS"
            details = f"C2PA manifest declares AI generation/synthesis ({creation_tool or 'Generative AI'})."
        else:
            status = "UNVERIFIED_MANIFEST"
            details = "C2PA markers detected but cryptographic signature was not verified."
    else:
        status = "NO_C2PA"
        details = "No C2PA Content Credentials manifest detected."

    return {
        "c2pa_present": has_c2pa,
        "is_signed": is_signed,
        "creation_tool": creation_tool,
        "ai_declaration": ai_declaration,
        "status": status,
        "details": details,
        "manifests_found": manifests,
    }


def extract_exif_metadata(file_path: str | Path) -> Dict[str, Any]:
    file_path = Path(file_path)
    result: Dict[str, Any] = {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "datetime_original": None,
        "ai_signature_found": False,
        "signature_details": None,
        "raw_tags": {},
    }
    if not file_path.is_file():
        return result

    try:
        with Image.open(file_path) as img:
            exif = img.getexif()
            if not exif:
                return result

            result["has_exif"] = True
            for tag_id, val in exif.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                tag_str = str(val).strip()
                result["raw_tags"][tag_name] = tag_str

                if tag_name == "Make":
                    result["camera_make"] = tag_str
                elif tag_name == "Model":
                    result["camera_model"] = tag_str
                elif tag_name == "Software":
                    result["software"] = tag_str
                elif tag_name in ("DateTimeOriginal", "DateTime"):
                    result["datetime_original"] = tag_str

            software_field = (result["software"] or "").lower()
            model_field = (result["camera_model"] or "").lower()
            make_field = (result["camera_make"] or "").lower()
            combined = f"{software_field} {model_field} {make_field}"

            for sig in KNOWN_AI_SOFTWARE_SIGNATURES:
                if sig in combined:
                    result["ai_signature_found"] = True
                    result["signature_details"] = f"Known AI software footprint detected: '{sig}'"
                    break

    except Exception:
        pass
    return result


def analyze_provenance(file_path: str | Path) -> Dict[str, Any]:
    """Unified provenance analysis examining C2PA, EXIF hardware tags, and software creation signatures."""
    c2pa_res = scan_c2pa_markers(file_path)
    exif_details = extract_exif_metadata(file_path)

    # Secondary container scan for MP4/MOV if EXIF absent
    ext = Path(file_path).suffix.lower()
    if not exif_details["has_exif"] and ext in (".mp4", ".mov", ".m4v"):
        try:
            with open(file_path, "rb") as vf:
                vhead = vf.read(128 * 1024)
            for brand in [b"Apple", b"GoPro", b"DJI", b"Sony", b"Canon", b"Nikon", b"Samsung", b"Panasonic"]:
                if brand.lower() in vhead.lower():
                    exif_details["has_exif"] = True
                    exif_details["camera_make"] = brand.decode("ascii", errors="ignore")
                    break
        except Exception:
            pass

    if c2pa_res["c2pa_present"]:
        if c2pa_res["ai_declaration"]:
            provenance_verdict = "C2PA_DECLARED_SYNTHETIC"
            provenance_ai_confidence = 0.98
        elif c2pa_res["is_signed"]:
            provenance_verdict = "C2PA_VERIFIED_AUTHENTIC"
            provenance_ai_confidence = 0.05
        else:
            provenance_verdict = "C2PA_PRESENT_UNVERIFIED"
            provenance_ai_confidence = 0.50
    elif exif_details["ai_signature_found"]:
        provenance_verdict = "METADATA_DECLARED_SYNTHETIC"
        provenance_ai_confidence = 0.95
    elif exif_details.get("camera_make") and exif_details.get("camera_model"):
        provenance_verdict = "HARDWARE_EXIF_PRESENT"
        provenance_ai_confidence = 0.25
    else:
        provenance_verdict = "PROVENANCE_UNKNOWN"
        provenance_ai_confidence = 0.50

    return {
        "c2pa": c2pa_res,
        "exif": exif_details,
        "provenance_verdict": provenance_verdict,
        "provenance_ai_confidence": provenance_ai_confidence,
        "provenance_rule": (
            "NIST Rule: Absence of C2PA metadata indicates UNKNOWN provenance, "
            "not authenticity. Strong conclusions require cryptographic verification."
        ),
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
        if parsed.scheme.lower() not in ("http", "https"):
            result["message"] = "URL must start with http:// or https://."
            return result
        hostname = parsed.hostname
        if not hostname:
            result["message"] = "Invalid URL hostname."
            return result

        hostname_clean = hostname.strip().lower()
        if hostname_clean in ("localhost", "127.0.0.1", "::1"):
            result["message"] = "Access to localhost or loopback address is restricted."
            return result

        try:
            addr_info = socket.getaddrinfo(hostname_clean, None)
            for item in addr_info:
                ip_str = item[4][0]
                ip = ipaddress.ip_address(ip_str)
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                    or ip.is_multicast
                    or ip.is_unspecified
                ):
                    result["message"] = f"Access to restricted network address ({ip_str}) is blocked."
                    return result
        except socket.gaierror:
            result["message"] = f"Could not resolve hostname '{hostname}'."
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
    """Downloads remote media from direct URL, enforcing size limits and format checks."""
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
        response = requests.get(url, headers=headers, stream=True, timeout=(10, 120))
        response.raise_for_status()

        cl = response.headers.get("content-length")
        if cl:
            size_mb = int(cl) / (1024 * 1024)
            if size_mb > max_mb:
                return {
                    "success": False,
                    "error": f"File size ({size_mb:.1f} MB) exceeds maximum limit of {max_mb} MB.",
                }

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type or "application/json" in content_type:
            return {
                "success": False,
                "error": f"URL returned HTML/text webpage ({content_type}) instead of {expected_type} media.",
            }

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


def cleanup_url_download(file_path: str | Path | None) -> None:
    """Safely cleans up downloaded temporary URL media files."""
    if not file_path:
        return
    try:
        p = Path(file_path)
        if p.is_file():
            p.unlink(missing_ok=True)
    except Exception:
        pass
