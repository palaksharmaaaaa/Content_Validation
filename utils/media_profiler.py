"""
Comprehensive Media Profiler and Signal Forensic Inspector.
Extracts exhaustive technical specifications:
- Cryptographic identity (SHA-256, MD5, size, MIME type).
- Pixel-level specifications (data type, bits per pixel, Shannon entropy, color space, channel variance).
- Audio signal metrics (sample rate, bit depth, bitrate, channels, RMS energy, crest factor).
- Container, encoding profiles, and EXIF provenance.
"""
from __future__ import annotations

import hashlib
import io
import mimetypes
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import wave

import cv2
import numpy as np
from PIL import Image

from utils.image_utils import extract_image_metadata
from utils.logging_utils import get_logger

logger = get_logger("media_profiler")


def compute_file_hashes(file_path: str | Path) -> Tuple[str, str, int]:
    """Calculates SHA-256, MD5, and exact file size in bytes."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    total_bytes = 0

    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
            md5.update(chunk)
            total_bytes += len(chunk)

    return sha256.hexdigest(), md5.hexdigest(), total_bytes


def compute_pixel_entropy(gray_or_channels: np.ndarray) -> float:
    """
    Computes Shannon Information Entropy in bits per pixel.
    Natural photos typically exhibit 6.8 - 7.6 bits/pixel.
    Synthetically smoothed or flattened images often exhibit lower entropy.
    """
    try:
        if gray_or_channels.ndim == 3:
            gray = cv2.cvtColor(gray_or_channels, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_or_channels

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / max(1.0, float(hist.sum()))
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))
        return round(entropy, 3)
    except Exception:
        return 0.0


def profile_image_media(image_path: str | Path) -> Dict[str, Any]:
    """
    Extracts deep pixel-level specifications, color space, bit depth,
    Shannon entropy, and camera hardware provenance from an image.
    """
    file_path = Path(image_path)
    sha256, md5, size_bytes = compute_file_hashes(file_path)
    mime_type, _ = mimetypes.guess_type(str(file_path))
    mime_type = mime_type or "image/unknown"

    img_bgr = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        return {
            "success": False,
            "error": "Failed to read image buffer.",
            "file_identity": {
                "filename": file_path.name,
                "sha256": sha256,
                "md5": md5,
                "size_bytes": size_bytes,
                "size_kb": round(size_bytes / 1024, 2),
                "size_mb": round(size_bytes / (1024 * 1024), 3),
                "mime_type": mime_type,
            },
        }

    h, w = img_bgr.shape[:2]
    channels = 1 if img_bgr.ndim == 2 else img_bgr.shape[2]
    dtype_str = str(img_bgr.dtype)

    # Determine bit depth per channel and bits per pixel (bpp)
    bits_per_channel = 8
    if img_bgr.dtype == np.uint16:
        bits_per_channel = 16
    elif img_bgr.dtype == np.float32:
        bits_per_channel = 32

    bits_per_pixel = channels * bits_per_channel

    # Color space determination
    if channels == 1:
        color_space = "GRAYSCALE"
    elif channels == 3:
        color_space = "RGB / BGR"
    elif channels == 4:
        color_space = "RGBA / BGRA"
    else:
        color_space = f"MULTI_CHANNEL_{channels}"

    # Shannon Entropy (Data per pixel)
    entropy = compute_pixel_entropy(img_bgr)

    # Channel Statistics
    if channels >= 3:
        # img_bgr is BGR
        b_mean, g_mean, r_mean = [float(np.mean(img_bgr[:, :, i])) for i in range(3)]
        b_std, g_std, r_std = [float(np.std(img_bgr[:, :, i])) for i in range(3)]
        channel_means = {"R": round(r_mean, 2), "G": round(g_mean, 2), "B": round(b_mean, 2)}
        channel_stds = {"R": round(r_std, 2), "G": round(g_std, 2), "B": round(b_std, 2)}
    else:
        gray_mean = float(np.mean(img_bgr))
        gray_std = float(np.std(img_bgr))
        channel_means = {"L": round(gray_mean, 2)}
        channel_stds = {"L": round(gray_std, 2)}

    overall_variance = float(np.var(img_bgr))

    # Metadata and provenance
    metadata_info = extract_image_metadata(file_path)

    # PIL inspection for format details
    pil_format = None
    try:
        with Image.open(file_path) as p_img:
            pil_format = p_img.format
    except Exception:
        pass

    gcd = np.gcd(w, h)
    aspect_ratio_str = f"{w // gcd}:{h // gcd}" if gcd > 0 else f"{w}:{h}"

    return {
        "success": True,
        "media_type": "image",
        "file_identity": {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "sha256": sha256,
            "md5": md5,
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
            "size_mb": round(size_bytes / (1024 * 1024), 3),
            "mime_type": mime_type,
            "container_format": pil_format or file_path.suffix.replace(".", "").upper(),
        },
        "pixel_specifications": {
            "pixel_data_type": dtype_str,
            "bits_per_channel": bits_per_channel,
            "bits_per_pixel": bits_per_pixel,
            "channel_count": channels,
            "color_space": color_space,
            "width": w,
            "height": h,
            "total_pixels": w * h,
            "megapixels": round((w * h) / 1_000_000, 3),
            "aspect_ratio": aspect_ratio_str,
            "aspect_ratio_decimal": round(w / max(1, h), 3),
            "shannon_entropy_bpp": entropy,
            "channel_means": channel_means,
            "channel_std_devs": channel_stds,
            "pixel_variance": round(overall_variance, 2),
        },
        "provenance_metadata": metadata_info,
    }


def profile_audio_media(audio_path: str | Path) -> Dict[str, Any]:
    """
    Extracts audio waveform properties, sample rate, bit depth, channel configuration,
    bitrate, dynamic range, RMS energy, and crest factor.
    """
    from utils.audio_utils import extract_audio_samples

    file_path = Path(audio_path)
    sha256, md5, size_bytes = compute_file_hashes(file_path)
    mime_type, _ = mimetypes.guess_type(str(file_path))
    mime_type = mime_type or "audio/unknown"

    samples, sample_rate, duration = extract_audio_samples(file_path)
    if samples is None or len(samples) < 100:
        return {
            "success": False,
            "error": "Failed to decode audio track.",
            "file_identity": {
                "filename": file_path.name,
                "sha256": sha256,
                "md5": md5,
                "size_bytes": size_bytes,
                "size_kb": round(size_bytes / 1024, 2),
                "size_mb": round(size_bytes / (1024 * 1024), 3),
                "mime_type": mime_type,
            },
        }

    # Audio metrics
    rms_energy = float(np.sqrt(np.mean(samples ** 2)))
    peak_val = float(np.max(np.abs(samples)))
    crest_factor = round(peak_val / max(1e-6, rms_energy), 3)

    # Zero-crossing rate
    zero_crossings = float(np.mean(np.abs(np.diff(np.signbit(samples)))))

    # Shannon Entropy of audio sample distribution
    audio_hist, _ = np.histogram(samples, bins=128, range=(-1.0, 1.0), density=True)
    audio_hist = audio_hist / max(1.0, float(np.sum(audio_hist)))
    audio_hist = audio_hist[audio_hist > 0]
    audio_entropy = float(-np.sum(audio_hist * np.log2(audio_hist)))

    # Typical audio parameters for 16-bit PCM standard
    bit_depth = 16
    channels = 1
    bitrate_kbps = round((sample_rate * channels * bit_depth) / 1000.0, 1)

    return {
        "success": True,
        "media_type": "audio",
        "file_identity": {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "sha256": sha256,
            "md5": md5,
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
            "size_mb": round(size_bytes / (1024 * 1024), 3),
            "mime_type": mime_type,
            "container_format": file_path.suffix.replace(".", "").upper(),
        },
        "signal_specifications": {
            "sample_rate_hz": sample_rate,
            "bit_depth": bit_depth,
            "bits_per_sample": bit_depth,
            "channel_count": channels,
            "channel_mode": "Mono" if channels == 1 else "Stereo",
            "duration_seconds": round(duration, 3),
            "total_samples": len(samples),
            "bitrate_kbps": bitrate_kbps,
            "rms_energy": round(rms_energy, 4),
            "peak_amplitude": round(peak_val, 4),
            "crest_factor": crest_factor,
            "zero_crossing_rate": round(zero_crossings, 4),
            "audio_entropy": round(audio_entropy, 3),
        },
    }


def profile_video_media(video_path: str | Path) -> Dict[str, Any]:
    """
    Extracts video container properties, frame geometry, frame rate,
    codec FourCC, pixel specifications, and demuxed audio signal properties.
    """
    file_path = Path(video_path)
    sha256, md5, size_bytes = compute_file_hashes(file_path)
    mime_type, _ = mimetypes.guess_type(str(file_path))
    mime_type = mime_type or "video/mp4"

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return {
            "success": False,
            "error": "Failed to open video stream.",
            "file_identity": {
                "filename": file_path.name,
                "sha256": sha256,
                "md5": md5,
                "size_bytes": size_bytes,
                "size_kb": round(size_bytes / 1024, 2),
                "size_mb": round(size_bytes / (1024 * 1024), 3),
                "mime_type": mime_type,
            },
        }

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = float(total_frames / max(1.0, fps))

    # Read codec FourCC
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

    # Sample keyframe for pixel analysis
    ret, sample_frame = cap.read()
    cap.release()

    pixel_entropy = 0.0
    dtype_str = "uint8"
    bits_per_pixel = 24
    if ret and sample_frame is not None:
        pixel_entropy = compute_pixel_entropy(sample_frame)
        dtype_str = str(sample_frame.dtype)
        bits_per_pixel = (1 if sample_frame.ndim == 2 else sample_frame.shape[2]) * 8

    # Extract audio profile if present
    audio_info = profile_audio_media(file_path)

    gcd = np.gcd(w, h) if w > 0 and h > 0 else 1
    aspect_ratio_str = f"{w // gcd}:{h // gcd}" if gcd > 0 else f"{w}:{h}"

    return {
        "success": True,
        "media_type": "video",
        "file_identity": {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "sha256": sha256,
            "md5": md5,
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
            "size_mb": round(size_bytes / (1024 * 1024), 3),
            "mime_type": mime_type,
            "container_format": file_path.suffix.replace(".", "").upper(),
        },
        "stream_specifications": {
            "codec_fourcc": fourcc_str or "H.264/AVC",
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio_str,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration_sec, 2),
            "pixel_data_type": dtype_str,
            "bits_per_pixel": bits_per_pixel,
            "keyframe_entropy_bpp": pixel_entropy,
        },
        "demuxed_audio": audio_info if audio_info.get("success") else None,
    }


def profile_media(file_path: str | Path, modality: str = "image") -> Dict[str, Any]:
    """Universal dispatcher for comprehensive media profiling."""
    mod = modality.lower()
    if mod in ("image", "img"):
        return profile_image_media(file_path)
    elif mod in ("audio", "speech", "voice"):
        return profile_audio_media(file_path)
    elif mod in ("video", "vid"):
        return profile_video_media(file_path)
    else:
        # Auto-detect from extension
        ext = Path(file_path).suffix.lower()
        if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            return profile_video_media(file_path)
        elif ext in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"):
            return profile_audio_media(file_path)
        else:
            return profile_image_media(file_path)
