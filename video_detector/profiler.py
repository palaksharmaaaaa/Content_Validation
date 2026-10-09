"""
video_detector.profiler: Technical Media Profiler and Signal Forensic Inspector for Video.
Extracts:
1. Cryptographic identity (SHA-256, MD5, size in bytes).
2. Video stream specifications (fps, duration, total frames, resolution, aspect ratio).
3. Codec and container details.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from core.hashing import file_digests
import cv2

logger = logging.getLogger("video_detector.profiler")


def compute_file_hashes(file_path: str | Path) -> Tuple[str, str, int]:
    """Calculates SHA-256, MD5, and exact file size in bytes (shared cached implementation)."""
    return file_digests(file_path)


class VideoProfiler:
    """Profiles technical stream specifications of video files."""

    def __init__(self):
        pass

    def profile_video(self, file_path: str | Path) -> Dict[str, Any]:
        """Extracts complete stream specifications and hashes from a video."""
        path = Path(file_path)
        if not path.is_file():
            return {"valid": False, "error": f"File not found: {path}"}

        sha256, md5, size_bytes = compute_file_hashes(path)
        size_mb = size_bytes / (1024.0 * 1024.0)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {
                "valid": False,
                "filename": path.name,
                "file_size_mb": round(size_mb, 3),
                "sha256": sha256,
                "md5": md5,
                "error": "Failed to open video container with OpenCV.",
            }

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        cap.release()

        duration = (total_frames / fps) if fps > 0 else 0.0

        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
        if not codec:
            codec = path.suffix.lstrip(".").upper()
        aspect = round(float(width) / max(1.0, float(height)), 2)
        # Average bitrate of the whole file (video and audio together) = size / duration; unknown when the duration is.
        bitrate_kbps = round(size_bytes * 8 / duration / 1000.0, 1) if duration > 0 else None
        container = path.suffix.lstrip(".").upper()

        return {
            "valid": True,
            "filename": path.name,
            "file_size_bytes": size_bytes,
            "file_size_mb": round(size_mb, 3),
            "sha256": sha256,
            "md5": md5,
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "aspect_ratio": aspect,
            "codec": codec,
            "container_format": container,
            # The nested blocks below are what the dossier, the explanation and the result page read.
            "geometry": {"width": width, "height": height, "fps": round(fps, 2), "duration_seconds": round(duration, 2),
                         "total_frames": total_frames, "aspect_ratio": aspect},
            "codec_and_container": {"container": container, "codec": codec, "bitrate_kbps": bitrate_kbps},
            "cryptographic_hashes": {"sha256": sha256, "md5": md5},
            "bitrate_kbps": bitrate_kbps,
        }
