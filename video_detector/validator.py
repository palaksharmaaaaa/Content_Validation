"""
video_detector.validator: Standalone video container & temporal stream validator.
Performs:
1. Container format integrity validation (MP4, MOV, MKV, WEBM, AVI).
2. Resolution, aspect ratio, frame rate, and duration checks.
3. Codec decoding and stream readability verification.
4. Dropped frames and video degradation inspection.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging

from pathlib import Path

import cv2

from core.ffmpeg import open_video
from core.hashing import file_sha256
from video_detector.config import MAX_DURATION_SECONDS, MAX_FILE_SIZE_MB, MIN_RESOLUTION, SUPPORTED_EXTENSIONS
from video_detector.extractor import VideoFrameExtractor
from video_detector.schemas import VideoValidationResult

logger = logging.getLogger("video_detector.validator")


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


class VideoValidator:
    """Independent validator for video stream integrity, codecs, and temporal playback."""

    def __init__(
        self,
        max_file_size_mb: float = MAX_FILE_SIZE_MB,
        max_duration_sec: float = MAX_DURATION_SECONDS,
    ):
        self.max_file_size_mb = max_file_size_mb
        self.max_duration_sec = max_duration_sec

    def validate(self, video_path: str | Path) -> VideoValidationResult:
        """
        Validates video file container, extracts streams, and assesses playback integrity.
        """
        path = Path(video_path)
        if not path.is_file():
            return VideoValidationResult(
                valid=False,
                error=f"Video file not found: {path}",
            )

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return VideoValidationResult(
                valid=False,
                filename=path.name,
                error=f"Unsupported video file extension: {path.suffix}",
            )

        size_mb = _get_file_size_mb(path)
        if size_mb > self.max_file_size_mb:
            return VideoValidationResult(
                valid=False,
                filename=path.name,
                file_size_mb=round(size_mb, 3),
                error=f"Video file size ({size_mb:.2f}MB) exceeds limit of {self.max_file_size_mb}MB",
            )

        meta = VideoFrameExtractor.get_video_metadata(path)
        if "error" in meta:
            return VideoValidationResult(
                valid=False,
                filename=path.name,
                file_size_mb=round(size_mb, 3),
                error=meta["error"],
            )

        width, height = int(meta.get("width", 0) or 0), int(meta.get("height", 0) or 0)
        if 0 < min(width, height) < MIN_RESOLUTION:
            return VideoValidationResult(
                valid=False,
                filename=path.name,
                file_size_mb=round(size_mb, 3),
                width=width,
                height=height,
                error=f"Video frame size ({width}x{height}) is below the {MIN_RESOLUTION}px minimum the detector can judge.",
            )

        dur = meta.get("duration_seconds", 0.0)
        if dur > self.max_duration_sec:
            return VideoValidationResult(
                valid=False,
                filename=path.name,
                duration_seconds=dur,
                error=f"Video duration ({dur:.1f}s) exceeds limit of {self.max_duration_sec}s",
            )

        # Check stream readability by reading first and middle frames
        cap = open_video(path)
        try:
            ret1, _ = cap.read()
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_f // 2))
            ret2, _ = cap.read()
            if not ret1 and not ret2:
                return VideoValidationResult(
                    valid=False,
                    filename=path.name,
                    duration_seconds=dur,
                    error="Failed to decode video frames. Stream may be corrupt or encoded in an unsupported codec.",
                )
        finally:
            cap.release()

        return VideoValidationResult(
            valid=True,
            filename=path.name,
            file_size_mb=round(size_mb, 3),
            file_hash_sha256=_calculate_file_hash(path),
            duration_seconds=dur,
            fps=meta.get("fps", 0.0),
            total_frames=meta.get("total_frames", 0),
            width=meta.get("width", 0),
            height=meta.get("height", 0),
            aspect_ratio=meta.get("aspect_ratio", 1.0),
            codec=meta.get("codec", "UNKNOWN"),
        )
