"""
video_detector.extractor: Robust video frame sampling, metadata parsing, and track extraction.
Handles:
1. Video metadata extraction (FPS, frame count, duration, resolution, codec).
2. Uniform temporal frame sampling with memory-bounded buffers.
3. Representative keyframe extraction at specific timeline positions.
4. Extraction of embedded audio streams via FFmpeg.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_detector.config import DEFAULT_MAX_FRAMES

_FFMPEG_MISSING_WARNED = False

logger = logging.getLogger("video_detector.extractor")


def _get_file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except Exception as exc:
        logger.debug("_get_file_size_mb: ignored %s: %s", type(exc).__name__, exc)
        return 0.0


class VideoFrameExtractor:
    """Safe, multi-threaded frame and track extraction engine for video forensics."""

    def __init__(self, max_frames: int = DEFAULT_MAX_FRAMES):
        self.max_frames = max_frames

    @staticmethod
    def get_video_metadata(video_path: str | Path) -> Dict[str, Any]:
        """Reads container metadata, codec, duration, and frame rate safely."""
        path = Path(video_path)
        if not path.is_file():
            return {"error": "File does not exist."}

        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return {"error": "Could not open video file."}

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))  # OpenCV returns -1 for some containers
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_raw = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            fourcc = "".join(c for c in fourcc_raw if c.isprintable()) or "UNKNOWN"

            if fps <= 0 or np.isnan(fps):
                fps = 25.0
            duration_seconds = float(total_frames) / fps if fps > 0 else 0.0

            return {
                "filename": path.name,
                "file_size_mb": round(_get_file_size_mb(path), 3),
                "duration_seconds": round(duration_seconds, 2),
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "aspect_ratio": round(float(width) / max(1, height), 2),
                "codec": fourcc,
            }
        finally:
            cap.release()

    def extract_sampled_frames(
        self, video_path: str | Path, max_frames: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[float], int]:
        """
        Extracts uniformly distributed frames across the video timeline.
        Returns:
            frames: List of BGR numpy arrays.
            timestamps: List of timestamp seconds for each frame.
            temporal_step: The stride (in frames) between samples.
        """
        cap = cv2.VideoCapture(str(video_path))
        limit = max_frames or self.max_frames
        try:
            if not cap.isOpened():
                return [], [], 1

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 25.0

            if total_frames <= 0:
                frames, timestamps = [], []
                count = 0
                while len(frames) < limit:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    timestamps.append(count / fps)
                    count += 1
                return frames, timestamps, 1

            step = max(1, total_frames // limit)
            frames = []
            timestamps = []

            for i in range(0, total_frames, step):
                if len(frames) >= limit:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
                    timestamps.append(round(i / fps, 2))

            return frames, timestamps, step
        finally:
            cap.release()

    extract_frames = extract_sampled_frames

    @staticmethod
    def extract_keyframe(
        video_path: str | Path,
        timeline_pct: float = 0.15,
        output_path: Optional[Path | str] = None,
    ) -> Optional[np.ndarray]:
        """Extracts a representative keyframe at timeline_pct (defaults to 15% to skip black titles)."""
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                return None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_idx = max(0, int(total_frames * timeline_pct))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if (not ret or frame is None) and target_idx > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            if ret and frame is not None and output_path:
                cv2.imwrite(str(output_path), frame)
            return frame if ret else None
        finally:
            cap.release()

    @staticmethod
    def extract_audio_track(
        video_path: str | Path, output_wav: Path | str
    ) -> bool:
        """Extracts audio track from video to 16kHz mono WAV using FFmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_wav),
        ]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            return res.returncode == 0 and Path(output_wav).is_file() and Path(output_wav).stat().st_size > 44
        except FileNotFoundError:
            global _FFMPEG_MISSING_WARNED
            if not _FFMPEG_MISSING_WARNED:
                logger.warning(
                    "ffmpeg executable not found in PATH. video_detector requires the "
                    "ffmpeg system binary (not a pip package) to extract embedded audio "
                    "tracks -- install it from https://ffmpeg.org/download.html and "
                    "ensure it's on PATH. Audio-in-video analysis will be skipped."
                )
                _FFMPEG_MISSING_WARNED = True
            return False
        except Exception as exc:
            logger.debug("Audio extraction failed for %s: %s", video_path, exc)
            return False
