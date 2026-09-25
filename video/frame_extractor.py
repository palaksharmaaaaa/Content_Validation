"""
Video metadata extraction and sampled frame retrieval utilities.
"""
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


def get_video_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            return None

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0.0

        codec_value = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((codec_value >> (8 * i)) & 0xFF) for i in range(4))

        return {
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 2),
            "codec": codec.strip() or "Unknown",
        }
    finally:
        cap.release()


def extract_sampled_frames(video_path: str, num_frames: int = 30) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        sample_count = min(int(num_frames), total_frames)
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append({"index": int(index), "frame": frame})

        return frames
    finally:
        cap.release()