"""Memory allocation and leak profiling for video extraction operations.

Uses Python's standard library `tracemalloc` (Python-level allocations only; native OpenCV buffers are not counted) for
deterministic memory monitoring on every platform.
Checks that the Python-side memory used by frame extraction stays bounded; it cannot see leaks in OpenCV's native buffers.
"""
from __future__ import annotations

import gc
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

from video_detector.extractor import VideoFrameExtractor


def _create_synthetic_video(path: Path, num_frames: int = 30, size: tuple[int, int] = (160, 120)) -> Path:
    """Creates a temporary synthetic MP4 video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 15.0, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), (i * 7) % 256, dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


def profile_video_extraction_memory(video_path: Path | str, max_frames: int = 15) -> dict[str, float]:
    """Measures current and peak Python memory allocated during frame extraction."""
    extractor = VideoFrameExtractor(max_frames=max_frames)
    gc.collect()
    tracemalloc.start()
    try:
        frames, timestamps, step = extractor.extract_sampled_frames(video_path, max_frames=max_frames)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return {
        "frame_count": len(frames),
        "current_mb": current / (1024 * 1024),
        "peak_mb": peak / (1024 * 1024),
        "step": step,
    }


def test_video_extractor_peak_memory_bounded(tmp_path):
    """Peak memory during frame extraction must stay well within safe bounds (< 30 MB for test clip)."""
    vid_file = tmp_path / "test_memory.mp4"
    _create_synthetic_video(vid_file, num_frames=30, size=(160, 120))

    stats = profile_video_extraction_memory(vid_file, max_frames=10)
    assert stats["frame_count"] > 0
    # 10 frames of 160x120x3 is ~576 KB of raw pixel buffers; total peak should be under 30 MB
    assert stats["peak_mb"] < 30.0, f"Peak memory {stats['peak_mb']} MB exceeded 30 MB ceiling"


def test_video_extractor_no_memory_leak_on_repeated_runs(tmp_path):
    """Repeated extraction calls must not leak memory across iterations."""
    vid_file = tmp_path / "test_leak.mp4"
    _create_synthetic_video(vid_file, num_frames=20, size=(128, 128))

    extractor = VideoFrameExtractor(max_frames=5)

    gc.collect()
    tracemalloc.start()
    try:
        # Warmup iteration
        extractor.extract_sampled_frames(vid_file, max_frames=5)
        gc.collect()
        mem_after_warmup, _ = tracemalloc.get_traced_memory()

        # Run 15 sequential extraction cycles
        for _ in range(15):
            frames, _, _ = extractor.extract_sampled_frames(vid_file, max_frames=5)
            assert len(frames) > 0
            del frames
            gc.collect()

        mem_after_stress, peak = tracemalloc.get_traced_memory()
        delta_kb = (mem_after_stress - mem_after_warmup) / 1024.0

        # Memory delta after warmup must not systematically grow (tolerance: under 250 KB)
        assert delta_kb < 250.0, f"Memory leaked by {delta_kb} KB across sequential extractions"
    finally:
        tracemalloc.stop()
