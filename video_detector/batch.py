"""
video_detector.batch: High-Speed Batch Processing Engine for Video Forensics.
Processes batches of video files or directories with optional progress reporting.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional

from video_detector.config import SUPPORTED_EXTENSIONS
from video_detector.pipeline import VideoForensicPipeline

logger = logging.getLogger("video_detector.batch")


class VideoBatchProcessor:
    """Batch forensic analyzer for multiple videos or entire video directories."""

    def __init__(self, pipeline: Optional[VideoForensicPipeline] = None):
        self.pipeline = pipeline or VideoForensicPipeline()

    def process_files(
        self,
        file_paths: List[str | Path],
        sensitivity: str = "balanced",
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Processes a sequence of video files."""
        results = []
        t0 = time.perf_counter()
        total = len(file_paths)

        ai_count = 0
        real_count = 0
        undecided_count = 0
        error_count = 0

        for idx, p in enumerate(file_paths):
            res = self.pipeline.analyze(p, sensitivity=sensitivity)
            results.append(res)

            # Case-insensitive substring matching (not exact equality) so this stays
            # correct if the taxonomy ever grows beyond the current three live states --
            # mirrors image_detector.batch's matching style rather than silently bucketing
            # any new/renamed status string into "errors".
            status = str(res.get("final_status", "")).upper()
            if not res.get("content_valid", True) or res.get("error"):
                error_count += 1
            elif "AI-GENERATED" in status:
                ai_count += 1
            elif "REAL" in status:
                real_count += 1
            elif "UNDECIDED" in status:
                undecided_count += 1
            else:
                error_count += 1

            if progress_callback:
                progress_callback(idx + 1, total, res)

        elapsed = time.perf_counter() - t0

        return {
            "total_processed": total,
            "elapsed_seconds": round(elapsed, 2),
            "mean_latency_seconds": round(elapsed / max(1, total), 3),
            "summary": {
                "ai_generated": ai_count,
                "real": real_count,
                "undecided": undecided_count,
                "errors": error_count,
            },
            "results": results,
        }

    def process_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True,
        sensitivity: str = "balanced",
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Scans and evaluates all supported video files in a directory."""
        dir_p = Path(directory_path)
        if not dir_p.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_p}")

        glob_pat = "**/*" if recursive else "*"
        all_files = [p for p in dir_p.glob(glob_pat) if p.suffix.lower() in SUPPORTED_EXTENSIONS]

        return self.process_files(all_files, sensitivity=sensitivity, progress_callback=progress_callback)
