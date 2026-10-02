"""
image_detector.batch: High-Speed Batch Processing Engine for Image Forensics.
Processes batches of image files or directories with optional progress reporting.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional

from image_detector.config import SUPPORTED_EXTENSIONS
from image_detector.pipeline import ImageForensicPipeline

logger = logging.getLogger("image_detector.batch")


class ImageBatchProcessor:
    """Batch forensic analyzer for multiple images or entire directories."""

    def __init__(self, pipeline: Optional[ImageForensicPipeline] = None):
        self.pipeline = pipeline or ImageForensicPipeline()

    def process_files(
        self,
        file_paths: List[str | Path],
        sensitivity: str = "balanced",
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Processes a sequence of image files."""
        results = []
        t0 = time.perf_counter()
        total = len(file_paths)

        ai_count = 0
        composite_count = 0
        real_count = 0
        undecided_count = 0
        error_count = 0

        for idx, p in enumerate(file_paths):
            res = self.pipeline.analyze(p, sensitivity=sensitivity)
            results.append(res)

            status = str(res.get("final_status", "")).upper()
            if not res.get("content_valid", True) or res.get("error"):
                error_count += 1
            elif "AI-GENERATED" in status or "FULLY_AI" in status:
                ai_count += 1
            elif "COMPOSITE" in status or "ENHANCED" in status:
                composite_count += 1
            elif "REAL" in status or "AUTHENTIC" in status:
                real_count += 1
            else:
                undecided_count += 1

            if progress_callback:
                progress_callback(idx + 1, total, res)

        elapsed = time.perf_counter() - t0

        return {
            "total_processed": total,
            "elapsed_seconds": round(elapsed, 2),
            "mean_latency_seconds": round(elapsed / max(1, total), 3),
            "summary": {
                "ai_generated": ai_count,
                "composite": composite_count,
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
        """Scans and evaluates all supported image files in a directory."""
        dir_p = Path(directory_path)
        if not dir_p.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_p}")

        glob_pat = "**/*" if recursive else "*"
        all_files = [p for p in dir_p.glob(glob_pat) if p.suffix.lower() in SUPPORTED_EXTENSIONS]

        return self.process_files(all_files, sensitivity=sensitivity, progress_callback=progress_callback)
