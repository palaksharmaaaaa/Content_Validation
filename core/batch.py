"""core.batch: the sequential headless batch runner shared by image, audio and video (the app's own batches are parallel, see ui.adapters).

Every file is analysed on its own: one that raises is recorded as an error and the run continues. A result is bucketed by its exact
final status, so a verdict word one modality spells differently can never fall into the wrong bucket.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger("core.batch")

_BUCKETS = {
    "LIKELY_SYNTHETIC": "ai_generated", "LIKELY AI-GENERATED": "ai_generated",
    "PARTIALLY_SYNTHETIC_OR_EDITED": "composite", "AI-ENHANCED / COMPOSITE": "composite",
    "LIKELY_AUTHENTIC": "real", "LIKELY REAL": "real", "AUTHENTIC (CONVENTIONALLY EDITED)": "real", "AUTHENTIC_EDITED": "real",
    "BLANK_OR_DEGRADED": "no_content",
}


def bucket_of(result: Dict[str, Any]) -> str:
    """ai_generated, composite, real, no_content, undecided or errors, from a pipeline result."""
    if result.get("error") or result.get("content_valid") is False:
        return "errors"
    return _BUCKETS.get(str(result.get("final_status", "")).upper(), "undecided")


class SequentialBatch:
    """Analyse files one after another with optional progress reporting; subclasses name the pipeline and the supported extensions."""

    extensions: Iterable[str] = ()

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def process_files(
        self,
        file_paths: List[str | Path],
        sensitivity: str = "balanced",
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        total = len(file_paths)
        counts = {k: 0 for k in ("ai_generated", "composite", "real", "undecided", "no_content", "errors")}
        results: List[Dict[str, Any]] = []
        for index, path in enumerate(file_paths, 1):
            try:
                res = self.pipeline.analyze(path, sensitivity=sensitivity)
            except Exception as exc:                                  # a bad file must not stop the rest
                logger.warning("batch: %s failed (%s: %s)", Path(path).name, type(exc).__name__, exc)
                res = {"error": f"{type(exc).__name__}: {exc}", "file_path": str(path)}
            results.append(res)
            counts[bucket_of(res)] += 1
            if progress_callback:
                progress_callback(index, total, res)
        elapsed = time.perf_counter() - started
        return {
            "total_processed": total,
            "elapsed_seconds": round(elapsed, 2),
            "mean_latency_seconds": round(elapsed / max(1, total), 3),
            "summary": counts,
            "results": results,
        }

    def process_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True,
        sensitivity: str = "balanced",
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        folder = Path(directory_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Directory not found: {folder}")
        wanted = {e.lower() for e in self.extensions}
        files = sorted(p for p in folder.glob("**/*" if recursive else "*") if p.is_file() and p.suffix.lower() in wanted)
        return self.process_files(files, sensitivity=sensitivity, progress_callback=progress_callback)
