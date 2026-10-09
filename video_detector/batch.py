"""video_detector.batch: sequential headless batch runner for video forensics; see core.batch."""
from __future__ import annotations

from typing import Optional

from core.batch import SequentialBatch
from video_detector.config import SUPPORTED_EXTENSIONS
from video_detector.pipeline import VideoForensicPipeline


class VideoBatchProcessor(SequentialBatch):
    """Batch forensic analyzer for multiple video files or a whole folder."""

    extensions = SUPPORTED_EXTENSIONS

    def __init__(self, pipeline: Optional[VideoForensicPipeline] = None):
        super().__init__(pipeline or VideoForensicPipeline())

