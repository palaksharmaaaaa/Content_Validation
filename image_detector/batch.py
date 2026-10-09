"""image_detector.batch: sequential headless batch runner for image forensics; see core.batch."""
from __future__ import annotations

from typing import Optional

from core.batch import SequentialBatch
from image_detector.config import SUPPORTED_EXTENSIONS
from image_detector.pipeline import ImageForensicPipeline


class ImageBatchProcessor(SequentialBatch):
    """Batch forensic analyzer for multiple image files or a whole folder."""

    extensions = SUPPORTED_EXTENSIONS

    def __init__(self, pipeline: Optional[ImageForensicPipeline] = None):
        super().__init__(pipeline or ImageForensicPipeline())

