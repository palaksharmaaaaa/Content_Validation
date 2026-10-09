"""audio_detector.batch: sequential headless batch runner for audio forensics; see core.batch."""
from __future__ import annotations

from typing import Optional

from core.batch import SequentialBatch
from audio_detector.config import SUPPORTED_EXTENSIONS
from audio_detector.pipeline import AudioForensicPipeline


class AudioBatchProcessor(SequentialBatch):
    """Batch forensic analyzer for multiple audio files or a whole folder."""

    extensions = SUPPORTED_EXTENSIONS

    def __init__(self, pipeline: Optional[AudioForensicPipeline] = None):
        super().__init__(pipeline or AudioForensicPipeline())

