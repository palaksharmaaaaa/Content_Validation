"""audio_detector.batch: sequential headless batch runner for audio forensics; see core.batch."""
from __future__ import annotations

from audio_detector.config import SUPPORTED_EXTENSIONS
from audio_detector.pipeline import AudioForensicPipeline
from core.batch import SequentialBatch


class AudioBatchProcessor(SequentialBatch):
    """Batch forensic analyzer for multiple audio files or a whole folder."""

    extensions = SUPPORTED_EXTENSIONS

    def __init__(self, pipeline: AudioForensicPipeline | None = None):
        super().__init__(pipeline or AudioForensicPipeline())

