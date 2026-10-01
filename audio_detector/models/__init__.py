"""
audio_detector.models: Neural model architectures and checkpoints for Audio AI Detection.
"""
from audio_detector.models.backbone import AudioClassifierNet, build_audio_classifier

__all__ = ["AudioClassifierNet", "build_audio_classifier"]
