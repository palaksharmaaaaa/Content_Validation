"""
video_detector.models: Internal video neural models and checkpoints.
"""
from video_detector.models.backbone import VideoTemporalTransitionModel, build_video_temporal_model

__all__ = ["VideoTemporalTransitionModel", "build_video_temporal_model"]
