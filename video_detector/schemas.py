"""
video_detector.schemas: Data structures, scores, and result schemas for Video AI Detection.
Self-contained module schemas with zero outside dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VideoModalityScore:
    """NIST-aligned three-state authenticity percentage distribution for video."""
    ai_percentage: float = 0.0
    real_percentage: float = 0.0
    undecided_percentage: float = 100.0
    confidence: float = 0.0
    label: str = "UNDECIDED"
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.ai_percentage = max(0.0, float(self.ai_percentage))
        self.real_percentage = max(0.0, float(self.real_percentage))
        self.undecided_percentage = max(0.0, float(self.undecided_percentage))
        total = self.ai_percentage + self.real_percentage + self.undecided_percentage
        if total > 0 and abs(total - 100.0) > 0.05:
            self.ai_percentage = (self.ai_percentage / total) * 100.0
            self.real_percentage = (self.real_percentage / total) * 100.0
            self.undecided_percentage = max(0.0, 100.0 - (self.ai_percentage + self.real_percentage))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ai_percentage": round(self.ai_percentage, 1),
            "real_percentage": round(self.real_percentage, 1),
            "undecided_percentage": round(self.undecided_percentage, 1),
            "confidence": round(self.confidence, 2),
            "label": self.label,
            "details": self.details,
        }


@dataclass
class VideoTemporalSegment:
    """Contiguous timeline interval with classification."""
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    label: str
    frame_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_seconds": round(self.start_seconds, 2),
            "end_seconds": round(self.end_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "label": self.label,
            "frame_count": self.frame_count,
        }


@dataclass
class VideoForensicResult:
    """Complete video forensic analysis output schema."""
    valid: bool
    filename: str = ""
    duration_seconds: float = 0.0
    total_frames: int = 0
    sampled_frames_count: int = 0
    ai_percentage: float = 0.0
    real_percentage: float = 0.0
    undecided_percentage: float = 100.0
    confidence: float = 0.0
    label: str = "UNDECIDED"
    prediction: str = "UNDECIDED"
    temporal_consistency: Dict[str, Any] = field(default_factory=dict)
    diffusion_flicker: Dict[str, Any] = field(default_factory=dict)
    temporal_segments: List[Dict[str, Any]] = field(default_factory=list)
    forensic_cues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "filename": self.filename,
            "duration_seconds": self.duration_seconds,
            "total_frames": self.total_frames,
            "sampled_frames_count": self.sampled_frames_count,
            "ai_percentage": self.ai_percentage,
            "real_percentage": self.real_percentage,
            "undecided_percentage": self.undecided_percentage,
            "confidence": self.confidence,
            "label": self.label,
            "prediction": self.prediction,
            "temporal_consistency": self.temporal_consistency,
            "diffusion_flicker": self.diffusion_flicker,
            "temporal_segments": self.temporal_segments,
            "forensic_cues": self.forensic_cues,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class VideoValidationResult:
    """Video container and stream validation result."""
    valid: bool
    filename: str = ""
    file_size_mb: float = 0.0
    file_hash_sha256: str = ""
    duration_seconds: float = 0.0
    fps: float = 0.0
    total_frames: int = 0
    width: int = 0
    height: int = 0
    aspect_ratio: float = 1.0
    codec: str = "UNKNOWN"
    error: Optional[str] = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "filename": self.filename,
            "file_size_mb": self.file_size_mb,
            "file_hash_sha256": self.file_hash_sha256,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "codec": self.codec,
            "error": self.error,
        }


@dataclass
class VideoFeedbackRecord:
    """Feedback record for video continual learning."""
    timestamp: str
    video_path: str
    user_label: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "video_path": self.video_path,
            "user_label": self.user_label,
            "metrics": self.metrics,
            "notes": self.notes,
        }


@dataclass
class VideoBenchmarkMetrics:
    """Benchmark evaluation metrics for video datasets."""
    total_videos: int = 0
    total_frames_analyzed: int = 0
    video_accuracy: float = 0.0
    video_f1_score: float = 0.0
    video_roc_auc: float = 0.0
    mean_latency_seconds: float = 0.0
    confusion_matrix: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_videos": self.total_videos,
            "total_frames_analyzed": self.total_frames_analyzed,
            "video_accuracy": round(self.video_accuracy, 4),
            "video_f1_score": round(self.video_f1_score, 4),
            "video_roc_auc": round(self.video_roc_auc, 4),
            "mean_latency_seconds": round(self.mean_latency_seconds, 2),
            "confusion_matrix": self.confusion_matrix,
        }
