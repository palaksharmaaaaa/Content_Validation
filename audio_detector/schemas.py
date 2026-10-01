"""
audio_detector.schemas: Data structures, scores, and result schemas for Audio AI Detection.
Self-contained module schemas with zero outside dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AudioModalityScore:
    """NIST-aligned three-state authenticity percentage distribution for audio."""
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
class AudioTemporalSegment:
    """Contiguous timeline interval with audio classification."""
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    label: str
    ai_prob: float = 0.0
    has_vocoder_cutoff: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_seconds": round(self.start_seconds, 2),
            "end_seconds": round(self.end_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "label": self.label,
            "ai_prob": round(self.ai_prob, 3),
            "has_vocoder_cutoff": self.has_vocoder_cutoff,
        }


@dataclass
class AudioForensicResult:
    """Complete audio forensic analysis output schema."""
    valid: bool
    filename: str = ""
    duration_seconds: float = 0.0
    sample_rate: int = 16000
    ai_percentage: float = 0.0
    real_percentage: float = 0.0
    undecided_percentage: float = 100.0
    confidence: float = 0.0
    label: str = "UNDECIDED"
    prediction: str = "UNDECIDED"
    has_vocoder_cutoff: bool = False
    cutoff_freq_hz: float = 0.0
    spectral_flatness: float = 0.0
    digital_silence_ratio: float = 0.0
    high_freq_ratio: float = 0.0
    acoustic_features: Dict[str, Any] = field(default_factory=dict)
    temporal_segments: List[Dict[str, Any]] = field(default_factory=list)
    forensic_cues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    spectrogram_image: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "filename": self.filename,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "ai_percentage": self.ai_percentage,
            "real_percentage": self.real_percentage,
            "undecided_percentage": self.undecided_percentage,
            "confidence": self.confidence,
            "label": self.label,
            "prediction": self.prediction,
            "has_vocoder_cutoff": self.has_vocoder_cutoff,
            "cutoff_freq_hz": self.cutoff_freq_hz,
            "spectral_flatness": self.spectral_flatness,
            "digital_silence_ratio": self.digital_silence_ratio,
            "high_freq_ratio": self.high_freq_ratio,
            "acoustic_features": self.acoustic_features,
            "temporal_segments": self.temporal_segments,
            "forensic_cues": self.forensic_cues,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class AudioValidationResult:
    """Audio container and stream validation result."""
    valid: bool
    filename: str = ""
    file_size_mb: float = 0.0
    file_hash_sha256: str = ""
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    format: str = "UNKNOWN"
    is_clipped: bool = False
    is_silent: bool = False
    error: Optional[str] = None
    extracted_samples: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "filename": self.filename,
            "file_size_mb": self.file_size_mb,
            "file_hash_sha256": self.file_hash_sha256,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format,
            "is_clipped": self.is_clipped,
            "is_silent": self.is_silent,
            "error": self.error,
        }


@dataclass
class AudioFeedbackRecord:
    """Feedback record for audio continual learning."""
    timestamp: str
    audio_path: str
    user_label: str
    features: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "audio_path": self.audio_path,
            "user_label": self.user_label,
            "features": self.features,
            "notes": self.notes,
        }


@dataclass
class AudioBenchmarkMetrics:
    """Benchmark evaluation metrics for audio datasets."""
    total_audio_files: int = 0
    total_duration_seconds: float = 0.0
    audio_accuracy: float = 0.0
    audio_f1_score: float = 0.0
    audio_roc_auc: float = 0.0
    mean_latency_seconds: float = 0.0
    confusion_matrix: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_audio_files": self.total_audio_files,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "audio_accuracy": round(self.audio_accuracy, 4),
            "audio_f1_score": round(self.audio_f1_score, 4),
            "audio_roc_auc": round(self.audio_roc_auc, 4),
            "mean_latency_seconds": round(self.mean_latency_seconds, 3),
            "confusion_matrix": self.confusion_matrix,
        }
