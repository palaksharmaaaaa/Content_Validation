"""
Schemas and data structures for media content validation and AI generation detection.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ModalityScore:
    """Represents AI vs Real vs Undecided percentages for a given modality (image, video, audio)."""
    ai_percentage: float = 0.0
    real_percentage: float = 0.0
    undecided_percentage: float = 100.0
    confidence: float = 0.0
    label: str = "UNDECIDED"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ai_percentage": round(self.ai_percentage, 2),
            "real_percentage": round(self.real_percentage, 2),
            "undecided_percentage": round(self.undecided_percentage, 2),
            "confidence": round(self.confidence, 2),
            "label": self.label,
            "details": self.details,
        }


@dataclass
class ValidationDecision:
    """Represents the final verdict of the decision engine."""
    content_valid: bool
    final_status: str
    reason: str
    media_type: str
    scores: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_valid": self.content_valid,
            "final_status": self.final_status,
            "reason": self.reason,
            "media_type": self.media_type,
            "scores": self.scores,
        }
