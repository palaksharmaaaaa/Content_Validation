"""
image_detector.schemas: Data structures, scores, and result schemas for Image AI Detection.
Self-contained module schemas with zero outside dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ImageModalityScore:
    """Three-state authenticity percentage distribution."""
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


class ImageTaxonomyState:
    """Multi-state image authenticity taxonomy."""
    AUTHENTIC_REAL_PHOTOGRAPH = "AUTHENTIC_REAL_PHOTOGRAPH"
    AUTHENTIC_EDITED = "AUTHENTIC_EDITED"
    AUTHENTIC_RECAPTURED_SCREEN = "AUTHENTIC_RECAPTURED_SCREEN"
    AI_ENHANCED_COMPOSITE = "AI_ENHANCED_COMPOSITE"
    FULLY_AI_GENERATED = "FULLY_AI_GENERATED"
    PROCEDURAL_CGI_SYNTHETIC = "PROCEDURAL_CGI_SYNTHETIC"
    AUTHENTIC_SCREENSHOT = "AUTHENTIC_SCREENSHOT"
    AI_ENHANCED_SCREENSHOT = "AI_ENHANCED_SCREENSHOT"
    AI_GENERATED_SCREENSHOT = "AI_GENERATED_SCREENSHOT"

    LABELS = {
        AUTHENTIC_REAL_PHOTOGRAPH: "Authentic Real Photograph",
        AUTHENTIC_EDITED: "Authentic Created Photograph (Edited / Graphic Design)",
        AUTHENTIC_RECAPTURED_SCREEN: "Authentic Screen Re-photography (Recaptured)",
        AI_ENHANCED_COMPOSITE: "AI-Enhanced / Composite (Mix)",
        FULLY_AI_GENERATED: "Fully AI Generated",
        PROCEDURAL_CGI_SYNTHETIC: "Procedural CGI Synthetic (3D Render)",
        AUTHENTIC_SCREENSHOT: "Authentic Device Screenshot",
        AI_ENHANCED_SCREENSHOT: "AI-Enhanced / Composite Screenshot",
        AI_GENERATED_SCREENSHOT: "AI-Generated Content Screenshot",
    }

    DESCRIPTIONS = {
        AUTHENTIC_REAL_PHOTOGRAPH: (
            "Looks like a camera or mobile-phone photograph: it has camera-like fine grain and shows none of the signs of generation "
            "or manipulation that this tool checks for. This is a heuristic reading, not proof of origin."
        ),
        AUTHENTIC_EDITED: (
            "Looks like a photograph that went through conventional software edits (cropping, canvas resizing, "
            "background removal/cutout, color grading, or Canva/Photoshop graphic composition) with no sign of generative AI synthesis."
        ),
        AUTHENTIC_RECAPTURED_SCREEN: (
            "Looks like a camera photograph of a physical display screen (CRT/LCD/OLED), showing the "
            "spatial-frequency Moiré interference patterns such re-photography leaves."
        ),
        AI_ENHANCED_COMPOSITE: (
            "Real base capture augmented or modified via neural models (generative inpainting, "
            "AI object addition/removal, or deep learning upscaling/restoration software such as Topaz Photo AI)."
        ),
        FULLY_AI_GENERATED: (
            "Synthesized end-to-end via generative diffusion or autoregressive transformer models (Midjourney, DALL-E, "
            "Stable Diffusion, Flux, Imagen), supported by the pixel statistics, a visible watermark, or algorithmic IPTC metadata (the metadata is unauthenticated)."
        ),
        PROCEDURAL_CGI_SYNTHETIC: (
            "Synthetic imagery generated via deterministic procedural 3D ray-tracing/rasterization engines (Blender, Unreal Engine, "
            "Maya) characterized by mathematical geometric polygons and non-stochastic texture shaders."
        ),
        AUTHENTIC_SCREENSHOT: (
            "Digital screen capture from a mobile phone, tablet, laptop, or desktop monitor showing OS/app UI, "
            "documents, or photographic content with no sign of AI manipulation."
        ),
        AI_ENHANCED_SCREENSHOT: (
            "Digital screen capture containing or displaying media modified by AI tools or generative enhancement."
        ),
        AI_GENERATED_SCREENSHOT: (
            "Digital screen capture displaying fully synthetic AI-generated content (e.g. generative AI prompt outputs, "
            "diffusion artwork, or AI avatar applications)."
        ),
    }

    @classmethod
    def get_label(cls, state: str) -> str:
        return cls.LABELS.get(state, "Undetermined")

    @classmethod
    def get_description(cls, state: str) -> str:
        return cls.DESCRIPTIONS.get(state, "")


@dataclass
class ImageForensicResult:
    """Complete image forensic output schema."""
    is_available: bool
    backend: str
    prediction: str
    label: str
    confidence: float
    ai_percentage: float
    real_percentage: float
    undecided_percentage: float
    taxonomy_state: str = ImageTaxonomyState.AUTHENTIC_REAL_PHOTOGRAPH
    taxonomy_label: str = "Authentic Real Photograph"
    taxonomy_description: str = ""
    taxonomy_reasons: List[str] = field(default_factory=list)
    # Multi-Dimensional Taxonomy Categorization (Dimensions B, C, D, E)
    subject_genre: Optional[str] = None
    visual_medium: str = "Photographic Capture"
    sensor_spectrum: str = "Visible light (colour RGB)"
    document_layout: str = "None (Standard Visual Content)"
    watermark_detected: bool = False
    watermark_details: Optional[str] = None
    background_cutout_detected: bool = False
    scanned_photo_detected: bool = False
    screen_recapture_detected: bool = False
    screen_recapture_details: Optional[Dict[str, Any]] = None
    neural_enhancer_detected: bool = False
    digital_art_detected: bool = False
    screenshot_detected: bool = False
    screenshot_details: Optional[Dict[str, Any]] = None
    inpainting_detected: bool = False
    inpainting_details: Optional[Dict[str, Any]] = None
    log_likelihood_ratios: Dict[str, float] = field(default_factory=dict)
    posterior_log_odds: float = 0.0
    ai_spatial_area_pct: float = 0.0
    heatmap_rgb: Optional[Any] = None
    forensic_cues: List[str] = field(default_factory=list)
    forensic_metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_available": self.is_available,
            "backend": self.backend,
            "prediction": self.prediction,
            "label": self.label,
            "confidence": self.confidence,
            "ai_percentage": self.ai_percentage,
            "real_percentage": self.real_percentage,
            "undecided_percentage": self.undecided_percentage,
            "taxonomy_state": self.taxonomy_state,
            "taxonomy_label": self.taxonomy_label,
            "taxonomy_description": self.taxonomy_description,
            "taxonomy_reasons": self.taxonomy_reasons,
            "subject_genre": self.subject_genre,
            "visual_medium": self.visual_medium,
            "sensor_spectrum": self.sensor_spectrum,
            "document_layout": self.document_layout,
            "watermark_detected": self.watermark_detected,
            "watermark_details": self.watermark_details,
            "background_cutout_detected": self.background_cutout_detected,
            "scanned_photo_detected": self.scanned_photo_detected,
            "screen_recapture_detected": self.screen_recapture_detected,
            "screen_recapture_details": self.screen_recapture_details,
            "neural_enhancer_detected": self.neural_enhancer_detected,
            "digital_art_detected": self.digital_art_detected,
            "screenshot_detected": self.screenshot_detected,
            "screenshot_details": self.screenshot_details,
            "inpainting_detected": self.inpainting_detected,
            "inpainting_details": self.inpainting_details,
            "log_likelihood_ratios": self.log_likelihood_ratios,
            "posterior_log_odds": self.posterior_log_odds,
            "ai_spatial_area_pct": self.ai_spatial_area_pct,
            "heatmap_rgb": self.heatmap_rgb,
            "forensic_cues": self.forensic_cues,
            "forensic_metrics": self.forensic_metrics,
            "error": self.error,
        }


@dataclass
class ImageValidationResult:
    """Image file structure, metadata, and quality analysis result."""
    valid: bool
    filename: str = ""
    width: int = 0
    height: int = 0
    aspect_ratio: float = 1.0
    format: str = ""
    color_mode: str = ""
    file_size_mb: float = 0.0
    file_hash_sha256: str = ""
    quality: Dict[str, Any] = field(default_factory=dict)
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
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "format": self.format,
            "color_mode": self.color_mode,
            "file_size_mb": self.file_size_mb,
            "file_hash_sha256": self.file_hash_sha256,
            "quality": self.quality,
            "error": self.error,
        }


@dataclass
class ImageFeedbackRecord:
    """Structured record for continuous learning and calibration memory."""
    timestamp: str
    image_path: str
    user_label: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "user_label": self.user_label,
            "metrics": self.metrics,
            "notes": self.notes,
        }
