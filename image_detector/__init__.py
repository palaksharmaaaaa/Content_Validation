"""
image_detector: Completely independent, self-contained, feedback-calibrated Image AI Detection package (see learner.py for exactly what the calibration does and does not do).
Contains its own:
- Configurations & thresholds (image_detector.config)
- Schemas & data structures (image_detector.schemas)
- Neural models & checkpoints (image_detector.models)
- Feature extraction routines (image_detector.features)
- Bayesian scoring & evidence pooling (image_detector.scoring)
- Image file & visual quality validators (image_detector.validator)
- Continual learning & dynamic calibration (image_detector.learner)
- Neural classifier trainer (image_detector.trainer)
- Benchmark evaluation suite (image_detector.benchmarks)
- Main detection engine (image_detector.detector.ImageAIDetector)
- Facial deepfake detection (image_detector.face.FaceDeepfakeDetector)
- Scene & content intelligence (image_detector.content.ImageContentAnalyzer)
- Generator model attribution (image_detector.attribution.ImageModelAttributionEngine)
- Cryptographic C2PA & EXIF provenance (image_detector.provenance.ImageProvenanceValidator)
- Signal profiling & entropy (image_detector.profiler.ImageProfiler)
- Safe URL image downloader (image_detector.downloader.ImageDownloader)
- End-to-end forensic pipeline (image_detector.pipeline.ImageForensicPipeline)
- High-speed batch processor (image_detector.batch.ImageBatchProcessor)
"""
from image_detector.attribution import ImageModelAttributionEngine
from image_detector.batch import ImageBatchProcessor
from image_detector.benchmarks import ImageBenchmarkSuite
from image_detector.content import ImageContentAnalyzer
from image_detector.detector import ImageAIDetector
from image_detector.downloader import ImageDownloader
from image_detector.explain import build_nine_dimensions_dossier, generate_newbie_explanation
from image_detector.face import FaceDeepfakeDetector
from image_detector.features import (
    analyze_fft_radial_power_spectrum,
    calculate_sensor_noise_profile,
    calculate_surface_smoothness,
    compute_ela,
    detect_inpainting_and_manipulation,
    detect_screenshot,
    extract_image_metadata,
    generate_spatial_manipulation_heatmap,
)
from image_detector.learner import ImageSelfImprover
from image_detector.models.backbone import build_image_classifier
from image_detector.pipeline import ImageForensicPipeline
from image_detector.profiler import (
    ImageProfiler,
    compute_file_hashes,
    compute_pixel_entropy,
    extract_all_image_details,
    rgb_to_color_name,
)
from image_detector.provenance import ImageProvenanceValidator
from image_detector.schemas import (
    ImageBenchmarkMetrics,
    ImageFeedbackRecord,
    ImageForensicResult,
    ImageModalityScore,
    ImageTaxonomyState,
    ImageValidationResult,
)
from image_detector.scoring import (
    calculate_image_epistemic_uncertainty,
    normalize_percentages,
    pool_bayesian_log_odds,
)
from image_detector.trainer import ImageDetectorTrainer
from image_detector.validator import ImageValidator, analyze_image

__all__ = [
    # Core Engines & Pipelines
    "ImageAIDetector",
    "ImageForensicPipeline",
    "ImageBatchProcessor",
    # Auxiliary Analysis Engines
    "FaceDeepfakeDetector",
    "ImageContentAnalyzer",
    "ImageModelAttributionEngine",
    "ImageProvenanceValidator",
    "ImageProfiler",
    "ImageDownloader",
    "build_nine_dimensions_dossier",
    "generate_newbie_explanation",
    # Training & Continual Improvement
    "ImageDetectorTrainer",
    "ImageSelfImprover",
    "ImageValidator",
    "ImageBenchmarkSuite",
    # Signal & Forensic Features
    "calculate_sensor_noise_profile",
    "calculate_surface_smoothness",
    "analyze_fft_radial_power_spectrum",
    "compute_ela",
    "extract_image_metadata",
    "generate_spatial_manipulation_heatmap",
    "detect_screenshot",
    "detect_inpainting_and_manipulation",
    "compute_file_hashes",
    "compute_pixel_entropy",
    "extract_all_image_details",
    "rgb_to_color_name",
    # Scoring
    "pool_bayesian_log_odds",
    "normalize_percentages",
    "calculate_image_epistemic_uncertainty",
    # Models
    "build_image_classifier",
    # Schemas
    "ImageModalityScore",
    "ImageForensicResult",
    "ImageValidationResult",
    "ImageFeedbackRecord",
    "ImageBenchmarkMetrics",
    "ImageTaxonomyState",
]
