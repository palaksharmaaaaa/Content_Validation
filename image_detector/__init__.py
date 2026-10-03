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
- C2PA marker-presence & EXIF provenance (image_detector.provenance.ImageProvenanceValidator)
- Signal profiling & entropy (image_detector.profiler.ImageProfiler)
- Safe URL image downloader (image_detector.downloader.ImageDownloader)
- End-to-end forensic pipeline (image_detector.pipeline.ImageForensicPipeline)
- High-speed batch processor (image_detector.batch.ImageBatchProcessor)
"""
from core.lazy import install_lazy_exports

_LAZY_EXPORTS = {
    "FaceDeepfakeDetector": ("image_detector.face", "FaceDeepfakeDetector"),
    "FeatureBankDataset": ("image_detector.feature_store", "FeatureBankDataset"),
    "FeatureClassifierHead": ("image_detector.feature_store", "FeatureClassifierHead"),
    "FeatureStore": ("image_detector.feature_store", "FeatureStore"),
    "ImageAIDetector": ("image_detector.detector", "ImageAIDetector"),
    "ImageBatchProcessor": ("image_detector.batch", "ImageBatchProcessor"),
    "ImageBenchmarkMetrics": ("image_detector.schemas", "ImageBenchmarkMetrics"),
    "ImageBenchmarkSuite": ("image_detector.benchmarks", "ImageBenchmarkSuite"),
    "ImageContentAnalyzer": ("image_detector.content", "ImageContentAnalyzer"),
    "ImageDetectorTrainer": ("image_detector.trainer", "ImageDetectorTrainer"),
    "ImageDownloader": ("image_detector.downloader", "ImageDownloader"),
    "ImageFeedbackRecord": ("image_detector.schemas", "ImageFeedbackRecord"),
    "ImageForensicPipeline": ("image_detector.pipeline", "ImageForensicPipeline"),
    "ImageForensicResult": ("image_detector.schemas", "ImageForensicResult"),
    "ImageModalityScore": ("image_detector.schemas", "ImageModalityScore"),
    "ImageModelAttributionEngine": ("image_detector.attribution", "ImageModelAttributionEngine"),
    "ImageProfiler": ("image_detector.profiler", "ImageProfiler"),
    "ImageProvenanceValidator": ("image_detector.provenance", "ImageProvenanceValidator"),
    "ImageSelfImprover": ("image_detector.learner", "ImageSelfImprover"),
    "ImageTaxonomyState": ("image_detector.schemas", "ImageTaxonomyState"),
    "ImageValidationResult": ("image_detector.schemas", "ImageValidationResult"),
    "ImageValidator": ("image_detector.validator", "ImageValidator"),
    "analyze_fft_radial_power_spectrum": ("image_detector.features", "analyze_fft_radial_power_spectrum"),
    "analyze_image": ("image_detector.validator", "analyze_image"),
    "build_image_classifier": ("image_detector.models.backbone", "build_image_classifier"),
    "build_nine_dimensions_dossier": ("image_detector.explain", "build_nine_dimensions_dossier"),
    "calculate_image_epistemic_uncertainty": ("image_detector.scoring", "calculate_image_epistemic_uncertainty"),
    "calculate_sensor_noise_profile": ("image_detector.features", "calculate_sensor_noise_profile"),
    "calculate_surface_smoothness": ("image_detector.features", "calculate_surface_smoothness"),
    "compute_ela": ("image_detector.features", "compute_ela"),
    "compute_file_hashes": ("image_detector.profiler", "compute_file_hashes"),
    "compute_pixel_entropy": ("image_detector.profiler", "compute_pixel_entropy"),
    "detect_inpainting_and_manipulation": ("image_detector.features", "detect_inpainting_and_manipulation"),
    "detect_screenshot": ("image_detector.features", "detect_screenshot"),
    "extract_all_image_details": ("image_detector.profiler", "extract_all_image_details"),
    "extract_image_metadata": ("image_detector.features", "extract_image_metadata"),
    "generate_newbie_explanation": ("image_detector.explain", "generate_newbie_explanation"),
    "generate_spatial_manipulation_heatmap": ("image_detector.features", "generate_spatial_manipulation_heatmap"),
    "normalize_percentages": ("image_detector.scoring", "normalize_percentages"),
    "pool_bayesian_log_odds": ("image_detector.scoring", "pool_bayesian_log_odds"),
    "rgb_to_color_name": ("image_detector.profiler", "rgb_to_color_name"),
}
install_lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    # Core Engines & Pipelines
    "ImageAIDetector",
    "ImageForensicPipeline",
    "ImageBatchProcessor",
    "FeatureStore",
    "FeatureBankDataset",
    "FeatureClassifierHead",
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
