"""
video_detector: Completely independent, self-contained, feedback-calibrated Video AI Detection package (see learner.py for exactly what the calibration does and does not do).
Contains its own:
- Configurations & temporal baselines (video_detector.config)
- Schemas & data structures (video_detector.schemas)
- Neural models & checkpoints (video_detector.models)
- Temporal analysis & optical flow routines (video_detector.temporal)
- Frame extraction & audio isolation (video_detector.extractor)
- Temporal evidence pooling & scoring (video_detector.scoring)
- Video container, codec, & integrity validators (video_detector.validator)
- Continual learning & dynamic calibration (video_detector.learner)
- Neural classifier trainer (video_detector.trainer)
- Benchmark evaluation suite (video_detector.benchmarks)
- Main detection engine (video_detector.detector.VideoAIDetector)
- Facial deepfake detection (video_detector.face.VideoFaceDeepfakeDetector)
- Scene & content intelligence (video_detector.content.VideoContentAnalyzer)
- Generator model attribution (video_detector.attribution.VideoModelAttributionEngine)
- C2PA marker-presence & container atoms (video_detector.provenance.VideoProvenanceValidator)
- Stream profiling & specifications (video_detector.profiler.VideoProfiler)
- Cross-modal audio-visual synchronization (video_detector.cross_modal.CrossModalConsistencyEngine)
- Safe URL video downloader (video_detector.downloader.VideoDownloader)
- End-to-end forensic pipeline (video_detector.pipeline.VideoForensicPipeline)
- High-speed batch processor (video_detector.batch.VideoBatchProcessor)
"""
from core.lazy import install_lazy_exports

_LAZY_EXPORTS = {
    "CrossModalConsistencyEngine": ("video_detector.cross_modal", "CrossModalConsistencyEngine"),
    "VideoAIDetector": ("video_detector.detector", "VideoAIDetector"),
    "VideoBatchProcessor": ("video_detector.batch", "VideoBatchProcessor"),
    "VideoBenchmarkMetrics": ("video_detector.schemas", "VideoBenchmarkMetrics"),
    "VideoBenchmarkSuite": ("video_detector.benchmarks", "VideoBenchmarkSuite"),
    "VideoContentAnalyzer": ("video_detector.content", "VideoContentAnalyzer"),
    "VideoDetectorTrainer": ("video_detector.trainer", "VideoDetectorTrainer"),
    "VideoDownloader": ("video_detector.downloader", "VideoDownloader"),
    "VideoFaceDeepfakeDetector": ("video_detector.face", "VideoFaceDeepfakeDetector"),
    "VideoFeedbackRecord": ("video_detector.schemas", "VideoFeedbackRecord"),
    "VideoForensicPipeline": ("video_detector.pipeline", "VideoForensicPipeline"),
    "VideoForensicResult": ("video_detector.schemas", "VideoForensicResult"),
    "VideoFrameExtractor": ("video_detector.extractor", "VideoFrameExtractor"),
    "VideoModalityScore": ("video_detector.schemas", "VideoModalityScore"),
    "VideoModelAttributionEngine": ("video_detector.attribution", "VideoModelAttributionEngine"),
    "VideoProfiler": ("video_detector.profiler", "VideoProfiler"),
    "VideoProvenanceValidator": ("video_detector.provenance", "VideoProvenanceValidator"),
    "VideoSelfImprover": ("video_detector.learner", "VideoSelfImprover"),
    "VideoTemporalSegment": ("video_detector.schemas", "VideoTemporalSegment"),
    "VideoTemporalTransitionModel": ("video_detector.models.backbone", "VideoTemporalTransitionModel"),
    "VideoValidationResult": ("video_detector.schemas", "VideoValidationResult"),
    "VideoValidator": ("video_detector.validator", "VideoValidator"),
    "build_video_nine_dimensions_dossier": ("video_detector.explain", "build_video_nine_dimensions_dossier"),
    "calculate_video_epistemic_uncertainty": ("video_detector.scoring", "calculate_video_epistemic_uncertainty"),
    "compute_file_hashes": ("video_detector.profiler", "compute_file_hashes"),
    "compute_interframe_motion_variance": ("video_detector.temporal", "compute_interframe_motion_variance"),
    "detect_diffusion_flickering": ("video_detector.temporal", "detect_diffusion_flickering"),
    "evaluate_cross_modal_consistency": ("video_detector.cross_modal", "evaluate_cross_modal_consistency"),
    "evaluate_video_decision": ("video_detector.scoring", "evaluate_video_decision"),
    "generate_video_newbie_explanation": ("video_detector.explain", "generate_video_newbie_explanation"),
    "group_temporal_segments": ("video_detector.temporal", "group_temporal_segments"),
    "normalize_percentages": ("video_detector.scoring", "normalize_percentages"),
    "pool_video_temporal_score": ("video_detector.scoring", "pool_video_temporal_score"),
    "validate_video_stream": ("video_detector.validator", "validate_video_stream"),
}
install_lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    # Core Engine & Pipelines
    "VideoAIDetector",
    "VideoForensicPipeline",
    "VideoBatchProcessor",
    # Auxiliary Analysis Engines
    "VideoFaceDeepfakeDetector",
    "VideoContentAnalyzer",
    "VideoModelAttributionEngine",
    "VideoProvenanceValidator",
    "VideoProfiler",
    "CrossModalConsistencyEngine",
    "VideoDownloader",
    # Training & Continual Improvement
    "VideoDetectorTrainer",
    "VideoSelfImprover",
    "VideoValidator",
    "VideoFrameExtractor",
    "VideoBenchmarkSuite",
    # Explainability & Taxonomy
    "build_video_nine_dimensions_dossier",
    "generate_video_newbie_explanation",
    # Temporal Analysis
    "compute_interframe_motion_variance",
    "detect_diffusion_flickering",
    "group_temporal_segments",
    "compute_file_hashes",
    # Scoring
    "pool_video_temporal_score",
    "normalize_percentages",
    "calculate_video_epistemic_uncertainty",
    "evaluate_video_decision",
    # Models
    "VideoTemporalTransitionModel",
    # Schemas
    "VideoModalityScore",
    "VideoTemporalSegment",
    "VideoForensicResult",
    "VideoValidationResult",
    "VideoFeedbackRecord",
    "VideoBenchmarkMetrics",
]
