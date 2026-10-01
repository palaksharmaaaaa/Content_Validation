"""
video_detector: Completely independent, self-contained, and self-improving Video AI Detection package.
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
- Cryptographic C2PA & container atoms (video_detector.provenance.VideoProvenanceValidator)
- Stream profiling & specifications (video_detector.profiler.VideoProfiler)
- Cross-modal audio-visual synchronization (video_detector.cross_modal.CrossModalConsistencyEngine)
- Safe URL video downloader (video_detector.downloader.VideoDownloader)
- End-to-end forensic pipeline (video_detector.pipeline.VideoForensicPipeline)
- High-speed batch processor (video_detector.batch.VideoBatchProcessor)
"""
from video_detector.attribution import VideoModelAttributionEngine
from video_detector.batch import VideoBatchProcessor
from video_detector.benchmarks import VideoBenchmarkSuite
from video_detector.content import VideoContentAnalyzer
from video_detector.cross_modal import CrossModalConsistencyEngine, evaluate_cross_modal_consistency
from video_detector.detector import VideoAIDetector
from video_detector.downloader import VideoDownloader
from video_detector.extractor import VideoFrameExtractor
from video_detector.face import VideoFaceDeepfakeDetector
from video_detector.learner import VideoSelfImprover
from video_detector.models.backbone import VideoTemporalTransitionModel
from video_detector.pipeline import VideoForensicPipeline
from video_detector.profiler import VideoProfiler, compute_file_hashes
from video_detector.provenance import VideoProvenanceValidator
from video_detector.schemas import (
    VideoBenchmarkMetrics,
    VideoFeedbackRecord,
    VideoForensicResult,
    VideoModalityScore,
    VideoTemporalSegment,
    VideoValidationResult,
)
from video_detector.scoring import (
    calculate_video_epistemic_uncertainty,
    evaluate_video_decision,
    normalize_percentages,
    pool_video_temporal_score,
)
from video_detector.temporal import (
    compute_interframe_motion_variance,
    detect_diffusion_flickering,
    group_temporal_segments,
)
from video_detector.trainer import VideoDetectorTrainer
from video_detector.validator import VideoValidator, analyze_video

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
