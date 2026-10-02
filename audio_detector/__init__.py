"""
audio_detector: Completely independent, self-contained, and self-improving Audio AI Detection package.
Contains its own:
- Configurations & acoustic thresholds (audio_detector.config)
- Schemas & data structures (audio_detector.schemas)
- Neural models & checkpoints (audio_detector.models)
- Acoustic feature extraction & STFT analysis (audio_detector.features)
- Evidence pooling & scoring (audio_detector.scoring)
- Audio stream & format validator (audio_detector.validator)
- Continual learning & dynamic calibration (audio_detector.learner)
- Neural classifier trainer (audio_detector.trainer)
- Benchmark evaluation suite (audio_detector.benchmarks)
- Main detection engine (audio_detector.detector.AudioAIDetector)
- Scene & acoustic delivery tone (audio_detector.content.AudioContentAnalyzer)
- Voice & music synthesizer attribution (audio_detector.attribution.AudioModelAttributionEngine)
- Cryptographic C2PA & chunk provenance (audio_detector.provenance.AudioProvenanceValidator)
- Signal profiling & dynamic range (audio_detector.profiler.AudioProfiler)
- Safe URL audio downloader (audio_detector.downloader.AudioDownloader)
- End-to-end forensic pipeline (audio_detector.pipeline.AudioForensicPipeline)
- High-speed batch processor (audio_detector.batch.AudioBatchProcessor)
"""
from audio_detector.attribution import AudioModelAttributionEngine
from audio_detector.batch import AudioBatchProcessor
from audio_detector.benchmarks import AudioBenchmarkSuite
from audio_detector.content import AudioContentAnalyzer
from audio_detector.detector import AudioAIDetector
from audio_detector.downloader import AudioDownloader
from audio_detector.explain import build_audio_nine_dimensions_dossier, generate_audio_newbie_explanation
from audio_detector.features import (
    compute_spectral_features,
    generate_spectrogram_image,
    segment_audio_temporal,
)
from audio_detector.learner import AudioSelfImprover
from audio_detector.models.backbone import AudioClassifierNet, build_audio_classifier
from audio_detector.pipeline import AudioForensicPipeline
from audio_detector.profiler import AudioProfiler, compute_file_hashes
from audio_detector.provenance import AudioProvenanceValidator
from audio_detector.schemas import (
    AudioBenchmarkMetrics,
    AudioFeedbackRecord,
    AudioForensicResult,
    AudioModalityScore,
    AudioTemporalSegment,
    AudioValidationResult,
)
from audio_detector.scoring import (
    calculate_audio_epistemic_uncertainty,
    evaluate_audio_decision,
    normalize_percentages,
    pool_acoustic_evidence,
)
from audio_detector.trainer import AudioDetectorTrainer
from audio_detector.validator import AudioValidator

__all__ = [
    # Core Engine & Pipelines
    "AudioAIDetector",
    "AudioForensicPipeline",
    "AudioBatchProcessor",
    # Auxiliary Analysis Engines
    "AudioContentAnalyzer",
    "AudioModelAttributionEngine",
    "AudioProvenanceValidator",
    "AudioProfiler",
    "AudioDownloader",
    # Training & Continual Improvement
    "AudioDetectorTrainer",
    "AudioSelfImprover",
    "AudioValidator",
    "AudioBenchmarkSuite",
    # Explainability & Taxonomy
    "build_audio_nine_dimensions_dossier",
    "generate_audio_newbie_explanation",
    # Features & Signal Processing
    "compute_spectral_features",
    "generate_spectrogram_image",
    "segment_audio_temporal",
    "compute_file_hashes",
    # Scoring
    "pool_acoustic_evidence",
    "normalize_percentages",
    "calculate_audio_epistemic_uncertainty",
    "evaluate_audio_decision",
    # Models
    "AudioClassifierNet",
    "build_audio_classifier",
    # Schemas
    "AudioModalityScore",
    "AudioTemporalSegment",
    "AudioForensicResult",
    "AudioValidationResult",
    "AudioFeedbackRecord",
    "AudioBenchmarkMetrics",
]
