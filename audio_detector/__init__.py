"""
audio_detector: Completely independent, self-contained, feedback-calibrated Audio AI Detection package (see learner.py for exactly what the calibration does and does not do).
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
- C2PA marker-presence & chunk provenance (audio_detector.provenance.AudioProvenanceValidator)
- Signal profiling & dynamic range (audio_detector.profiler.AudioProfiler)
- Safe URL audio downloader (audio_detector.downloader.AudioDownloader)
- End-to-end forensic pipeline (audio_detector.pipeline.AudioForensicPipeline)
- High-speed batch processor (audio_detector.batch.AudioBatchProcessor)
"""
from core.lazy import install_lazy_exports

_LAZY_EXPORTS = {
    "AudioAIDetector": ("audio_detector.detector", "AudioAIDetector"),
    "AudioBatchProcessor": ("audio_detector.batch", "AudioBatchProcessor"),
    "AudioBenchmarkMetrics": ("audio_detector.schemas", "AudioBenchmarkMetrics"),
    "AudioBenchmarkSuite": ("audio_detector.benchmarks", "AudioBenchmarkSuite"),
    "AudioClassifierNet": ("audio_detector.models.backbone", "AudioClassifierNet"),
    "AudioContentAnalyzer": ("audio_detector.content", "AudioContentAnalyzer"),
    "AudioDetectorTrainer": ("audio_detector.trainer", "AudioDetectorTrainer"),
    "AudioDownloader": ("audio_detector.downloader", "AudioDownloader"),
    "AudioFeedbackRecord": ("audio_detector.schemas", "AudioFeedbackRecord"),
    "AudioForensicPipeline": ("audio_detector.pipeline", "AudioForensicPipeline"),
    "AudioForensicResult": ("audio_detector.schemas", "AudioForensicResult"),
    "AudioModalityScore": ("audio_detector.schemas", "AudioModalityScore"),
    "AudioModelAttributionEngine": ("audio_detector.attribution", "AudioModelAttributionEngine"),
    "AudioProfiler": ("audio_detector.profiler", "AudioProfiler"),
    "AudioProvenanceValidator": ("audio_detector.provenance", "AudioProvenanceValidator"),
    "AudioSelfImprover": ("audio_detector.learner", "AudioSelfImprover"),
    "AudioTemporalSegment": ("audio_detector.schemas", "AudioTemporalSegment"),
    "AudioValidationResult": ("audio_detector.schemas", "AudioValidationResult"),
    "AudioValidator": ("audio_detector.validator", "AudioValidator"),
    "build_audio_classifier": ("audio_detector.models.backbone", "build_audio_classifier"),
    "build_audio_nine_dimensions_dossier": ("audio_detector.explain", "build_audio_nine_dimensions_dossier"),
    "calculate_audio_epistemic_uncertainty": ("audio_detector.scoring", "calculate_audio_epistemic_uncertainty"),
    "compute_file_hashes": ("audio_detector.profiler", "compute_file_hashes"),
    "compute_spectral_features": ("audio_detector.features", "compute_spectral_features"),
    "evaluate_audio_decision": ("audio_detector.scoring", "evaluate_audio_decision"),
    "generate_audio_newbie_explanation": ("audio_detector.explain", "generate_audio_newbie_explanation"),
    "generate_spectrogram_image": ("audio_detector.features", "generate_spectrogram_image"),
    "normalize_percentages": ("audio_detector.scoring", "normalize_percentages"),
    "pool_acoustic_evidence": ("audio_detector.scoring", "pool_acoustic_evidence"),
    "segment_audio_temporal": ("audio_detector.features", "segment_audio_temporal"),
}
install_lazy_exports(__name__, _LAZY_EXPORTS, globals())

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
