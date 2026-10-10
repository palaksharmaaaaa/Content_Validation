"""
audio_detector.config: Configuration parameters, thresholds, and paths for Audio AI Detection.
Self-contained module configuration with zero outside dependencies.
"""
from __future__ import annotations

from pathlib import Path

from core.limits import limits_for

# Paths
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
MODELS_DIR = MODULE_DIR / "models"
DATASET_DIR = MODULE_DIR / "dataset"

DEFAULT_AUDIO_CHECKPOINT = MODELS_DIR / "audio_detector.pt"
CHECKPOINT_LOG_FILE = MODELS_DIR / "CHECKPOINT_LOG.md"
RETRAIN_SUGGEST_THRESHOLD = 15  # queued corrections before the UI suggests a retrain
MEMORY_FILE = DATA_DIR / "audio_memory.json"
CALIBRATION_FILE = DATA_DIR / "audio_calibration.json"


# Audio Processing & Limits
TARGET_SAMPLE_RATE = 16000
MAX_FILE_SIZE_MB = limits_for("audio").file_mb      # core/limits.py: default, limits.toml or environment variable
MAX_DURATION_SECONDS = 3600.0
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a"}

# Vocoder High-Frequency Cutoff Bands (Hz)
VOCODER_CUTOFF_BAND_LOW_MIN = 6500
VOCODER_CUTOFF_BAND_LOW_MAX = 8200
VOCODER_CUTOFF_BAND_HIGH_MIN = 15000
VOCODER_CUTOFF_BAND_HIGH_MAX = 16500

# Wiener Spectral Flatness Limits
SYNTHETIC_FLATNESS_LOW_THRESHOLD = 0.002
SYNTHETIC_FLATNESS_HIGH_THRESHOLD = 0.45

# Digital Zero Silence Ratio Limits
DIGITAL_SILENCE_RATIO_THRESHOLD = 0.12

# Acoustic Evidence Weights
DEFAULT_WEIGHT_VOCODER = 0.40
DEFAULT_WEIGHT_FLATNESS = 0.30
DEFAULT_WEIGHT_SILENCE = 0.20
DEFAULT_WEIGHT_HF_RATIO = 0.10

# Temporal Segmentation
DEFAULT_WINDOW_SECONDS = 3.0

# Decision Thresholds
AI_THRESHOLD_HIGH = 50.0
AI_THRESHOLD_BALANCED = 55.0
REAL_THRESHOLD = 55.0
