"""
video_detector.config: Configuration parameters, thresholds, and paths for Video AI Detection.
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

DEFAULT_VIDEO_CHECKPOINT = MODELS_DIR / "video_detector.pt"
CHECKPOINT_LOG_FILE = MODELS_DIR / "CHECKPOINT_LOG.md"
RETRAIN_SUGGEST_THRESHOLD = 15  # queued corrections before the UI suggests a retrain
MEMORY_FILE = DATA_DIR / "video_memory.json"
CALIBRATION_FILE = DATA_DIR / "video_calibration.json"


# Sampling & Limits
DEFAULT_MAX_FRAMES = 30
MIN_RESOLUTION = 64          # shorter side, px: below this the noise and texture statistics the detector reads are not meaningful
MAX_FILE_SIZE_MB = limits_for("video").file_mb      # core/limits.py: default, limits.toml or environment variable
MAX_DURATION_SECONDS = 3600.0
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Per-frame noise judgement: the cut-offs live with the shared scorer (core.frame_scorer).
from core.frame_scorer import (  # noqa: E402,F401
    NOISE_AI_THRESHOLD,
    NOISE_AI_THRESHOLD_SENSITIVE,
    NOISE_BASELINE,
    SMOOTH_AI_THRESHOLD,
    SMOOTH_AI_THRESHOLD_SENSITIVE,
)

# Temporal Motion Variance Thresholds
MOTION_VAR_HIGH_WARPING = 140.0
MOTION_VAR_SUSPICIOUS_FLICKER = 75.0
MOTION_VAR_UNNATURAL_FREEZE = 0.8
MEAN_DELTA_HIGH_THRESHOLD = 32.0
MEAN_DELTA_SUSPICIOUS_THRESHOLD = 22.0

# Diffusion Flickering
FLICKER_RATIO_THRESHOLD = 0.65
MEAN_LUM_JUMP_THRESHOLD = 3.5

# Temporal Weights
DEFAULT_WEIGHT_FRAME_AI = 0.60
DEFAULT_WEIGHT_WARPING = 0.25
DEFAULT_WEIGHT_FLICKER = 0.15

# Decision Thresholds
AI_THRESHOLD_HIGH = 50.0
AI_THRESHOLD_BALANCED = 55.0
REAL_THRESHOLD = 55.0
