"""
video_detector.config: Configuration parameters, thresholds, and paths for Video AI Detection.
Self-contained module configuration with zero outside dependencies.
"""
from __future__ import annotations

from pathlib import Path

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
DEFAULT_KEYFRAME_TIMELINE_PCT = 0.15
MAX_FILE_SIZE_MB = 500.0
MAX_DURATION_SECONDS = 3600.0
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Per-frame noise judgement (median-filter residual of the grey frame; every camera frame has about NOISE_BASELINE of it)
NOISE_BASELINE = 0.70
NOISE_AI_THRESHOLD = 2.0              # balanced sensitivity: noise above the baseline below this is AI-like
NOISE_AI_THRESHOLD_SENSITIVE = 2.4    # high / aggressive sensitivity

# Temporal Motion Variance Thresholds
MOTION_VAR_HIGH_WARPING = 140.0
MOTION_VAR_SUSPICIOUS_FLICKER = 75.0
MOTION_VAR_UNNATURAL_FREEZE = 0.8
MEAN_DELTA_HIGH_THRESHOLD = 32.0

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
