from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MAX_FILE_SIZE_MB = 100

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
}

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}

# Video analysis
DEFAULT_VIDEO_SAMPLE_FRAMES = 30

# Image quality
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200

BLANK_VARIANCE_THRESHOLD = 5.0
BLANK_EXTREME_PIXEL_RATIO = 0.98

BLUR_THRESHOLD = 50.0

DARK_MEAN_THRESHOLD = 30.0
BRIGHT_MEAN_THRESHOLD = 245.0

# Video validity
MIN_CONTENT_COVERAGE = 0.70
PARTIAL_CONTENT_COVERAGE = 0.30

# AI detector
AI_DETECTOR_MODEL = "Reju983/ai-generated-image-detector"

AI_LIKELY_THRESHOLD = 0.70
AI_UNCERTAIN_LOW = 0.30
AI_UNCERTAIN_HIGH = 0.70