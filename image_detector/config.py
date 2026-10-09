"""
image_detector.config: Configuration parameters, thresholds, and paths for Image AI Detection.
Self-contained module configuration with zero outside dependencies.
"""
from __future__ import annotations

from pathlib import Path

# Paths
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
MODELS_DIR = MODULE_DIR / "models"
DATASET_DIR = MODULE_DIR / "dataset"

DEFAULT_CHECKPOINT = MODELS_DIR / "ai_detector.pt"
CHECKPOINT_LOG_FILE = MODELS_DIR / "CHECKPOINT_LOG.md"
RETRAIN_SUGGEST_THRESHOLD = 15  # queued corrections before the UI suggests a retrain
MEMORY_FILE = DATA_DIR / "image_memory.json"
CALIBRATION_FILE = DATA_DIR / "image_calibration.json"


def ensure_directories() -> None:
    """Safely creates runtime data directories when needed without import-time side effects."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


# Image Resolution & Format Settings
IMAGE_SIZE = 224
MIN_RESOLUTION = 64
MAX_FILE_SIZE_MB = 100.0
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Physical Forensic Baselines
# Natural optical camera sensor noise (PRNU): mu ~ 2.45, sigma ~ 0.65
REAL_NOISE_MU = 2.45
REAL_NOISE_SIGMA = 0.65
AI_NOISE_MU = 0.85
AI_NOISE_SIGMA = 0.45

# Surface Texture Smoothness (Bilateral Filter Edge Difference)
REAL_SMOOTH_MU = 1.95
REAL_SMOOTH_SIGMA = 0.50
AI_SMOOTH_MU = 0.75
AI_SMOOTH_SIGMA = 0.40

# 2D FFT Radial Decay (Natural 1/f^alpha field law)
FFT_DECAY_ALPHA_OPTICAL = 2.15
FFT_DECAY_ALPHA_JPEG = 2.68

# Canonical AI Canvas Resolutions (Flux, SDXL, Midjourney, DALL-E, Imagen)
CANONICAL_RESOLUTIONS = {
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1024, 1792),
    (1792, 1024),
    (896, 1152),
    (1152, 896),
    (896, 1200),
    (1200, 896),
    (1344, 768),
    (768, 1344),
    (1536, 1024),
    (1024, 1536),
    (1216, 832),
    (832, 1216),
    (1408, 704),
    (704, 1408),
    # 2026 Emerging Canvas Standards (Ideogram 2.0, Recraft v3, Midjourney v6.1, Flux.1)
    (1280, 720),
    (720, 1280),
    (1440, 960),
    (960, 1440),
    (1920, 1080),
    (1080, 1920),
}

# Canonical Device Screen Resolutions & Profiles (Mobile, Tablet, Laptop, Desktop)
CANONICAL_SCREEN_RESOLUTIONS = {
    # Mobile Phones
    (1080, 2400): ("Mobile Phone", "20:9 FHD+ (Samsung, Xiaomi, Redmi, OnePlus)"),
    (1080, 2340): ("Mobile Phone", "19.5:9 FHD+ (Redmi Note, iPhone 12/13/14, Xiaomi)"),
    (1080, 1920): ("Mobile Phone", "16:9 FHD Standard Smartphone"),
    (1170, 2532): ("Mobile Phone", "19.5:9 Super Retina XDR (iPhone 12/13/14)"),
    (1179, 2556): ("Mobile Phone", "19.5:9 Dynamic Island (iPhone 14/15/16 Pro)"),
    (1290, 2796): ("Mobile Phone", "19.5:9 Dynamic Island Max (iPhone Pro Max)"),
    (1284, 2778): ("Mobile Phone", "19.5:9 Super Retina XDR Max (iPhone 12/13 Pro Max)"),
    (1242, 2688): ("Mobile Phone", "19.5:9 Super Retina (iPhone XS Max / 11 Pro Max)"),
    (828, 1792): ("Mobile Phone", "19.5:9 Liquid Retina (iPhone XR / 11)"),
    (750, 1334): ("Mobile Phone", "16:9 Retina (iPhone 6/7/8/SE)"),
    (1440, 3120): ("Mobile Phone", "19.5:9 QHD+ (OnePlus, Pixel Pro)"),
    (1440, 3088): ("Mobile Phone", "19.3:9 Dynamic AMOLED (Samsung Galaxy Ultra)"),
    (1440, 3200): ("Mobile Phone", "20:9 WQHD+ (Samsung S20/S21 Ultra)"),
    (720, 1600): ("Mobile Phone", "20:9 HD+ Budget Smartphone"),
    (720, 1560): ("Mobile Phone", "19.5:9 HD+ Budget Smartphone"),
    # Tablets
    (2048, 1536): ("Tablet", "4:3 Retina (iPad Air / iPad mini)"),
    (2732, 2048): ("Tablet", "4:3 Liquid Retina (iPad Pro 12.9)"),
    (2388, 1668): ("Tablet", "4.3:3 Liquid Retina (iPad Pro 11)"),
    (2360, 1640): ("Tablet", "4.3:3 Liquid Retina (iPad Air 10.9)"),
    (2560, 1600): ("Tablet / Laptop", "16:10 WQXGA (Galaxy Tab / MacBook Air)"),
    (1920, 1200): ("Tablet / Laptop", "16:10 WUXGA (Android Tablet / Laptop)"),
    (2880, 1920): ("Tablet / Laptop", "3:2 PixelSense (Surface Pro)"),
    # Laptops & Desktops
    (1920, 1080): ("Laptop / Desktop", "16:9 Full HD 1080p Display"),
    (2560, 1440): ("Desktop / Laptop", "16:9 2K QHD 1440p Display"),
    (3840, 2160): ("Desktop", "16:9 4K UHD 2160p Display"),
    (1366, 768): ("Laptop", "16:9 HD Laptop Display"),
    (1600, 900): ("Laptop / Desktop", "16:9 HD+ Display"),
    (1280, 720): ("Laptop / Monitor", "16:9 Standard HD 720p"),
    (2880, 1800): ("Laptop", "16:10 Retina Display (MacBook Pro 15)"),
    (3024, 1964): ("Laptop", "16:10 Liquid Retina XDR (MacBook Pro 14)"),
    (3456, 2234): ("Laptop", "16:10 Liquid Retina XDR (MacBook Pro 16)"),
    (2560, 1080): ("Desktop", "21:9 UltraWide Full HD Display"),
    (3440, 1440): ("Desktop", "21:9 UltraWide QHD Display"),
}

# Known AI Software Fingerprints in Metadata (EXIF/XMP Software, Creator, or Processing tags).
# Single source of truth: features.extract_image_metadata matches against this list (provenance reads its result),
# so there is exactly one place to update as new tools ship.
KNOWN_AI_SOFTWARE_SIGNATURES = [
    "midjourney",
    "stable diffusion",
    "dall-e",
    "novelai",
    "adobe firefly",
    "comfyui",
    "civitai",
    "automatic1111",
    "invokeai",
    "fooocus",
    "bing image creator",
    "imagen",
    "gemini",
    "flux",
    "runway",
    "kling",
    "sora",
    "pika",
    "luma",
    "ideogram",
    "recraft",
    "magnific",
    "topaz photo ai",
    "liveportrait",
    "ic-light",
]

# Sensitivity Mode Prior Log-Odds
SENSITIVITY_PRIORS = {
    "aggressive": 0.40,  # Prior P(AI) ~ 71%
    "high": 0.20,        # Prior P(AI) ~ 61%
    "balanced": 0.00,    # Neutral Prior P(AI) = 50%
}

# Default Decision Thresholds
AI_THRESHOLD_HIGH = 50.0
AI_THRESHOLD_BALANCED = 55.0
REAL_THRESHOLD = 55.0


# Unauthenticated-metadata trust policy. Camera EXIF is trivially forgeable, so it must not be allowed to cancel
# strong physical evidence. When the raw (pre-discount) sensor-noise + surface-smoothness evidence leans synthetic
# by at least EXIF_CONTRADICTION_LR (base-10 log-odds), camera EXIF is treated as UNTRUSTED: the physical-signal
# discount is not applied and the hardware credit shrinks from EXIF_TRUSTED_CREDIT to EXIF_UNTRUSTED_CREDIT.
EXIF_CONTRADICTION_LR = 2.0

# Images no larger than this (longest side, px) have been downscaled enough that sensor-noise and smoothness
# statistics no longer separate camera photos from synthetic ones. Kept below 512, the smallest common generator canvas.
SMALL_IMAGE_MAX_SIDE = 400

# Weight of the "digital art" (saturation / ink-line) finding in the AI score, as base-10 log-odds per unit of its confidence.
# A likelihood ratio cannot exceed P(finding | AI) / P(finding | real): on the real photo sets it fires on 4-20 % of photographs
# (colourful corals, fish, sunsets, skin), so even if every AI image tripped it the ratio is at most about 5-8, i.e. below 1.0.
DIGITAL_ART_LR_SCALE = 0.8
EXIF_TRUSTED_CREDIT = -1.4
EXIF_UNTRUSTED_CREDIT = -0.3
