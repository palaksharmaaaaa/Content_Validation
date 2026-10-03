"""core.forensics.config: caps and operator-supplied paths. No dependencies."""
from __future__ import annotations

import os
from pathlib import Path

# Log-odds (base-10, positive = toward AI) caps for hybrid scoring.
PER_FINDING_LLR_CAP = 0.25
EXPLICIT_GENERATOR_LLR_CAP = 0.40
TOTAL_LLR_CAP = 0.40

ENV_HARDBLOCK_FILE = "OMNI_HARDBLOCK_SHA256_FILE"
ENV_CONTEXT_INDEX = "OMNI_CONTEXT_HASH_INDEX"
ENV_AUDIO_FP_INDEX = "OMNI_AUDIO_FP_INDEX"
ENV_VIDEO_FP_INDEX = "OMNI_VIDEO_FP_INDEX"

_CORE_DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_HARDBLOCK_FILE = _CORE_DATA / "hardblock_sha256.txt"


def hardblock_file() -> Path:
    """Path of the hard-block SHA-256 list (``OMNI_HARDBLOCK_SHA256_FILE`` or ``core/data/hardblock_sha256.txt``)."""
    env = os.environ.get(ENV_HARDBLOCK_FILE)
    return Path(env) if env else DEFAULT_HARDBLOCK_FILE


def context_index_file(default: Path) -> Path:
    """Path of the image re-use index (``OMNI_CONTEXT_HASH_INDEX`` or the given default)."""
    env = os.environ.get(ENV_CONTEXT_INDEX)
    return Path(env) if env else default


def audio_fp_index_file(default: Path) -> Path:
    """Path of the audio fingerprint index (``OMNI_AUDIO_FP_INDEX`` or the given default)."""
    env = os.environ.get(ENV_AUDIO_FP_INDEX)
    return Path(env) if env else default


def video_fp_index_file(default: Path) -> Path:
    """Path of the video fingerprint index (``OMNI_VIDEO_FP_INDEX`` or the given default)."""
    env = os.environ.get(ENV_VIDEO_FP_INDEX)
    return Path(env) if env else default
