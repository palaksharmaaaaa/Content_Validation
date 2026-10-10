"""core.limits: the one place that says how large an upload may be.

Per file and per batch, for each of image, video and audio. The defaults below can be changed without touching code, in this order of
precedence:

1. an environment variable, e.g. ``OMNIFORENSICS_MAX_IMAGE_FILE_MB=20`` or ``OMNIFORENSICS_MAX_VIDEO_BATCH_MB=1000``;
2. a ``limits.toml`` file in the repository root (or the file named by ``OMNIFORENSICS_LIMITS_FILE``), see ``limits.toml``;
3. the defaults in ``DEFAULTS``.

A value that is not a positive finite number is ignored (with a warning) and the next source is used. Streamlit refuses any upload larger
than ``server.maxUploadSize`` before this code sees it, so that setting in ``.streamlit/config.toml`` must be at least the largest
per-file limit; ``streamlit_upload_ceiling_mb`` reports what is configured so the app can warn when it is too low.
"""
from __future__ import annotations

import logging
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MODALITIES = ("image", "video", "audio")
# (maximum size of one file, maximum total size of one batch), in megabytes (1 MB = 1024 * 1024 bytes)
DEFAULTS: Dict[str, Dict[str, float]] = {
    "image": {"file_mb": 10.0, "batch_mb": 500.0},
    "video": {"file_mb": 100.0, "batch_mb": 500.0},
    "audio": {"file_mb": 10.0, "batch_mb": 500.0},
}
LIMITS_FILE = Path(__file__).resolve().parent.parent / "limits.toml"
BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class Limits:
    file_mb: float
    batch_mb: float

    @property
    def file_bytes(self) -> int:
        return int(self.file_mb * BYTES_PER_MB)

    @property
    def batch_bytes(self) -> int:
        return int(self.batch_mb * BYTES_PER_MB)


def _positive(value: Any, source: str) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number) or number <= 0:
        logger.warning("Ignoring size limit %r from %s: it must be a positive number of megabytes", value, source)
        return None
    return number


def _file_values() -> Dict[str, Any]:
    path = Path(os.environ.get("OMNIFORENSICS_LIMITS_FILE") or LIMITS_FILE)
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Ignoring unreadable limits file %s: %s", path, exc)
        return {}


def limits_for(modality: str) -> Limits:
    """The per-file and per-batch size limits for ``image``, ``video`` or ``audio``."""
    name = str(modality).lower()
    if name not in DEFAULTS:
        raise ValueError(f"modality must be one of {MODALITIES}, not {modality!r}")
    table = _file_values().get(name)
    table = table if isinstance(table, dict) else {}
    resolved: Dict[str, float] = {}
    for key, default in DEFAULTS[name].items():
        env_name = f"OMNIFORENSICS_MAX_{name.upper()}_{key.upper()}"
        value = _positive(os.environ[env_name], f"environment variable {env_name}") if env_name in os.environ else None
        if value is None and key in table:
            value = _positive(table[key], f"limits file ([{name}] {key})")
        resolved[key] = default if value is None else value
    return Limits(file_mb=resolved["file_mb"], batch_mb=resolved["batch_mb"])


def streamlit_upload_ceiling_mb() -> Optional[float]:
    """Streamlit's configured ``server.maxUploadSize`` in MB, or None if it cannot be read."""
    try:
        from streamlit import config

        value = config.get_option("server.maxUploadSize")
        return float(value) if value is not None else None
    except Exception:
        return None
