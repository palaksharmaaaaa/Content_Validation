"""core.calibration_io: a calibration file is data somebody can edit or corrupt; it is checked before the detectors trust it.

Calibration files are small JSON documents of weights, thresholds and offsets that feedback nudges. A hand-edited, truncated or
tampered file (a NaN weight, a threshold of 1e9, a string where a number belongs) would otherwise flow straight into the scoring maths.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict

logger = logging.getLogger(__name__)

# (lowest, highest) accepted for a number, by the kind of section it sits in; anything outside is reset to the default.
_RANGES = {"weights": (0.0, 2.0), "offsets": (-1.0, 1.0), "thresholds": (0.0, 1.0e6)}


def _range_for(path: str):
    for kind, bounds in _RANGES.items():
        if kind in path:
            return bounds
    return (-1.0e9, 1.0e9)


def _clean(loaded: Any, default: Any, path: str) -> Any:
    if isinstance(default, dict):
        source = loaded if isinstance(loaded, dict) else {}
        out: Dict[str, Any] = {}
        for key, dv in default.items():
            out[key] = _clean(source.get(key), dv, f"{path}.{key}")
        for key, value in source.items():                                # extra keys survive only if they are plain finite values
            if key not in default and (isinstance(value, (str, bool)) or (isinstance(value, (int, float)) and math.isfinite(value))):
                out[key] = value
        return out
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return loaded if loaded is not None and type(loaded) is type(default) else default
    low, high = _range_for(path)
    if isinstance(loaded, bool) or not isinstance(loaded, (int, float)) or not math.isfinite(loaded) or not low <= loaded <= high:
        if loaded is not None:
            logger.warning("Calibration value %s=%r is not usable; using the default %r", path.lstrip("."), loaded, default)
        return default
    return loaded


def sanitize_calibration(loaded: Any, default: Dict[str, Any]) -> Dict[str, Any]:
    """``loaded`` with every missing, non-numeric, non-finite or out-of-range value replaced by its default, section by section."""
    cleaned = _clean(loaded, default, "")
    samples = cleaned.get("samples_processed")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 0:
        cleaned["samples_processed"] = 0
    return cleaned
