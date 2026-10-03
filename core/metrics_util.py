"""core.metrics_util: shared sanitiser that turns arbitrary detector metric values into JSON-safe scalars."""
from __future__ import annotations

from typing import Any


def sanitize_metric_value(val: Any) -> Any:
    """Returns a JSON-safe value, or None for arrays/unsupported objects (which callers drop)."""
    if hasattr(val, "shape"):
        # 0-d numpy scalars are real metrics (previously dropped by the per-learner copies); arrays are not.
        return val.item() if getattr(val, "shape", None) == () and hasattr(val, "item") else None
    if isinstance(val, (float, int, str, bool)):
        return val
    if hasattr(val, "item") and getattr(val, "size", 1) == 1:
        return val.item()
    if isinstance(val, dict):
        out = {}
        for k, v in val.items():
            s = sanitize_metric_value(v)
            if s is not None:
                out[str(k)] = s
        return out
    if isinstance(val, (list, tuple)):
        return [s for s in (sanitize_metric_value(x) for x in val) if s is not None]
    return None
