"""
core.calibration_report: honest accuracy / calibration measurement from labeled (label, P(AI)%) pairs.

Pure maths, no detector imports. Without labeled data the engine's percentages are uncalibrated
heuristics; this module is how that claim is replaced by measured numbers (see services/calibration_cli.py).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.bands import classify_band_from_percent

MIN_RELIABLE_SAMPLES = 30


_AI_LABELS = {"ai", "ai_generated", "ai-generated", "synthetic", "fake", "generated"}
_REAL_LABELS = {"real", "authentic", "genuine", "human"}


def _label_class(label: str) -> Optional[int]:
    """1 for an AI label, 0 for a real label, None for anything else: an unrecognised label is never counted as real."""
    name = str(label).strip().lower()
    return 1 if name in _AI_LABELS else 0 if name in _REAL_LABELS else None


def evaluate(pairs: Iterable[Tuple[str, float]], threshold: float = 50.0, bins: int = 10) -> Dict[str, Any]:
    """``pairs``: (ground-truth label, predicted P(AI) in percent). Returns accuracy, ECE, Brier, band occupancy."""
    pairs = list(pairs)
    classes = [(_label_class(lab), p) for lab, p in pairs]
    unlabeled = sum(1 for c, _ in classes if c is None)
    data = [(c, max(0.0, min(100.0, float(p))) / 100.0) for c, p in classes if c is not None and p is not None and float(p) == float(p)]
    n = len(data)
    n_ai = sum(y for y, _ in data)
    warnings: List[str] = []
    if len(pairs) - n:
        warnings.append(f"{len(pairs) - n} sample(s) were left out: an unrecognised label ({unlabeled}) or a missing score.")
    if n == 0:
        return {"n": 0, "warnings": ["No labeled samples: nothing can be measured."]}
    if n < MIN_RELIABLE_SAMPLES:
        warnings.append(f"Only {n} samples (< {MIN_RELIABLE_SAMPLES}): all figures below are statistically unreliable.")
    if n_ai in (0, n):
        warnings.append("Only one class present: precision/recall and calibration are not meaningful.")

    t = threshold / 100.0
    tp = sum(1 for y, p in data if y == 1 and p >= t)
    fp = sum(1 for y, p in data if y == 0 and p >= t)
    fn = sum(1 for y, p in data if y == 1 and p < t)
    tn = n - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None

    table: List[Dict[str, Any]] = []
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [(y, p) for y, p in data if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not members:
            continue
        conf = sum(p for _, p in members) / len(members)
        freq = sum(y for y, _ in members) / len(members)
        ece += len(members) / n * abs(conf - freq)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(members), "mean_predicted": round(conf, 4), "observed_ai_rate": round(freq, 4)})

    occupancy: Dict[str, Counter] = {"ai": Counter(), "real": Counter()}
    for y, p in data:
        occupancy["ai" if y else "real"][classify_band_from_percent(p * 100.0).value] += 1

    return {
        "n": n, "n_ai": n_ai, "n_real": n - n_ai, "threshold_percent": threshold,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": round((tp + tn) / n, 4),
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        "brier": round(sum((p - y) ** 2 for y, p in data) / n, 4),
        "ece": round(ece, 4),
        "reliability": table,
        "band_occupancy": {k: dict(v) for k, v in occupancy.items()},
        "warnings": warnings,
    }
