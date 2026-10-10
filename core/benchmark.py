"""core.benchmark: accuracy of a detector on a labelled folder (``real/`` and ``ai_generated/``), shared by all three modalities.

Honest by construction: a file the detector abstains on ("UNDECIDED") or fails on is counted and reported, never silently scored as
"real"; accuracy, precision and recall are over decided files, and ``coverage`` says how many that was. ROC-AUC is the rank statistic,
so tied scores count half and the number does not depend on file order.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from core.atomic_io import atomic_write_json

logger = logging.getLogger("core.benchmark")

AI_LABEL = "LIKELY AI-GENERATED"
REAL_LABEL = "LIKELY REAL"


def roc_auc(scores: List[float], truth: List[int]) -> Optional[float]:
    """P(a random AI file scores above a random real one), ties counting half; None when one class is missing."""
    pos = [s for s, t in zip(scores, truth) if t == 1]
    neg = [s for s, t in zip(scores, truth) if t == 0]
    if not pos or not neg:
        return None
    s = np.asarray(scores, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):                         # average rank within each run of equal scores
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    t = np.asarray(truth)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _as_mapping(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        return result
    to_dict = getattr(result, "to_dict", None)
    return to_dict() if callable(to_dict) else {}


def _ratio(num: float, den: float) -> Optional[float]:
    return round(num / den, 4) if den else None


def evaluate_folder(
    dataset_dir: Path | str,
    extensions: Iterable[str],
    predict: Callable[[Path], Any],
) -> Dict[str, Any]:
    """Run ``predict`` on every supported file under ``<dataset>/ai_generated`` and ``<dataset>/real`` and score the labels it returns."""
    root = Path(dataset_dir)
    exts = {e.lower() for e in extensions}
    labelled: List[Tuple[Path, int]] = []
    for sub, truth in (("ai_generated", 1), ("real", 0)):
        folder = root / sub
        if folder.is_dir():
            labelled += [(p, truth) for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    if not labelled:
        raise ValueError(f"No supported files found in {root / 'ai_generated'} or {root / 'real'}")

    tp = fp = tn = fn = abstained = failed = 0
    scores: List[float] = []
    score_truth: List[int] = []
    latencies: List[float] = []
    for path, truth in labelled:
        started = time.perf_counter()
        try:
            res = _as_mapping(predict(path))
        except Exception as exc:                                  # one unreadable file must not abort the run
            logger.warning("benchmark: %s failed (%s: %s)", path.name, type(exc).__name__, exc)
            failed += 1
            continue
        if res.get("error") or res.get("success") is False or res.get("content_valid") is False:
            failed += 1                                           # an unreadable file is a failure, not an abstention from deciding
            continue
        latencies.append(time.perf_counter() - started)
        label = res.get("label")
        pct = res.get("ai_percentage")
        if isinstance(pct, (int, float)):
            scores.append(float(pct) / 100.0)
            score_truth.append(truth)
        if label == AI_LABEL:
            tp, fp = tp + (truth == 1), fp + (truth == 0)
        elif label == REAL_LABEL:
            tn, fn = tn + (truth == 0), fn + (truth == 1)
        else:
            abstained += 1

    decided = tp + fp + tn + fn
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    auc = roc_auc(scores, score_truth)
    return {
        "dataset": root.name,
        "files": len(labelled),
        "scored": len(latencies),
        "decided": decided,
        "abstained": abstained,
        "failed": failed,
        "coverage": _ratio(decided, len(labelled)),
        "accuracy": _ratio(tp + tn, decided),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": None if auc is None else round(auc, 4),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "mean_latency_seconds": round(float(np.mean(latencies)), 3) if latencies else None,
        "p95_latency_seconds": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
    }


def write_report(report: Dict[str, Any], destination: Path | str) -> None:
    atomic_write_json(destination, report)
