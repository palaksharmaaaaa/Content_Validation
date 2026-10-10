"""
core.retrain_engine: Modality-agnostic fine-tune-validate-promote-or-rollback loop.

Each <modality>_detector.retrain supplies a `train_fn` that does the actual (torch) training and
writes a candidate checkpoint; this module owns everything else: choosing samples from the
in-place MediaLibrary, comparing against the last promoted accuracy in CHECKPOINT_LOG.md, and
either promoting the candidate or discarding it. core/ stays torch-free.
"""
from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path
from typing import Callable, List, Tuple

from core.checkpoint_log import (
    RetrainResult,
    append_row,
    next_version,
    read_cumulative_samples,
    read_last_accuracy,
)
from core.media_library import MediaLibrary

logger = logging.getLogger("core.retrain_engine")

Sample = Tuple[str, Path, str]  # (sha256, path, "ai_generated" | "real")
# train_fn(train_samples, val_samples, candidate_path) -> (train_loss, val_accuracy)
TrainFn = Callable[[List[Sample], List[Sample], Path], Tuple[float, float]]


def run_retrain(
    library: MediaLibrary,
    log_path: Path,
    live_checkpoint: Path,
    train_fn: TrainFn,
    min_new: int = 1,
    replay_ratio: int = 4,
    min_replay: int = 50,
) -> RetrainResult:
    """Fine-tune on newly labelled media plus a replay sample, then promote the candidate only if its validation accuracy is not worse than the last promoted one; otherwise discard it and leave the live checkpoint untouched."""
    log_path, live_checkpoint = Path(log_path), Path(live_checkpoint)
    old_acc = read_last_accuracy(log_path)
    cumulative = read_cumulative_samples(log_path)

    new_samples = library.samples(status="new")
    if len(new_samples) < max(1, min_new):
        return RetrainResult(False, old_acc or 0.0, 0, cumulative,
                             f"Only {len(new_samples)} new sample(s); need {min_new}.", old_acc)

    train, val = library.select_for_retrain(replay_ratio=replay_ratio, min_replay=min_replay)
    if not train or not val:
        return RetrainResult(False, old_acc or 0.0, 0, cumulative,
                             f"Need both training and validation samples (have {len(train)} / {len(val)}); "
                             "add more labeled media.", old_acc)

    candidate = live_checkpoint.with_name(live_checkpoint.stem + ".candidate" + live_checkpoint.suffix)
    try:
        train_loss, val_acc = train_fn(train, val, candidate)
    except Exception:
        candidate.unlink(missing_ok=True)
        raise

    if not (isinstance(val_acc, (int, float)) and math.isfinite(val_acc) and 0.0 <= val_acc <= 1.0):
        candidate.unlink(missing_ok=True)
        return RetrainResult(False, 0.0, 0, cumulative,
                             f"Rolled back: the candidate's validation accuracy ({val_acc!r}) is not a number between 0 and 1. Labels stay queued.", old_acc)

    if old_acc is not None and val_acc < old_acc:
        candidate.unlink(missing_ok=True)
        logger.info("Retrain rolled back: val %.4f < previous %.4f", val_acc, old_acc)
        return RetrainResult(False, val_acc, 0, cumulative,
                             f"Rolled back: validation accuracy {val_acc:.1%} is below the current "
                             f"checkpoint's {old_acc:.1%}. Labels stay queued.", old_acc)

    candidate.replace(live_checkpoint)
    folded = len(new_samples)
    library.mark_seen(sha for sha, _, _ in new_samples)
    cumulative += folded
    append_row(log_path, next_version(log_path), date.today().isoformat(), train_loss, val_acc, folded, cumulative)
    return RetrainResult(True, val_acc, folded, cumulative,
                         f"Promoted: validation accuracy {val_acc:.1%} on {len(val)} held-out files.", old_acc)
