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
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from core.checkpoint_log import (
    RetrainResult,
    append_row,
    next_version,
    read_cumulative_samples,
    read_last_accuracy,
)
from core.media_library import MediaLibrary

logger = logging.getLogger("core.retrain_engine")

Sample = tuple[str, Path, str]  # (sha256, path, "ai_generated" | "real")
# train_fn(train_samples, val_samples, candidate_path) -> (train_loss, val_accuracy)
TrainFn = Callable[[list[Sample], list[Sample], Path], tuple[float, float]]


MIN_DRIFT_SAMPLES = 10    # a two-sample statistical test says nothing; below this the report is skipped


def generate_drift_report(
    reference_samples: list[Sample],
    current_samples: list[Sample],
    report_output_path: Path | None = None,
) -> dict[str, Any]:
    """Produces an Evidently AI dataset drift report comparing reference vs. current samples.

    The only feature compared is file size: it flags a batch of very differently sized files, not a change in what the pictures show.
    Evidently is an optional dependency (requirements-dev.txt); without it the report is skipped with a message, never an error.

    Saves an interactive HTML report if report_output_path is provided.
    """
    if min(len(reference_samples), len(current_samples)) < MIN_DRIFT_SAMPLES:
        return {
            "drift_detected": False,
            "drift_share": 0.0,
            "drifted_features": 0,
            "report_path": None,
            "message": f"Too few samples for a drift report (need at least {MIN_DRIFT_SAMPLES} on each side; "
                       f"have {len(reference_samples)} and {len(current_samples)}).",
        }

    try:
        import pandas as pd
        from evidently import Report
        from evidently.presets import DataDriftPreset

        def _to_df(samples: list[Sample]) -> pd.DataFrame:
            rows = []
            for _, path, _label in samples:
                p = Path(path)
                size_kb = (p.stat().st_size / 1024.0) if p.is_file() else 0.0
                rows.append({"file_size_kb": size_kb})          # the class label is not a feature: a different AI/real mix is not drift
            return pd.DataFrame(rows)

        ref_df = _to_df(reference_samples)
        curr_df = _to_df(current_samples)

        report = Report([DataDriftPreset()])
        snapshot = report.run(reference_data=ref_df, current_data=curr_df)

        report_str = None
        if report_output_path:
            p_out = Path(report_output_path)
            p_out.parent.mkdir(parents=True, exist_ok=True)
            snapshot.save_html(str(p_out))
            report_str = str(p_out)

        snap_dict = snapshot.dict()
        metrics = snap_dict.get("metrics", [])
        drift_share = 0.0
        drift_count = 0
        for m in metrics:
            val = m.get("value")
            if isinstance(val, dict) and "share" in val:
                drift_share = float(val.get("share", 0.0))
                drift_count = int(val.get("count", 0))
                break

        drift_detected = drift_share > 0.5
        return {
            "drift_detected": drift_detected,
            "drift_share": round(drift_share, 4),
            "drifted_features": drift_count,
            "report_path": report_str,
            "message": f"Drift analysis complete. Drifted features: {drift_count} (share: {drift_share:.1%}).",
        }
    except Exception as exc:
        logger.warning("Evidently drift report generation encountered an error: %s", exc)
        return {
            "drift_detected": False,
            "drift_share": 0.0,
            "drifted_features": 0,
            "report_path": None,
            "message": f"Evidently drift analysis failed: {exc}",
        }


def run_retrain(
    library: MediaLibrary,
    log_path: Path,
    live_checkpoint: Path,
    train_fn: TrainFn,
    min_new: int = 1,
    replay_ratio: int = 4,
    min_replay: int = 50,
    drift_report_path: Path | None = None,
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

    # Automated Evidently AI drift analysis prior to promotion
    drift_info = generate_drift_report(val, train, report_output_path=drift_report_path)
    drift_msg = ""
    if drift_info.get("drift_detected"):
        logger.warning("Feature drift flagged prior to model promotion: %s", drift_info.get("message"))
        drift_msg = f" [Drift warning: {drift_info['drifted_features']} feature(s) drifted ({drift_info['drift_share']:.1%})]"
    elif drift_info.get("report_path"):
        drift_msg = f" [Evidently drift report saved to {drift_info['report_path']}]"

    candidate.replace(live_checkpoint)
    folded = len(new_samples)
    library.mark_seen(sha for sha, _, _ in new_samples)
    cumulative += folded
    append_row(log_path, next_version(log_path), date.today().isoformat(), train_loss, val_acc, folded, cumulative)
    return RetrainResult(True, val_acc, folded, cumulative,
                         f"Promoted: validation accuracy {val_acc:.1%} on {len(val)} held-out files.{drift_msg}", old_acc)
