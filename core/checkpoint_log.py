"""
core.checkpoint_log: Git-tracked, plain-text provenance log for trained checkpoints, plus the
shared RetrainResult returned by every <modality>_detector.retrain.run_retrain().

The log records only numbers (version, date, training loss, held-out validation accuracy, sample
counts) -- never a file name, path, or any media content -- so it is safe to commit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_HEADER = "| Version | Date | Train Loss | Val Accuracy | New Samples | Cumulative Samples |\n|---|---|---|---|---|---|\n"


@dataclass
class RetrainResult:
    """Outcome of one retrain run."""
    promoted: bool
    new_accuracy: float
    samples_folded_in: int
    cumulative_samples: int
    message: str
    old_accuracy: Optional[float] = None


def _rows(log_path: Path) -> List[List[str]]:
    if not log_path.is_file():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or "Version" in line:
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def read_last_accuracy(log_path: str | Path) -> Optional[float]:
    """Validation accuracy of the most recently promoted checkpoint, or None if none was ever promoted."""
    rows = _rows(Path(log_path))
    if not rows:
        return None
    try:
        return float(rows[-1][3])
    except (IndexError, ValueError):
        return None


def next_version(log_path: str | Path) -> int:
    """Next checkpoint version number according to the log (1 if the log does not exist)."""
    rows = _rows(Path(log_path))
    if not rows:
        return 1
    try:
        return int(rows[-1][0]) + 1
    except (IndexError, ValueError):
        return len(rows) + 1


def read_cumulative_samples(log_path: str | Path) -> int:
    """Total number of samples recorded by the last promoted checkpoint (0 if none)."""
    total = 0
    for row in _rows(Path(log_path)):
        try:
            total += int(row[4])
        except (IndexError, ValueError):
            continue
    return total


def append_row(
    log_path: str | Path,
    version: int,
    date: str,
    train_loss: float,
    val_acc: float,
    new_samples: int,
    cumulative_samples: int,
) -> None:
    """Append one promoted-checkpoint row (version, date, losses, sample counts) to the log."""
    from core.atomic_io import atomic_write_text, process_lock

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with process_lock(log_path):                                   # read, append and swap as one step, also against another process
        existing = log_path.read_text(encoding="utf-8") if log_path.is_file() else _HEADER
        atomic_write_text(log_path, existing + f"| {version} | {date} | {train_loss:.4f} | {val_acc:.4f} | {new_samples} | {cumulative_samples} |\n")
