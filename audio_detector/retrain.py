"""
audio_detector.retrain: Fine-tune the audio checkpoint on files registered in the in-place media library.

Nothing is copied: training reads each file from wherever it already lives (see core.media_library).
The candidate checkpoint is promoted only if its accuracy on the content-hash validation partition
is at least the last promoted checkpoint's; otherwise it is discarded and the labels stay queued.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from audio_detector.config import CHECKPOINT_LOG_FILE, DEFAULT_AUDIO_CHECKPOINT
from audio_detector.trainer import AudioDetectorTrainer
from core.checkpoint_log import RetrainResult
from core.media_library import MediaLibrary, library_for
from core.retrain_engine import Sample
from core.retrain_engine import run_retrain as _run_retrain

logger = logging.getLogger("audio_detector.retrain")

_LABEL_TO_INT = {"ai_generated": 0, "real": 1}


def _pairs(samples: list[Sample]) -> list[tuple[Path, int]]:
    return [(path, _LABEL_TO_INT[label]) for _, path, label in samples]


def _load_live(trainer: AudioDetectorTrainer, live: Path) -> None:
    """Continues from the promoted checkpoint (fine-tune) when one exists; otherwise bootstraps."""
    if live.is_file():
        state = torch.load(live, map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(state["model_state_dict"])


def count_pending_corrections(library: MediaLibrary | None = None) -> int:
    """Labeled files registered since the last promoted checkpoint."""
    return (library or library_for("audio")).counts()["new"]


def run_retrain(
    min_new: int = 1,
    epochs: int = 5,
    library: MediaLibrary | None = None,
    log_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> RetrainResult:
    live = Path(checkpoint_path) if checkpoint_path else DEFAULT_AUDIO_CHECKPOINT

    def train_fn(train: list[Sample], val: list[Sample], candidate: Path) -> tuple[float, float]:
        trainer = AudioDetectorTrainer(checkpoint_path=candidate)
        _load_live(trainer, live)
        X, y = trainer.arrays_from_samples(_pairs(train))
        Xv, yv = trainer.arrays_from_samples(_pairs(val))
        if len(X) == 0 or len(Xv) == 0:
            raise RuntimeError("No decodable audio in the selected training/validation samples.")
        history = trainer.train(X, y, epochs=epochs, X_val=Xv, y_val=yv)  # saves to `candidate`
        return history["loss"][-1], history["val_accuracy"][-1]

    return _run_retrain(
        library or library_for("audio"),
        Path(log_path) if log_path else CHECKPOINT_LOG_FILE,
        live,
        train_fn,
        min_new=min_new,
    )
