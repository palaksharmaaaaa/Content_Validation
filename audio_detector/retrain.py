"""
audio_detector.retrain: Fine-tune the audio checkpoint on files registered in the in-place media library.

Nothing is copied: training reads each file from wherever it already lives (see core.media_library).
The candidate checkpoint is promoted only if its accuracy on the content-hash validation partition
is at least the last promoted checkpoint's; otherwise it is discarded and the labels stay queued.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from core.checkpoint_log import RetrainResult
from core.media_library import MediaLibrary, library_for
from core.retrain_engine import Sample, run_retrain as _run_retrain
from audio_detector.config import CHECKPOINT_LOG_FILE, DEFAULT_AUDIO_CHECKPOINT
from audio_detector.trainer import AudioDetectorTrainer

logger = logging.getLogger("audio_detector.retrain")

_LABEL_TO_INT = {"ai_generated": 0, "real": 1}


def _pairs(samples: List[Sample]) -> List[Tuple[Path, int]]:
    return [(path, _LABEL_TO_INT[label]) for _, path, label in samples]


def _load_live(trainer: AudioDetectorTrainer, live: Path) -> None:
    """Continues from the promoted checkpoint (fine-tune) when one exists; otherwise bootstraps."""
    if live.is_file():
        state = torch.load(live, map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(state["model_state_dict"])


def count_pending_corrections(library: Optional[MediaLibrary] = None) -> int:
    """Labeled files registered since the last promoted checkpoint."""
    return (library or library_for("audio")).counts()["new"]


def run_retrain(
    min_new: int = 1,
    epochs: int = 5,
    library: Optional[MediaLibrary] = None,
    log_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
) -> RetrainResult:
    live = Path(checkpoint_path) if checkpoint_path else DEFAULT_AUDIO_CHECKPOINT

    def train_fn(train: List[Sample], val: List[Sample], candidate: Path) -> Tuple[float, float]:
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
