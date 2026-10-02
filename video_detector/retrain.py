"""
video_detector.retrain: Fine-tune the video checkpoint on files registered in the in-place media library.

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
from video_detector.config import CHECKPOINT_LOG_FILE, DEFAULT_VIDEO_CHECKPOINT
from video_detector.trainer import VideoDetectorTrainer

logger = logging.getLogger("video_detector.retrain")

_LABEL_TO_INT = {"ai_generated": 0, "real": 1}


def _pairs(samples: List[Sample]) -> List[Tuple[Path, int]]:
    return [(path, _LABEL_TO_INT[label]) for _, path, label in samples]


def _load_live(trainer: VideoDetectorTrainer, live: Path) -> None:
    """Continues from the promoted checkpoint (fine-tune) when one exists; otherwise bootstraps."""
    if live.is_file():
        state = torch.load(live, map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(state["model_state_dict"])


def count_pending_corrections(library: Optional[MediaLibrary] = None) -> int:
    """Labeled files registered since the last promoted checkpoint."""
    return (library or library_for("video")).counts()["new"]


def run_retrain(
    min_new: int = 1,
    epochs: int = 5,
    library: Optional[MediaLibrary] = None,
    log_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
) -> RetrainResult:
    live = Path(checkpoint_path) if checkpoint_path else DEFAULT_VIDEO_CHECKPOINT

    def train_fn(train: List[Sample], val: List[Sample], candidate: Path) -> Tuple[float, float]:
        trainer = VideoDetectorTrainer(checkpoint_path=candidate)
        _load_live(trainer, live)
        pairs = trainer.pairs_from_samples(_pairs(train))
        val_pairs = trainer.pairs_from_samples(_pairs(val))
        if not pairs or not val_pairs:
            raise RuntimeError("No decodable video in the selected training/validation samples.")
        history = trainer.train(pairs, epochs=epochs, val_pairs=val_pairs)  # saves to `candidate`
        return history["loss"][-1], history["val_accuracy"][-1]

    return _run_retrain(
        library or library_for("video"),
        Path(log_path) if log_path else CHECKPOINT_LOG_FILE,
        live,
        train_fn,
        min_new=min_new,
    )
