"""
services.system_status: a plain-language snapshot of what is configured on this machine.

The engine ships blank: every modality starts in heuristic mode with default priors. This reports, per component,
whether it is active, so the UI (and a user debugging a surprising result) can see at a glance what the verdict is
based on. Read-only: nothing here writes files or loads models.
"""
from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from core.atomic_io import atomic_read_json
from core.forensics.config import hardblock_file

OK, INFO, WARN = "ok", "info", "warn"


@dataclass(frozen=True)
class StatusRow:
    component: str
    level: str      # "ok" | "info" | "warn"
    state: str      # short value, e.g. "Heuristic mode"
    hint: str       # what to do / what it means

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _learned_samples(calibration_file: Path) -> int:
    data = atomic_read_json(calibration_file, default=None)
    return int(data.get("samples_processed", 0)) if isinstance(data, dict) else 0


def _hardblock_entries(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip() and not line.startswith("#"))


def collect_status() -> List[StatusRow]:
    import audio_detector.config as audio_cfg
    import image_detector.config as image_cfg
    import video_detector.config as video_cfg

    rows: List[StatusRow] = []
    for label, cfg, checkpoint, ood_stats in (
        ("Image", image_cfg, image_cfg.DEFAULT_CHECKPOINT, Path(image_cfg.__file__).parent / "data" / "ood_stats.npz"),
        ("Video", video_cfg, video_cfg.DEFAULT_VIDEO_CHECKPOINT, Path(video_cfg.__file__).parent / "data" / "ood_stats.npz"),
        ("Audio", audio_cfg, audio_cfg.DEFAULT_AUDIO_CHECKPOINT, Path(audio_cfg.__file__).parent / "data" / "ood_stats.npz"),
    ):
        if checkpoint.is_file():
            rows.append(StatusRow(f"{label} model", OK, "Trained checkpoint loaded", "A neural score is blended with the heuristics."))
        else:
            rows.append(StatusRow(f"{label} model", INFO, "Heuristic mode", "No checkpoint yet. Register labelled files and retrain on the Learning tab."))
        n = _learned_samples(cfg.CALIBRATION_FILE)
        rows.append(StatusRow(f"{label} calibration", OK if n else INFO, f"{n} feedback sample(s)" if n else "Default priors",
                              "Thresholds adapt only when you submit feedback."))
        rows.append(StatusRow(f"{label} out-of-distribution check", OK if ood_stats.is_file() else INFO,
                              "Fitted" if ood_stats.is_file() else "Not fitted",
                              "Fit it from your library: python -m %s_detector.dimension_checks.fit_ood" % label.lower()))

    hb = _hardblock_entries(hardblock_file())
    rows.append(StatusRow("Hard-block list", OK if hb else INFO, f"{hb} hash(es)" if hb else "Empty",
                          "Optional: list SHA-256 hashes (one per line) in core/data/hardblock_sha256.txt."))
    ffmpeg = shutil.which("ffmpeg")
    rows.append(StatusRow("ffmpeg", OK if ffmpeg else WARN, "Found" if ffmpeg else "Missing",
                          "Needed to decode compressed audio/video." if not ffmpeg else "Used for audio/video decoding."))
    try:
        import torch

        gpu = torch.cuda.is_available()
    except Exception:  # torch missing or broken: report, do not crash the sidebar
        gpu = False
    rows.append(StatusRow("Compute", INFO, "GPU (CUDA)" if gpu else "CPU", "Analysis runs on CPU unless CUDA is available."))
    return rows
