"""
services.system_status: a plain-language snapshot of what is configured on this machine.

The engine ships blank: every modality starts in heuristic mode with default priors. This reports, per component,
whether it is active, so the UI (and a user debugging a surprising result) can see at a glance what the verdict is
based on. Read-only: nothing here writes files. It loads only the small bundled models (face detector, expression, identity,
face authenticity) and never the large downloaded ones.
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
    """One line of the sidebar status panel: component, level (ok / info / warn), state and a hint."""
    component: str
    level: str      # "ok" | "info" | "warn"
    state: str      # short value, e.g. "Heuristic mode"
    hint: str       # what to do / what it means

    def to_dict(self) -> Dict[str, str]:
        """Plain-dict form."""
        return asdict(self)


def _learned_samples(calibration_file: Path) -> int:
    data = atomic_read_json(calibration_file, default=None)
    return int(data.get("samples_processed", 0)) if isinstance(data, dict) else 0


def _hardblock_entries(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip() and not line.startswith("#"))


def _age_row() -> StatusRow:
    """Whether minor screening can run: the weights file is present (not a Git LFS pointer) and timm is installed. Nothing is loaded."""
    import importlib.util

    from core.perception.age import WEIGHTS_PATH

    if not WEIGHTS_PATH.is_file():
        return StatusRow("Minor screening (age model)", WARN, "Weights missing", f"Expected at {WEIGHTS_PATH.relative_to(WEIGHTS_PATH.parents[3])}.")
    if WEIGHTS_PATH.stat().st_size < 1024:
        return StatusRow("Minor screening (age model)", WARN, "Git LFS pointer", "Run `git lfs install` then `git lfs pull` in the repository.")
    if importlib.util.find_spec("timm") is None:
        return StatusRow("Minor screening (age model)", WARN, "timm missing", "pip install -r requirements.txt")
    return StatusRow("Minor screening (age model)", OK, "Ready", "MiVOLO v2 estimates apparent age; uncertain people are sent to review.")


def collect_status() -> List[StatusRow]:
    """Inspect the installation (models, calibration, OOD fit, hard-block list, ffmpeg, compute device) and return the status rows."""
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

    from core.face_detection import get_face_finder

    faces_ok = get_face_finder().available
    rows.append(StatusRow("Face detector", OK if faces_ok else WARN, "YuNet loaded" if faces_ok else "Unavailable",
                          "Counts faces with a trained network." if faces_ok else "Model file core/models/face_detection_yunet_2023mar.onnx is missing; faces will not be counted."))
    from core.perception.detector import MODEL_ID as DETECTOR_ID, MODEL_REVISION as DETECTOR_REV
    from core.perception.face_attributes import get_face_attributes
    from core.perception.recognizer import MODEL_ID as RECOGNIZER_ID, MODEL_REVISION as RECOGNIZER_REV
    from services.fetch_models import present

    # The loaders insist on the pinned revision, so "present" must mean that revision, not just any cached copy of the repository.
    for label, repo, revision, what in (
            ("Object detector", DETECTOR_ID, DETECTOR_REV, "RF-DETR Small: people, animals, vehicles, objects"),
            ("Scene and species recognizer", RECOGNIZER_ID, RECOGNIZER_REV, "SigLIP 2 Base: place, animal species, vehicle type, kind of photo")):
        have = present(repo, revision)
        rows.append(StatusRow(label, OK if have else WARN, "Ready" if have else "Not downloaded",
                              what if have else "Run: python -m services.fetch_models"))
    attrs = get_face_attributes()
    rows.append(StatusRow("Expression and same-person models", OK if attrs.expression_available and attrs.identity_available else WARN,
                          "Ready" if attrs.expression_available and attrs.identity_available else "Missing", "Bundled ONNX files in core/models."))
    from image_detector.face_authenticity import get_face_authenticity

    fa = get_face_authenticity()
    if fa.available:
        acc = (fa.meta.get("val_stress") or {}).get("accuracy")
        rows.append(StatusRow("Face authenticity model", OK, "Trained" + (f" ({acc * 100:.1f}% on its own held-out set)" if acc else ""),
                              "Judges whether each face is real or AI-generated. Trained on one public dataset."))
    else:
        rows.append(StatusRow("Face authenticity model", INFO, "Not trained",
                              "python -m image_detector.face_training --real <folder> --ai <folder>"))
    rows.append(_age_row())
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
