"""core.model_registry: which model and algorithm does each step use, where did it come from, and what was it trained on?

One honest answer for the UI, the JSON report and the documentation. Two kinds of fact live here:

* fixed facts about a model (its name, variant, licence, source), written once below and checked against the upstream model card or
  licence file on the date given in ``VERIFIED_ON``; and
* facts about *this installation* (how many files the bundled face model was trained on, whether a trained backbone is present, how many
  corrections have been recorded), which are read from the files at run time and never typed in.

A number that is not known is reported as "not stated" and never replaced by an estimate.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VERIFIED_ON = "2026-10-10"
ROOT = Path(__file__).resolve().parent.parent

# Origins, from least to most project-specific
PRETRAINED = "pretrained, used unchanged"
FINE_TUNED = "pretrained, then fine-tuned by this project"
TRAINED = "trained by this project"
HAND_WRITTEN = "hand-written signal processing, no learned model"
NOT_STATED = "not stated by the publisher"


@dataclass(frozen=True)
class ModelInfo:
    """One step of the analysis and the model or algorithm behind it."""
    step: str                      # what the step does, in plain words
    name: str                      # model or algorithm
    variant: str                   # exact checkpoint / version / size
    origin: str                    # one of the origin constants above
    training_data: str             # what it was trained on
    training_size: str             # how many files / images, or "not stated"
    licence: str                   # licence of the weights or code, with any caveat
    source: str                    # where it comes from
    used_by: tuple = ("image", "video", "audio")
    active: bool = True            # False when the weights are not present on this machine
    note: str = ""                 # anything the reader should know (limits, caveats)
    trained_files: Optional[int] = None   # distinct files this project trained it on (counted in the total); None if unknown or not trained here

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["used_by"] = list(self.used_by)
        return d


# --------------------------------------------------------------------------------------------- fixed facts
def _upstream() -> List[ModelInfo]:
    from core.perception.age import MODEL_ID as age_id, MODEL_REVISION as age_rev
    from core.perception.detector import MODEL_ID as det_id, MODEL_REVISION as det_rev
    from core.perception.recognizer import MODEL_ID as rec_id, MODEL_REVISION as rec_rev

    return [
        ModelInfo("Find faces", "YuNet", "face_detection_yunet_2023mar.onnx (OpenCV Zoo)", PRETRAINED,
                  "not stated in the model files shipped here", "not stated", "MIT (Copyright 2020 Shiqi Yu)",
                  "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet", used_by=("image", "video")),
        ModelInfo("Match the same person across faces", "SFace", "face_recognition_sface_2021dec.onnx (OpenCV Zoo)", PRETRAINED,
                  "not stated in the model files shipped here", "not stated", "Apache-2.0",
                  "https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface", used_by=("image",)),
        ModelInfo("Read facial expression", "MobileFaceNet + Progressive Teacher", "facial_expression_recognition_mobilefacenet_2022july.onnx (OpenCV Zoo)",
                  PRETRAINED, "not stated on the OpenCV Zoo page (RAF-DB is named only as the benchmark it was evaluated on)", "not stated",
                  "Apache-2.0 (the OpenCV Zoo directory states it covers all files in it; the licences of the upstream papers' code are not stated there)",
                  "https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition", used_by=("image",)),
        ModelInfo("Estimate apparent age (minor screening)", "MiVOLO v2", f"{age_id} @ {age_rev[:7]}, 28.8 M parameters, face + body crops", PRETRAINED,
                  "'proprietary and open-source datasets' (the author names none)", "not stated", "Apache-2.0 (weights and vendored network code)",
                  f"https://huggingface.co/{age_id}", used_by=("image", "video"),
                  note="Apparent age from appearance, not real age. Recall on minors is not 100 % (see LIMITATIONS)."),
        ModelInfo("Find people, animals, vehicles and objects", "RF-DETR Small", f"{det_id} @ {det_rev[:7]}, about 32 M parameters", PRETRAINED,
                  "COCO 2017 object detection (80 classes)", "not stated on the model card", "Apache-2.0",
                  f"https://huggingface.co/{det_id}", used_by=("image", "video")),
        ModelInfo("Name the scene, species, vehicle type, genre and age-group opinion", "SigLIP 2 Base (zero-shot image-text matching)",
                  f"{rec_id} @ {rec_rev[:7]}", PRETRAINED, "WebLI image-text pairs", "not stated on the model card", "Apache-2.0",
                  f"https://huggingface.co/{rec_id}", used_by=("image", "video"),
                  note="Always returns the best match inside a fixed vocabulary of 182 prompts; results under 30 % are shown as 'not sure'."),
    ]


_FACE_DATASET = "Kaggle 'Human Faces Dataset' (kaustubhdhote): real photographs and AI-generated faces"
_FACE_DATASET_LICENCE = ("The Kaggle page could not be read when this was checked and public mirrors disagree (CC BY-SA 4.0 versus 'Other'). "
                         "Confirm it on Kaggle before any commercial use of the trained weights.")


# --------------------------------------------------------------------------------------------- run-time facts
def _read_checkpoint_meta(path: Path) -> Optional[Dict[str, Any]]:
    """The ``meta`` saved in a checkpoint, ``{}`` for one saved without it, None if there is no usable checkpoint."""
    if not path.is_file() or path.stat().st_size < 1024:
        return None
    try:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        meta = state.get("meta") if isinstance(state, dict) else None
        return dict(meta) if isinstance(meta, dict) else {}
    except Exception as exc:
        logger.warning("Could not read checkpoint metadata from %s: %s", path.name, exc)
        return None


def _count(meta: Dict[str, Any], key: str) -> Optional[int]:
    value = meta.get(key)
    return int(value) if isinstance(value, (int, float)) and value >= 0 else None


def _trained_text(meta: Dict[str, Any]) -> tuple:
    """(description, size text, total files) for a checkpoint trained here."""
    train, val = _count(meta, "train_samples"), _count(meta, "val_samples")
    unit = meta.get("unit", "samples")
    if train is None:
        return "not recorded in the checkpoint (trained before sizes were saved)", "not recorded", None
    total = train + (val or 0)
    return f"{total:,} {unit}: {train:,} to train, {val or 0:,} held out to validate", f"{total:,} {unit}", total


def _face_model() -> ModelInfo:
    from image_detector.face_authenticity import CHECKPOINT

    meta = _read_checkpoint_meta(CHECKPOINT)
    base = dict(step="Judge whether each face is a real photograph or AI-generated", name="ResNet-18 two-class head",
                variant="torchvision ResNet-18, 112 px face crops, dropout 0.3 + linear(2)", used_by=("image",),
                licence="Code: Apache-2.0 (this project). Starting weights: torchvision ResNet-18 trained on ImageNet-1K, whose terms limit use to "
                        "non-commercial research and education; the face dataset's licence is unconfirmed (" + _FACE_DATASET_LICENCE + ")",
                source="image_detector/face_training.py")
    if meta is None:
        return ModelInfo(**base, origin=FINE_TUNED, training_data=_FACE_DATASET, training_size="no trained checkpoint on this machine", active=False)
    text, size, files = _trained_text(meta)
    dataset = meta.get("dataset") or _FACE_DATASET
    note = ""
    acc = meta.get("val_clean", {})
    if isinstance(acc, dict) and "accuracy" in acc:
        note = (f"Accuracy on its own held-out images: {acc['accuracy'] * 100:.1f} % (images from the same dataset it learned from; "
                "it says little about other generators or other photographs).")
    return ModelInfo(**base, origin=FINE_TUNED, training_data=f"{dataset}; {text}", training_size=size, note=note, trained_files=files)


def _backbone(modality: str) -> ModelInfo:
    import importlib

    cfg = importlib.import_module(f"{modality}_detector.config")
    path = getattr(cfg, "DEFAULT_CHECKPOINT", None) or getattr(cfg, "DEFAULT_AUDIO_CHECKPOINT", None) or getattr(cfg, "DEFAULT_VIDEO_CHECKPOINT", None)
    arch = {"image": "ResNet-18 / ResNet-50 / MobileNetV3-Small (chosen when training)",
            "video": "ResNet-18 judging pairs of consecutive frames",
            "audio": "small multi-layer perceptron over 5 acoustic features"}[modality]
    init = "torchvision ImageNet-1K weights" if modality != "audio" else "random weights"
    base = dict(step=f"Learned {modality} AI-versus-real classifier (optional)", name=arch, variant=f"{Path(str(path)).name}", used_by=(modality,),
                licence="Apache-2.0 (this project); " + ("starting weights are torchvision ImageNet-1K (ImageNet terms: non-commercial research)." if modality != "audio" else "no third-party weights."),
                source=f"{modality}_detector/trainer.py")
    meta = _read_checkpoint_meta(Path(path)) if path else None
    if meta is None:
        return ModelInfo(**base, origin=TRAINED if modality == "audio" else FINE_TUNED, training_data="none: no trained checkpoint is shipped",
                         training_size="0 files", active=False,
                         note="Not installed. Until you train one, the AI-versus-real score comes from the hand-written signal checks alone.")
    text, size, files = _trained_text(meta)
    return ModelInfo(**base, origin=TRAINED if modality == "audio" else FINE_TUNED, training_data=text, training_size=size, trained_files=files)


def _heuristics(modality: str) -> ModelInfo:
    steps = {
        "image": ("Pixel, spectral, noise, compression and metadata checks (16 checks)",
                  "median-filter noise residual, bilateral smoothness, 2-D Fourier slope, error-level analysis, JPEG quantisation tables, EXIF/XMP/C2PA marker scan"),
        "video": ("Container, frame and motion checks",
                  "per-frame noise and smoothness (core/frame_scorer.py), frame-to-frame motion and flicker, container atoms, C2PA marker scan"),
        "audio": ("Container, spectral and time-window checks",
                  "spectral flatness and cut-off, vocoder-band artefacts, silence structure, level statistics, per-window timeline, container tags"),
    }[modality]
    return ModelInfo(steps[0], "Statistical signal checks", "hand-set thresholds over: " + steps[1], HAND_WRITTEN,
                     "none: thresholds were set by hand on synthetic test files, not fitted to a labelled corpus", "0 files",
                     "Apache-2.0 (this project)", f"{modality}_detector/", used_by=(modality,),
                     note="These decide the verdict. Their accuracy on real-world AI media has not been measured.")


def _ood(modality: str) -> ModelInfo:
    import importlib

    from core.forensics.ood import OODGate

    path = ROOT / f"{modality}_detector" / "data" / "ood_stats.npz"
    gate = OODGate.load(path)
    if gate.calibrated:
        n = f"{gate.n_samples:,} files" if gate.n_samples else "fitted on an unrecorded number of files"
        return ModelInfo("Warn when a file is unlike the media the system was calibrated on", "Mahalanobis distance gate", "99th-percentile threshold", TRAINED,
                         "files you registered in the media library", n, "Apache-2.0 (this project)", "core/forensics/ood.py", used_by=(modality,))
    return ModelInfo("Warn when a file is unlike the media the system was calibrated on", "Mahalanobis distance gate", "not fitted", TRAINED,
                     "none", "0 files", "Apache-2.0 (this project)", "core/forensics/ood.py", used_by=(modality,), active=False,
                     note="Reports NOT_CALIBRATED until you fit it on at least as many files as it has features.")


def _calibration(modality: str) -> Optional[Dict[str, Any]]:
    from core.atomic_io import atomic_read_json

    data = atomic_read_json(ROOT / f"{modality}_detector" / "data" / f"{modality}_calibration.json", default=None)
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------------------------- public
def manifest(modality: str) -> Dict[str, Any]:
    """Every step used for ``modality`` (image, video or audio) and what is known about its model and training data."""
    if modality not in ("image", "video", "audio"):
        raise ValueError(f"modality must be image, video or audio, not {modality!r}")
    models: List[ModelInfo] = [_heuristics(modality)]
    if modality == "image":
        models.append(_face_model())
    models.append(_backbone(modality))
    models.append(_ood(modality))
    models += [m for m in _upstream() if modality in m.used_by]

    trained_here: List[Dict[str, Any]] = []
    total = 0
    for m in models:
        if m.active and m.trained_files is not None:
            total += m.trained_files
            trained_here.append({"model": m.name, "files": m.trained_files, "described_as": m.training_size})
    calibration = _calibration(modality) or {}
    corrections = int(calibration.get("samples_processed", 0) or 0)
    sentence = (f"For {modality} analysis this project trained " + "; ".join(f"the {t['model']} on {t['described_as']}" for t in trained_here)
                if trained_here else f"No model was trained by this project for {modality} analysis (0 files)")
    sentence += f"; {corrections:,} reviewer correction(s) have adjusted its thresholds. Pretrained models are listed with their upstream training data."
    return {"modality": modality, "verified_on": VERIFIED_ON, "training_files_total": total, "trained_by_this_project": trained_here,
            "feedback_corrections": corrections, "summary": sentence, "models": [m.to_dict() for m in models]}
