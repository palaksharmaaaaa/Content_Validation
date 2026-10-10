"""
image_detector.learner: Heuristic feedback-driven calibration engine for image forensics.

IMPORTANT -- what this module does NOT do: despite the "self-improving" language used
elsewhere in this project, record_feedback() below never touches the neural network's
weights (models/ai_detector.pt) and never retrains or fine-tunes anything. It only nudges
a handful of scalar constants (feature_weights, sensitivity_offsets) stored in
data/image_calibration.json by small fixed deltas per feedback event, which are then read
by scoring.pool_bayesian_log_odds(). This is a legitimate technique -- rule-based
calibration drift correction -- but it is not machine learning and has no mechanism to
improve classification for taxonomy states the underlying CNN was never trained to
distinguish (see trainer.py / models/backbone.py: the CNN is a binary real-vs-ai_generated
classifier only).

Maintains a dedicated feedback memory bank and calibrates the Bayesian scoring weights used
by image_detector.scoring based on user-verification feedback.
Depends only on core (atomic JSON I/O, media library) and this package.
"""
from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from image_detector.config import DATA_DIR, DEFAULT_FEATURE_WEIGHTS
from image_detector.schemas import ImageFeedbackRecord
from core.metrics_util import sanitize_metric_value
from core.hashing import already_recorded
from core.atomic_io import atomic_read_json, atomic_write_json, serialized_on
from core.media_library import MediaLibrary, library_for, register_feedback

logger = logging.getLogger("image_detector.learner")


def _measurement(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class ImageSelfImprover:
    """
    Dedicated feedback-calibration module for Image AI Detection (adjusts scoring constants only -- see module docstring above; not model training).
    Maintains a persistent memory of verified authentic and AI-generated image fingerprints,
    and dynamically optimizes Bayesian likelihood ratio weights and sensitivity offsets.
    """

    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        memory_file: Optional[Path] = None,
        calibration_file: Optional[Path] = None,
        library: Optional[MediaLibrary] = None,
    ):
        self.memory_dir = memory_dir or DATA_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = Path(memory_file) if memory_file else (self.memory_dir / "image_feedback.json")
        self.calibration_file = Path(calibration_file) if calibration_file else (self.memory_dir / "image_calibration.json")
        # Verified files are queued for retraining by reference (no copy). When a test isolates
        # storage via memory_file, don't touch the real library unless one is passed in.
        self.library = library if library is not None else (None if memory_file else library_for("image"))

    def load_calibration(self) -> Dict[str, Any]:
        """Loads active calibration parameters or initializes default priors."""
        default_calib = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "samples_processed": 0,
            "feature_weights": dict(DEFAULT_FEATURE_WEIGHTS),
            "sensitivity_offsets": {
                "noise_center_offset": 0.0,
                "smooth_center_offset": 0.0,
                "fft_decay_offset": 0.0,
            },
        }
        loaded = atomic_read_json(self.calibration_file, default=None)
        return loaded if isinstance(loaded, dict) else default_calib

    def save_calibration(self, calib: Dict[str, Any]) -> None:
        """Saves updated calibration parameters atomically."""
        calib["last_updated"] = datetime.now().isoformat()
        try:
            atomic_write_json(self.calibration_file, calib, indent=2)
        except Exception as e:
            logger.error("Failed to save image calibration atomically: %s", e)

    def load_memory(self) -> List[Dict[str, Any]]:
        """Loads verified image memory bank."""
        loaded = atomic_read_json(self.feedback_file, default=[])
        return loaded if isinstance(loaded, list) else []

    @serialized_on("feedback_file", "calibration_file")
    def record_feedback(
        self,
        image_path: str,
        user_label: str,  # 'REAL' or 'AI'
        forensic_metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
        *,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Registers user or ground-truth verification of an image, updates memory bank,
        and dynamically adapts forensic thresholds. ``metrics`` is the modality-neutral alias of ``forensic_metrics``.
        """
        forensic_metrics = forensic_metrics or metrics or {}
        memory = self.load_memory()
        calib = self.load_calibration()
        verdict = str(user_label).strip().upper()
        if verdict not in ("AI", "REAL"):
            return calib                                # "Not sure" or anything else is not ground truth: nothing is recorded
        duplicate, digest = already_recorded(memory, image_path, verdict)
        if duplicate:
            logger.info("Feedback for this file and label is already recorded; calibration is not stepped again")
            return calib
        register_feedback(self.library, image_path, user_label)

        sanitized_metrics = {}
        for k, v in forensic_metrics.items():
            if k in ("spatial_heatmap", "image", "raw_image"):
                continue
            s_val = sanitize_metric_value(v)
            if s_val is not None:
                sanitized_metrics[k] = s_val

        record = ImageFeedbackRecord(
            timestamp=datetime.now().isoformat(),
            image_path=str(image_path),
            user_label=user_label.upper(),
            metrics=sanitized_metrics,
            notes=notes,
        )
        memory.append({**record.to_dict(), "sha256": digest})

        try:
            atomic_write_json(self.feedback_file, memory, indent=2)
        except Exception as e:
            logger.error("Failed to save image memory atomically: %s", e)

        # Dynamic recalibration based on new feedback
        weights = calib.setdefault("feature_weights", {})
        offsets = calib.setdefault("sensitivity_offsets", {})
        calib["samples_processed"] = len(memory)

        # The app hands over the whole detector result, whose measurements sit under "forensic_metrics"; a caller may also pass them flat.
        measured = {**forensic_metrics, **(forensic_metrics.get("forensic_metrics") or {})}
        noise = _measurement(measured, "noise_residual_mean")
        smooth = _measurement(measured, "surface_smoothness")

        # A measurement that is not in the record moves nothing: calibration follows what was measured, never a stand-in value.
        if user_label.upper() == "AI":
            if noise is not None and noise < 2.5:
                weights["noise_residual"] = min(0.60, weights.get("noise_residual", DEFAULT_FEATURE_WEIGHTS["noise_residual"]) + 0.02)
                offsets["noise_center_offset"] = max(-0.40, offsets.get("noise_center_offset", 0.0) - 0.03)
            if smooth is not None and smooth < 3.0:
                weights["surface_smoothness"] = min(0.55, weights.get("surface_smoothness", DEFAULT_FEATURE_WEIGHTS["surface_smoothness"]) + 0.02)
                offsets["smooth_center_offset"] = min(0.40, offsets.get("smooth_center_offset", 0.0) + 0.03)
        elif user_label.upper() == "REAL":
            if noise is not None and noise < 2.0:
                offsets["noise_center_offset"] = min(0.35, offsets.get("noise_center_offset", 0.0) + 0.03)
            if smooth is not None and smooth < 2.0:
                offsets["smooth_center_offset"] = max(-0.35, offsets.get("smooth_center_offset", 0.0) - 0.03)

        self.save_calibration(calib)
        logger.info("ImageSelfImprover updated calibration from feedback (Total records: %d)", len(memory))
        return calib
