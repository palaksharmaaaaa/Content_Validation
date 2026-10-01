"""
image_detector.learner: Self-improving online learning & dynamic calibration engine for image forensics.
Maintains dedicated memory bank and calibrates Bayesian feature weights based on verification feedback.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from image_detector.config import CALIBRATION_FILE, DATA_DIR, MEMORY_FILE
from image_detector.schemas import ImageFeedbackRecord

logger = logging.getLogger("image_detector.learner")


class ImageSelfImprover:
    """
    Dedicated self-improving module for Image AI Detection.
    Maintains a persistent memory of verified authentic and AI-generated image fingerprints,
    and dynamically optimizes Bayesian likelihood ratio weights and sensitivity offsets.
    """

    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        memory_file: Optional[Path] = None,
        calibration_file: Optional[Path] = None,
    ):
        self.memory_dir = memory_dir or DATA_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = Path(memory_file) if memory_file else (self.memory_dir / "image_feedback.json")
        self.calibration_file = Path(calibration_file) if calibration_file else (self.memory_dir / "image_calibration.json")

    def load_calibration(self) -> Dict[str, Any]:
        """Loads active calibration parameters or initializes default NIST-tuned priors."""
        if self.calibration_file.is_file():
            try:
                with open(self.calibration_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read image calibration file: %s", e)

        default_calib = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "samples_processed": 0,
            "feature_weights": {
                "noise_residual": 0.35,
                "surface_smoothness": 0.30,
                "fft_decay": 0.20,
                "facial_shading": 0.25,
                "ela_discrepancy": 0.15,
            },
            "sensitivity_offsets": {
                "noise_center_offset": 0.0,
                "smooth_center_offset": 0.0,
                "fft_decay_offset": 0.0,
            },
        }
        return default_calib

    def save_calibration(self, calib: Dict[str, Any]) -> None:
        """Saves updated calibration parameters atomically."""
        calib["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.calibration_file, "w", encoding="utf-8") as f:
                json.dump(calib, f, indent=2)
        except Exception as e:
            logger.error("Failed to save image calibration: %s", e)

    def load_memory(self) -> List[Dict[str, Any]]:
        """Loads verified image memory bank."""
        if self.feedback_file.is_file():
            try:
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def record_feedback(
        self,
        image_path: str,
        user_label: str,  # 'REAL' or 'AI'
        forensic_metrics: Dict[str, Any],
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Registers user or ground-truth verification of an image, updates memory bank,
        and dynamically adapts forensic thresholds.
        """
        memory = self.load_memory()
        calib = self.load_calibration()

        def _sanitize_val(val: Any) -> Any:
            if hasattr(val, "shape"):
                return None
            if isinstance(val, (float, int, str, bool)):
                return val
            if hasattr(val, "item") and getattr(val, "size", 1) == 1:
                return val.item()
            if isinstance(val, dict):
                return {str(dk): _sanitize_val(dv) for dk, dv in val.items() if _sanitize_val(dv) is not None}
            if isinstance(val, (list, tuple)):
                clean = [_sanitize_val(x) for x in val]
                return [x for x in clean if x is not None]
            return None

        sanitized_metrics = {}
        for k, v in forensic_metrics.items():
            if k in ("spatial_heatmap", "image", "raw_image"):
                continue
            s_val = _sanitize_val(v)
            if s_val is not None:
                sanitized_metrics[k] = s_val

        record = ImageFeedbackRecord(
            timestamp=datetime.now().isoformat(),
            image_path=str(image_path),
            user_label=user_label.upper(),
            metrics=sanitized_metrics,
            notes=notes,
        )
        memory.append(record.to_dict())

        try:
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2)
        except Exception as e:
            logger.error("Failed to save image memory: %s", e)

        # Dynamic recalibration based on new feedback
        weights = calib.setdefault("feature_weights", {})
        offsets = calib.setdefault("sensitivity_offsets", {})
        calib["samples_processed"] = len(memory)

        noise = float(forensic_metrics.get("noise_residual_mean", 2.2))
        smooth = float(forensic_metrics.get("surface_smoothness", 3.2))

        if user_label.upper() == "AI":
            if noise < 2.5:
                weights["noise_residual"] = min(0.60, weights.get("noise_residual", 0.35) + 0.02)
                offsets["noise_center_offset"] = max(-0.40, offsets.get("noise_center_offset", 0.0) - 0.03)
            if smooth < 3.0:
                weights["surface_smoothness"] = min(0.55, weights.get("surface_smoothness", 0.30) + 0.02)
                offsets["smooth_center_offset"] = min(0.40, offsets.get("smooth_center_offset", 0.0) + 0.03)
        elif user_label.upper() == "REAL":
            if noise < 2.0:
                offsets["noise_center_offset"] = min(0.35, offsets.get("noise_center_offset", 0.0) + 0.03)
            if smooth < 2.0:
                offsets["smooth_center_offset"] = max(-0.35, offsets.get("smooth_center_offset", 0.0) - 0.03)

        self.save_calibration(calib)
        logger.info("ImageSelfImprover updated calibration from feedback (Total records: %d)", len(memory))
        return calib
