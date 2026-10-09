"""
video_detector.learner: Heuristic feedback-driven calibration engine for video forensics.

IMPORTANT -- what this module does NOT do: despite the "self-improving" language used
elsewhere in this project, record_feedback() below never touches the neural network's
weights and never retrains or fine-tunes anything (and in this project's current state,
no trained checkpoint even exists on disk -- see trainer.py). It only nudges scalar
constants (motion_thresholds, temporal_weights, sensitivity_offsets) stored in
data/video_calibration.json by small fixed deltas per feedback event, read by
scoring.pool_video_temporal_score() and temporal.compute_interframe_motion_variance(). This
is a legitimate technique -- rule-based calibration drift correction -- but it is not
machine learning.

Maintains a dedicated video feedback memory bank, adapts motion delta thresholds, and
optimizes the temporal scoring weights used by video_detector.scoring.
Depends only on core (atomic JSON I/O, media library) and this package.
"""
from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from video_detector.config import (
    DATA_DIR,
    DEFAULT_WEIGHT_FLICKER,
    DEFAULT_WEIGHT_FRAME_AI,
    DEFAULT_WEIGHT_WARPING,
    MOTION_VAR_HIGH_WARPING,
    MOTION_VAR_SUSPICIOUS_FLICKER,
    MOTION_VAR_UNNATURAL_FREEZE,
)
from video_detector.schemas import VideoFeedbackRecord
from core.metrics_util import sanitize_metric_value
from core.atomic_io import atomic_read_json, atomic_write_json, serialized_on
from core.media_library import MediaLibrary, library_for, register_feedback

logger = logging.getLogger("video_detector.learner")


class VideoSelfImprover:
    """
    Dedicated feedback-calibration module for Video AI Detection (adjusts scoring constants only -- see module docstring above; not model training).
    Maintains a persistent memory of verified authentic and AI-generated video fingerprints,
    and dynamically adapts temporal motion variance thresholds and frame scoring weights.
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
        self.feedback_file = Path(memory_file) if memory_file else (self.memory_dir / "video_feedback.json")
        self.calibration_file = Path(calibration_file) if calibration_file else (self.memory_dir / "video_calibration.json")
        # Verified files are queued for retraining by reference (no copy). When a test isolates
        # storage via memory_file, don't touch the real library unless one is passed in.
        self.library = library if library is not None else (None if memory_file else library_for("video"))

    def load_calibration(self) -> Dict[str, Any]:
        """Loads active video calibration parameters."""
        default_calib = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "samples_processed": 0,
            "motion_thresholds": {
                "high_warping_var": MOTION_VAR_HIGH_WARPING,
                "suspicious_flicker_var": MOTION_VAR_SUSPICIOUS_FLICKER,
                "unnatural_freeze_var": MOTION_VAR_UNNATURAL_FREEZE,
            },
            "temporal_weights": {
                "frame_ai_ratio": DEFAULT_WEIGHT_FRAME_AI,
                "motion_warping": DEFAULT_WEIGHT_WARPING,
                "diffusion_flicker": DEFAULT_WEIGHT_FLICKER,
            },
            "sensitivity_offsets": {
                "video_ai_offset": 0.0,
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
            logger.error("Failed to save video calibration atomically: %s", e)

    def load_memory(self) -> List[Dict[str, Any]]:
        """Loads verified video memory bank."""
        loaded = atomic_read_json(self.feedback_file, default=[])
        return loaded if isinstance(loaded, list) else []

    @serialized_on("feedback_file", "calibration_file")
    def record_feedback(
        self,
        video_path: str,
        user_label: str,  # 'REAL' or 'AI'
        video_metrics: Optional[Dict[str, Any]] = None,
        temporal_metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
        *,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Registers ground-truth video feedback and recalibrates temporal warping and flicker thresholds.
        """
        raw_metrics = video_metrics or temporal_metrics or metrics or {}
        memory = self.load_memory()
        calib = self.load_calibration()
        register_feedback(self.library, video_path, user_label)

        sanitized_metrics = {}
        for k, v in raw_metrics.items():
            if k in ("frames", "sampled_frames", "raw_frames"):
                continue
            s_val = sanitize_metric_value(v)
            if s_val is not None:
                sanitized_metrics[k] = s_val

        record = VideoFeedbackRecord(
            timestamp=datetime.now().isoformat(),
            video_path=str(video_path),
            user_label=user_label.upper(),
            metrics=sanitized_metrics,
            notes=notes,
        )
        memory.append(record.to_dict())

        try:
            atomic_write_json(self.feedback_file, memory, indent=2)
        except Exception as e:
            logger.error("Failed to save video memory atomically: %s", e)

        m_thresh = calib.setdefault("motion_thresholds", {})
        calib.setdefault("temporal_weights", {})
        offsets = calib.setdefault("sensitivity_offsets", {})
        calib["samples_processed"] = len(memory)

        tc = raw_metrics.get("temporal_consistency", {})
        motion_var = raw_metrics.get("motion_variance", tc.get("motion_variance"))
        motion_var = float(motion_var) if isinstance(motion_var, (int, float)) and not isinstance(motion_var, bool) else None

        if user_label.upper() == "AI":
            if motion_var is not None and motion_var < 100.0:
                m_thresh["suspicious_flicker_var"] = max(50.0, m_thresh.get("suspicious_flicker_var", MOTION_VAR_SUSPICIOUS_FLICKER) - 2.0)
                offsets["video_ai_offset"] = min(0.30, offsets.get("video_ai_offset", 0.0) + 0.03)
        elif user_label.upper() == "REAL":
            if motion_var is not None and motion_var > 60.0:
                m_thresh["high_warping_var"] = min(200.0, m_thresh.get("high_warping_var", MOTION_VAR_HIGH_WARPING) + 3.0)
                offsets["video_ai_offset"] = max(-0.30, offsets.get("video_ai_offset", 0.0) - 0.03)

        self.save_calibration(calib)
        logger.info("VideoSelfImprover updated calibration from feedback (Total records: %d)", len(memory))
        return calib
