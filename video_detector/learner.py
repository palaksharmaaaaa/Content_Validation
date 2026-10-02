"""
video_detector.learner: Self-improving online learning & dynamic calibration engine for video forensics.
Maintains dedicated video memory bank, adapts motion delta thresholds, and optimizes temporal segmentation.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from video_detector.config import CALIBRATION_FILE, DATA_DIR, MEMORY_FILE
from video_detector.schemas import VideoFeedbackRecord
from core.atomic_io import atomic_read_json, atomic_write_json

logger = logging.getLogger("video_detector.learner")


class VideoSelfImprover:
    """
    Dedicated self-improving module for Video AI Detection.
    Maintains a persistent memory of verified authentic and AI-generated video fingerprints,
    and dynamically adapts temporal motion variance thresholds and frame scoring weights.
    """

    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        memory_file: Optional[Path] = None,
        calibration_file: Optional[Path] = None,
    ):
        self.memory_dir = memory_dir or DATA_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = Path(memory_file) if memory_file else (self.memory_dir / "video_feedback.json")
        self.calibration_file = Path(calibration_file) if calibration_file else (self.memory_dir / "video_calibration.json")

    def load_calibration(self) -> Dict[str, Any]:
        """Loads active video calibration parameters."""
        default_calib = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "samples_processed": 0,
            "motion_thresholds": {
                "high_warping_var": 140.0,
                "suspicious_flicker_var": 75.0,
                "unnatural_freeze_var": 0.8,
            },
            "temporal_weights": {
                "frame_ai_ratio": 0.60,
                "motion_warping": 0.25,
                "diffusion_flicker": 0.15,
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

    def record_feedback(
        self,
        video_path: str,
        user_label: str,  # 'REAL' or 'AI'
        video_metrics: Optional[Dict[str, Any]] = None,
        temporal_metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Registers ground-truth video feedback and recalibrates temporal warping and flicker thresholds.
        """
        raw_metrics = video_metrics or temporal_metrics or kwargs.get("metrics") or {}
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
        for k, v in raw_metrics.items():
            if k in ("frames", "sampled_frames", "raw_frames"):
                continue
            s_val = _sanitize_val(v)
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
        t_weights = calib.setdefault("temporal_weights", {})
        offsets = calib.setdefault("sensitivity_offsets", {})
        calib["samples_processed"] = len(memory)

        tc = raw_metrics.get("temporal_consistency", {})
        motion_var = float(raw_metrics.get("motion_variance", tc.get("motion_variance", 50.0)))

        if user_label.upper() == "AI":
            if motion_var < 100.0:
                m_thresh["suspicious_flicker_var"] = max(50.0, m_thresh.get("suspicious_flicker_var", 75.0) - 2.0)
                offsets["video_ai_offset"] = min(0.30, offsets.get("video_ai_offset", 0.0) + 0.03)
        elif user_label.upper() == "REAL":
            if motion_var > 60.0:
                m_thresh["high_warping_var"] = min(200.0, m_thresh.get("high_warping_var", 140.0) + 3.0)
                offsets["video_ai_offset"] = max(-0.30, offsets.get("video_ai_offset", 0.0) - 0.03)

        self.save_calibration(calib)
        logger.info("VideoSelfImprover updated calibration from feedback (Total records: %d)", len(memory))
        return calib
