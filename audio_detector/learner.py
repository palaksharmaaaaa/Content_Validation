"""
audio_detector.learner: Heuristic feedback-driven calibration engine for audio forensics.

IMPORTANT -- what this module does NOT do: despite the "self-improving" language used
elsewhere in this project, record_feedback() below never touches the neural network's
weights and never retrains or fine-tunes anything (and in this project's current state,
no trained checkpoint even exists on disk -- see trainer.py). It only nudges scalar
constants (acoustic_weights, thresholds, sensitivity_offsets) stored in
data/audio_calibration.json by small fixed deltas per feedback event, read by
scoring.pool_acoustic_evidence(). This is a legitimate technique -- rule-based calibration
drift correction -- but it is not machine learning.

Maintains a dedicated audio feedback memory bank, tunes vocoder cutoff thresholds, and
adapts the acoustic scoring weights used by audio_detector.scoring.
Depends only on core (atomic JSON I/O, media library) and this package.
"""
from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from audio_detector.config import (
    CALIBRATION_FILE,
    DATA_DIR,
    DEFAULT_WEIGHT_FLATNESS,
    DEFAULT_WEIGHT_HF_RATIO,
    DEFAULT_WEIGHT_SILENCE,
    DEFAULT_WEIGHT_VOCODER,
    DIGITAL_SILENCE_RATIO_THRESHOLD,
    MEMORY_FILE,
    SYNTHETIC_FLATNESS_LOW_THRESHOLD,
)
from audio_detector.schemas import AudioFeedbackRecord
from core.metrics_util import sanitize_metric_value
from core.hashing import already_recorded
from core.calibration_io import sanitize_calibration
from core.atomic_io import atomic_read_json, atomic_write_json, serialized_on
from core.media_library import MediaLibrary, library_for, register_feedback

logger = logging.getLogger("audio_detector.learner")


class AudioSelfImprover:
    """
    Dedicated feedback-calibration module for Audio AI Detection (adjusts scoring constants only -- see module docstring above; not model training).
    Maintains a persistent memory of verified authentic speech and AI voice clones,
    and dynamically adapts vocoder cutoff frequency limits and Wiener flatness weights.
    """

    def __init__(
        self,
        memory_file: Optional[Path] = None,
        calibration_file: Optional[Path] = None,
        library: Optional[MediaLibrary] = None,
    ):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.memory_file = memory_file or MEMORY_FILE
        self.calibration_file = calibration_file or CALIBRATION_FILE
        # Verified files are queued for retraining by reference (no copy). When a test isolates
        # storage via memory_file, don't touch the real library unless one is passed in.
        self.library = library if library is not None else (None if memory_file else library_for("audio"))

    def load_calibration(self) -> Dict[str, Any]:
        """Loads active audio calibration parameters."""
        default_calib = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "samples_processed": 0,
            "acoustic_weights": {
                "vocoder_cutoff": DEFAULT_WEIGHT_VOCODER,
                "spectral_flatness": DEFAULT_WEIGHT_FLATNESS,
                "silence_ratio": DEFAULT_WEIGHT_SILENCE,
                "high_freq_roll": DEFAULT_WEIGHT_HF_RATIO,
            },
            "thresholds": {
                "flatness_synthetic_max": SYNTHETIC_FLATNESS_LOW_THRESHOLD,
                "silence_synthetic_min": DIGITAL_SILENCE_RATIO_THRESHOLD,
            },
            "sensitivity_offsets": {
                "audio_ai_offset": 0.0,
            },
        }
        loaded = atomic_read_json(self.calibration_file, default=None)
        return sanitize_calibration(loaded, default_calib) if isinstance(loaded, dict) else default_calib

    def save_calibration(self, calib: Dict[str, Any]) -> None:
        """Saves updated calibration parameters atomically."""
        calib["last_updated"] = datetime.now().isoformat()
        try:
            atomic_write_json(self.calibration_file, calib, indent=2)
        except Exception as e:
            logger.error("Failed to save audio calibration atomically: %s", e)

    def load_memory(self) -> List[Dict[str, Any]]:
        """Loads verified audio memory bank."""
        loaded = atomic_read_json(self.memory_file, default=[])
        return loaded if isinstance(loaded, list) else []

    @serialized_on("memory_file", "calibration_file")
    def record_feedback(
        self,
        audio_path: str,
        user_label: str,  # 'REAL' or 'AI'
        acoustic_metrics: Optional[Dict[str, Any]] = None,
        voice_generator_tag: Optional[str] = None,
        notes: str = "",
        *,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Registers audio ground truth feedback and recalibrates acoustic thresholds.
        ``metrics`` is the modality-neutral alias of ``acoustic_metrics``.
        """
        acoustic_metrics = acoustic_metrics or metrics or {}
        memory = self.load_memory()
        calib = self.load_calibration()
        verdict = str(user_label).strip().upper()
        if verdict not in ("AI", "REAL"):
            return calib                                # "Not sure" or anything else is not ground truth: nothing is recorded
        duplicate, digest = already_recorded(memory, audio_path, verdict)
        if duplicate:
            logger.info("Feedback for this file and label is already recorded; calibration is not stepped again")
            return calib
        register_feedback(self.library, audio_path, user_label)

        sanitized_metrics = {}
        for k, v in acoustic_metrics.items():
            if k in ("spectrogram_image", "samples", "waveform"):
                continue
            s_val = sanitize_metric_value(v)
            if s_val is not None:
                sanitized_metrics[k] = s_val

        record = AudioFeedbackRecord(
            timestamp=datetime.now().isoformat(),
            audio_path=str(audio_path),
            user_label=user_label.upper(),
            features=sanitized_metrics,
            notes=notes if not voice_generator_tag else f"tag: {voice_generator_tag}; {notes}",
        ).to_dict()
        memory.append({**record, "sha256": digest})

        try:
            atomic_write_json(self.memory_file, memory, indent=2)
        except Exception as e:
            logger.error("Failed to save audio memory atomically: %s", e)

        weights = calib.setdefault("acoustic_weights", {})
        thresh = calib.setdefault("thresholds", {})
        offsets = calib.setdefault("sensitivity_offsets", {})
        calib["samples_processed"] = len(memory)

        flatness = acoustic_metrics.get("spectral_flatness")
        flatness = float(flatness) if isinstance(flatness, (int, float)) and not isinstance(flatness, bool) else None

        if user_label.upper() == "AI":
            # If synthetic voice was missed, boost vocoder weight and audio AI offset
            weights["vocoder_cutoff"] = min(0.60, weights.get("vocoder_cutoff", DEFAULT_WEIGHT_VOCODER) + 0.02)
            offsets["audio_ai_offset"] = min(0.35, offsets.get("audio_ai_offset", 0.0) + 0.03)
            if flatness is not None and flatness > 0.002:
                thresh["flatness_synthetic_max"] = min(0.010, thresh.get("flatness_synthetic_max", SYNTHETIC_FLATNESS_LOW_THRESHOLD) + 0.0005)
        elif user_label.upper() == "REAL":
            # If natural whisper / phone audio triggered false positive, lower audio AI offset
            offsets["audio_ai_offset"] = max(-0.35, offsets.get("audio_ai_offset", 0.0) - 0.03)

        self.save_calibration(calib)
        logger.info("AudioSelfImprover dynamically updated calibration (Total records: %d)", len(memory))
        return calib
