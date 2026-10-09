"""
audio_detector.detector: Complete, self-contained Audio AI Detection engine.
Combines:
1. Native & FFmpeg multi-format audio decoding to 16kHz float32 PCM.
2. High-frequency brick-wall cutoff detection (a trait of many neural vocoders, and of some codecs).
3. Wiener spectral flatness analysis for unnatural harmonic smoothing.
4. Digital zero silence analysis for absent room tone.
5. Speech temporal timeline attribution.
6. Neural acoustic classifier inference (if checkpoint exists).
7. Rule-based feedback calibration (see learner.py -- adjusts scoring constants, not model weights).
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.shared_results import shift_probability_by_log_odds
from audio_detector.config import DEFAULT_AUDIO_CHECKPOINT
from audio_detector.features import (
    compute_spectral_features,
    feature_vector,
    generate_spectrogram_image,
    segment_audio_temporal,
)
from audio_detector.learner import AudioSelfImprover
from audio_detector.models.backbone import AudioClassifierNet, build_audio_classifier
from audio_detector.schemas import AudioForensicResult
from audio_detector.scoring import (
    calculate_audio_epistemic_uncertainty,
    evaluate_audio_decision,
    normalize_percentages,
    pool_acoustic_evidence,
)
from audio_detector.validator import AudioValidator

logger = logging.getLogger("audio_detector")


class AudioAIDetector:
    """
    Completely independent, self-contained Audio AI Detector with rule-based feedback calibration (see learner.py).
    Evaluates acoustic spectral anomalies, band-limit cutoffs and digital silence gaps.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        self_improver: Optional[AudioSelfImprover] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_AUDIO_CHECKPOINT
        self.self_improver = self_improver or AudioSelfImprover()
        self.validator = AudioValidator()
        self.model: Optional[AudioClassifierNet] = None
        self.device: Optional[torch.device] = None
        self._is_loaded = False
        self._load_lock = threading.RLock()

    def load(self) -> bool:
        """Loads once; concurrent callers wait for the first load instead of loading the model twice."""
        with self._load_lock:
            return self._load_locked()

    def _load_locked(self) -> bool:
        """Loads neural weights if checkpoint exists."""
        if self._is_loaded:
            return True
        if self.checkpoint_path.is_file():
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = build_audio_classifier(checkpoint_path=self.checkpoint_path, device=device)
                self.model = model
                self.device = device
                logger.info("AudioAIDetector loaded neural checkpoint: %s", self.checkpoint_path.name)
            except Exception as e:
                logger.warning("Could not load neural checkpoint, using the statistical acoustic analysis: %s", e)
        self._is_loaded = True
        return True

    def _neural_probability(self, spectral_feats: Dict[str, Any]) -> Optional[float]:
        """P(AI) from the optional neural head over five spectral features (None if unavailable or failing)."""
        if self.model is None or self.device is None:
            return None
        try:
            feat_vec = torch.tensor([feature_vector(spectral_feats)], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(feat_vec), dim=1)[0]
                return float(probs[0].item())
        except Exception as e:
            logger.debug("Neural inference bypassed: %s", e)
            return None

    @staticmethod
    def _forensic_cues(spectral_feats: Dict[str, Any], thresh: Dict[str, Any]) -> List[str]:
        cues: List[str] = []
        if spectral_feats.get("has_vocoder_cutoff"):
            cues.append(
                f"Sharp brick-wall frequency cutoff at {spectral_feats['cutoff_freq_hz']} Hz, well below the recording's Nyquist limit (neural vocoders do this; so do some codecs and band-limited sources)"
            )
        if spectral_feats.get("spectral_flatness", 1.0) < thresh.get("flatness_synthetic_max", 0.002):
            cues.append("Unnaturally smooth Wiener spectral flatness (synthetic voice harmonic profile)")
        if spectral_feats.get("digital_silence_ratio", 0.0) > thresh.get("silence_synthetic_min", 0.12):
            cues.append("Digital zero inter-phoneme silence gaps (absence of natural room tone)")
        return cues

    @staticmethod
    def _failure_result(path: Path, label: str, cue: str, status: str, error: str) -> Dict[str, Any]:
        """Result dict for unreadable / empty audio and for unexpected errors (same shape as a normal result)."""
        res = AudioForensicResult(
            valid=False, filename=path.name, label=label, prediction=label, forensic_cues=[cue], error=error,
        ).to_dict()
        res.update({
            "has_audio_track": False,
            "ai_duration_pct": 0.0,
            "spectral_features": {},
            "details": {"duration_seconds": 0.0, "status": status, "temporal_segments": []},
        })
        return res

    @staticmethod
    def _success_result(
        path: Path, sr: int, duration: float, spectral_feats: Dict[str, Any], temporal_segments: List[Dict[str, Any]],
        cues: List[str], spectrogram_img: Any, *, p_ai: float, p_real: float,
        percentages: Tuple[float, float, float], label: str,
    ) -> Dict[str, Any]:
        """The full result dict for an analysed track (typed result plus the legacy flat keys consumers read)."""
        ai_pct, real_pct, undecided_pct = percentages
        ai_duration_sec = sum(seg["duration_seconds"] for seg in temporal_segments if seg.get("label") == "LIKELY AI-GENERATED")
        ai_duration_pct = (ai_duration_sec / max(0.01, duration)) * 100.0 if duration > 0 else 0.0
        res = AudioForensicResult(
            valid=True,
            filename=path.name,
            duration_seconds=round(duration, 2),
            sample_rate=sr,
            ai_percentage=ai_pct,
            real_percentage=real_pct,
            undecided_percentage=undecided_pct,
            confidence=round(max(p_ai, p_real), 2),
            label=label,
            prediction=label,
            has_vocoder_cutoff=spectral_feats.get("has_vocoder_cutoff", False),
            cutoff_freq_hz=spectral_feats.get("cutoff_freq_hz", 0.0),
            spectral_flatness=spectral_feats.get("spectral_flatness", 0.0),
            digital_silence_ratio=spectral_feats.get("digital_silence_ratio", 0.0),
            high_freq_ratio=spectral_feats.get("high_freq_ratio", 0.0),
            acoustic_features=spectral_feats,
            temporal_segments=temporal_segments,
            forensic_cues=cues,
            spectrogram_image=spectrogram_img,
        ).to_dict()
        res.update({
            "has_audio_track": True,
            "ai_duration_pct": round(ai_duration_pct, 1),
            "spectral_features": spectral_feats,
            "spectrogram_image": spectrogram_img,
            "details": {
                "duration_seconds": round(duration, 2),
                "status": "Audible track extracted and evaluated.",
                "temporal_segments": temporal_segments,
            },
        })
        return res

    def analyze_audio_file(
        self,
        file_path: str | Path,
        sensitivity: str = "balanced",
        pre_extracted: Optional[Tuple[Optional[np.ndarray], int, float]] = None,
        generate_spectrogram: bool = False,
        extra_log_lrs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive acoustic AI voice clone and speech synthesis detection.
        ``extra_log_lrs``: optional capped base-10 log-odds terms from the dimension checks
        (see audio_detector.dimension_checks); None/empty leaves behavior unchanged.
        """
        self.load()
        path = Path(file_path)

        try:
            if pre_extracted is not None:
                samples, sr, duration = pre_extracted
            else:
                samples, sr, duration = self.validator.extract_pcm_samples(path)

            if samples is None or len(samples) < 1000:
                return self._failure_result(path, "NO_AUDIO", "No audio track found or audio stream unreadable.",
                                            "No audio track found.", "No audio track found or audio stream unreadable.")

            # 1. Dynamic Calibration from AudioSelfImprover
            calib = self.self_improver.load_calibration()
            weights = calib.get("acoustic_weights", {})
            thresh = calib.get("thresholds", {})
            offsets = calib.get("sensitivity_offsets", {})

            # 2. Spectral Feature Extraction
            spectral_feats = compute_spectral_features(samples, sr)
            temporal_segments = segment_audio_temporal(samples, sr)

            # 3. Neural Classifier Inference (if model available)
            neural_ai_prob = self._neural_probability(spectral_feats)

            # 4. Score Evidence Pooling
            sensitivity_offset = 0.0
            if sensitivity.lower() == "aggressive":
                sensitivity_offset = 0.08
            elif sensitivity.lower() == "high":
                sensitivity_offset = 0.04

            p_ai, p_real = pool_acoustic_evidence(
                features=spectral_feats,
                weights=weights,
                neural_prob=neural_ai_prob,
                sensitivity_offset=sensitivity_offset + offsets.get("audio_ai_offset", 0.0),
                thresholds=thresh,
            )

            p_ai, p_real, extra_cues = shift_probability_by_log_odds(p_ai, extra_log_lrs)

            # Epistemic Uncertainty via Shannon entropy
            entropy = calculate_audio_epistemic_uncertainty(p_ai)
            target_undecided = max(3.0, min(22.0, entropy * 18.0))

            ai_pct, real_pct, undecided_pct = normalize_percentages(
                ai_val=p_ai * 100.0,
                real_val=p_real * 100.0,
                undecided_val=target_undecided,
                min_undecided=3.0,
                decimals=1,
            )

            label = evaluate_audio_decision(ai_pct=ai_pct, real_pct=real_pct, sensitivity=sensitivity)

            cues = self._forensic_cues(spectral_feats, thresh) + extra_cues
            spectrogram_img = generate_spectrogram_image(samples, sr) if generate_spectrogram else None
            return self._success_result(
                path, sr, duration, spectral_feats, temporal_segments, cues, spectrogram_img,
                p_ai=p_ai, p_real=p_real, percentages=(ai_pct, real_pct, undecided_pct), label=label,
            )

        except Exception as exc:
            logger.error("Audio detection error on %s: %s", file_path, exc)
            return self._failure_result(path, "ERROR", f"Audio processing error: {exc}", f"Error: {exc}", str(exc))

    predict = analyze_audio_file
