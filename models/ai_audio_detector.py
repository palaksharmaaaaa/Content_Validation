"""
Multi-Feature Audio & Synthetic Voice Detector.
Detects AI-generated speech and voice clones (ElevenLabs, Bark, VALL-E, Tortoise, XTTS) using acoustic forensics:
1. High-frequency vocoder spectral cutoff & phase artifacts.
2. Spectral flatness (Wiener entropy) and lack of natural aspiration noise.
3. Digital zero silence gaps vs acoustic room tone.
4. Sliding-window temporal audio timeline attribution.
Supports audio tracks extracted directly from video containers (.mp4, .mov, etc.) and standalone audio files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from learning.forensic_memory import ForensicMemory
from schemas.result_schema import ModalityScore
from utils.audio_utils import (
    compute_spectral_features,
    extract_audio_samples,
    segment_audio_temporal,
)
from utils.logging_utils import get_logger

logger = get_logger("ai_audio_detector")


class AIAudioDetector:
    def __init__(self):
        self.is_available = True
        self.backend = "multimodal_acoustic_forensics"
        self.memory = ForensicMemory()

    def analyze_audio_file(self, file_path: str | Path, sensitivity: str = "high") -> Dict[str, Any]:
        """
        Extracts and analyzes audio from either a video container (.mp4, .mov, .mkv)
        or a standalone audio file (.mp3, .wav, .m4a, .aac, .flac).
        """
        samples, sr, duration = extract_audio_samples(file_path)

        if samples is None or len(samples) < 1000:
            return {
                "has_audio_track": False,
                "ai_percentage": 0.0,
                "real_percentage": 0.0,
                "undecided_percentage": 100.0,
                "confidence": 0.0,
                "label": "NO_AUDIO_TRACK",
                "details": {
                    "duration_seconds": 0.0,
                    "status": "No audible stream or track found in media container.",
                    "temporal_segments": [],
                },
            }

        calib = self.memory.load_calibration()
        weights = calib.get("feature_weights", {})
        offsets = calib.get("sensitivity_offsets", {})
        audio_offset = offsets.get("audio_ai_offset", 0.0)

        spectral_feats = compute_spectral_features(samples, sr)
        raw_p_ai = spectral_feats.get("p_audio_ai", 0.5)

        # Sensitivity threshold adjustment with learned offsets
        if sensitivity == "aggressive":
            calibrated_p_ai = min(0.96, raw_p_ai * 1.15 + 0.05 + audio_offset)
        elif sensitivity == "high":
            calibrated_p_ai = min(0.95, raw_p_ai * 1.08 + 0.02 + audio_offset)
        else:
            calibrated_p_ai = np.clip(raw_p_ai + audio_offset, 0.05, 0.95)

        # Query Forensic Memory for similar audio clips / voice profiles
        forensic_snapshot = {"spectral_features": spectral_feats}
        mem_res = self.memory.query_similar_media(file_path, modality="audio", forensic_data=forensic_snapshot)
        if mem_res.get("has_matches"):
            ai_boost = mem_res["prior_adjustment"]["ai_boost"]
            calibrated_p_ai = float(np.clip(calibrated_p_ai + ai_boost, 0.03, 0.97))

        calibrated_p_real = max(0.04, 1.0 - calibrated_p_ai)

        # Margin and uncertainty
        gap = abs(calibrated_p_ai - calibrated_p_real)
        undecided_pct = max(3.0, (1.0 - gap) * 25.0)
        remaining = 100.0 - undecided_pct

        ai_pct = round((calibrated_p_ai / (calibrated_p_ai + calibrated_p_real)) * remaining, 1)
        real_pct = round((calibrated_p_real / (calibrated_p_ai + calibrated_p_real)) * remaining, 1)
        undecided_pct = round(100.0 - (ai_pct + real_pct), 1)

        threshold = 50.0 if sensitivity in ("high", "aggressive") else 58.0
        if ai_pct >= threshold:
            label = "LIKELY AI-GENERATED"
        elif real_pct >= 58.0:
            label = "LIKELY REAL"
        else:
            label = "UNDECIDED"

        temporal_segments = segment_audio_temporal(samples, sr, window_sec=2.0)
        ai_seg_secs = sum(s["duration_seconds"] for s in temporal_segments if s["label"] == "LIKELY AI-GENERATED")
        ai_duration_pct = (ai_seg_secs / max(0.1, duration)) * 100.0 if duration > 0 else ai_pct

        cues = []
        if mem_res.get("has_matches"):
            cues.append(f"🧠 {mem_res['explanation']}")
        if spectral_feats.get("has_vocoder_cutoff"):
            cues.append("Sharp neural vocoder high-frequency cutoff detected (steep spectral drop above 7.5 kHz)")
        if spectral_feats.get("spectral_flatness", 0.0) < 0.002:
            cues.append("Unnaturally smooth Wiener spectral flatness (synthetic voice harmonic profile)")
        if spectral_feats.get("digital_silence_ratio", 0.0) > 0.18:
            cues.append("Digital zero inter-phoneme silence gaps (absence of natural room tone)")

        return {
            "has_audio_track": True,
            "ai_percentage": ai_pct,
            "real_percentage": real_pct,
            "undecided_percentage": undecided_pct,
            "confidence": round(max(calibrated_p_ai, calibrated_p_real), 2),
            "label": label,
            "ai_duration_pct": round(ai_duration_pct, 1),
            "forensic_cues": cues,
            "spectral_features": spectral_feats,
            "temporal_segments": temporal_segments,
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "memory_match": mem_res,
        }
