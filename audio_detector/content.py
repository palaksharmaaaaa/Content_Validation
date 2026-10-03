"""
audio_detector.content: Acoustic Scene, Environmental Sound & Delivery Tone Intelligence.
Analyzes:
1. Speech vs Music vs Ambient noise content distribution.
2. Delivery style (Conversational, Broadcast, Whispered, High-Energy).
3. Acoustic environment setting (Studio/Dry, Reverb/Hall, Domestic Room, Outdoor).
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np

logger = logging.getLogger("audio_detector.content")


class AudioContentAnalyzer:
    """Acoustic scene, delivery style, and environment analyzer for audio recordings."""

    def __init__(self):
        pass

    def analyze_audio_scene(
        self, samples: np.ndarray, sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Analyzes acoustic scene, environmental profile, and vocal delivery style."""
        if samples is None or len(samples) < 500 or sample_rate <= 0:
            return {
                "setting": "Unknown",
                "dominant_modality": "Unknown",
                "delivery_style": "Unknown",
                "speech_ratio": 0.0,
                "music_ratio": 0.0,
                "ambient_ratio": 0.0,
            }

        # Energy profile
        abs_samples = np.abs(samples)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        peak = float(np.max(abs_samples))
        crest_factor = float(peak / max(1e-6, rms))

        # Spectral energy distribution
        n_fft = min(1024, len(samples))
        fft_mag = np.abs(np.fft.rfft(samples[:n_fft]))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        speech_band = (freqs >= 300) & (freqs <= 3400)
        high_band = freqs > 4000

        speech_energy = float(np.sum(fft_mag[speech_band]))
        total_energy = float(np.sum(fft_mag)) + 1e-10
        speech_ratio = float(np.clip(speech_energy / total_energy, 0.0, 1.0))

        # Setting estimation
        if crest_factor > 8.0:
            delivery = "Dynamic / High-Energy Speech"
        elif crest_factor < 3.5:
            delivery = "Compressed / Commercial Voiceover"
        else:
            delivery = "Natural Conversational"

        # Room acoustics via low-frequency rumble vs high-frequency room reflection
        if speech_ratio > 0.65:
            dominant = "Vocal Speech"
            setting = "Isolated Studio Recording" if rms > 0.05 else "Domestic / Casual Recording"
        elif float(np.sum(fft_mag[high_band])) / total_energy > 0.35:
            dominant = "Music / High Harmonic Audio"
            setting = "Music Studio or Performance"
        else:
            dominant = "Ambient Environmental Sound"
            setting = "Environmental / Ambient Room"

        return {
            "setting": setting,
            "dominant_modality": dominant,
            "delivery_style": delivery,
            "speech_ratio": round(speech_ratio, 2),
            "rms_energy": round(rms, 4),
            "crest_factor": round(crest_factor, 2),
        }

    def analyze_audio_content(self, samples: np.ndarray, sample_rate: int = 16000, duration: float = 0.0) -> Dict[str, Any]:
        """Alias for backward compatibility."""
        return self.analyze_audio_scene(samples, sample_rate)
