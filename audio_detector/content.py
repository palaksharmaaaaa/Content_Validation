"""
audio_detector.content: what kind of sound a recording is, and how it is delivered.

Measured from the whole recording (an average magnitude spectrum over up to ``MAX_FRAMES`` evenly spaced frames, plus the overall
level and crest factor), and described as measurements, not as claims about a room or a studio, which a level and a spectrum
cannot show:

  * ``dominant_modality``  speech-like (most energy in the 300-3400 Hz voice band), music-like (a lot of energy above 4 kHz in a
                           spectrum that is not flat) or ambient / noise-like (including broadband hiss)
  * ``delivery_style``     what the crest factor (peak over RMS) says about dynamics
  * ``signal_level``       how strong the signal is (RMS)

It does not count speakers or identify a language; callers must not expect those fields.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger("audio_detector.content")

FRAME = 1024
MAX_FRAMES = 400                 # evenly spaced over the recording: bounded work for an hour of audio
SPEECH_BAND = (300.0, 3400.0)
HIGH_BAND_FROM = 4000.0
SPEECH_RATIO_MIN = 0.65
HIGH_BAND_RATIO_MIN = 0.35
FLATNESS_NOISE_MIN = 0.40        # a spectrum this flat is hiss or ambience, not music, however much of it lies above 4 kHz
CREST_DYNAMIC = 8.0
CREST_COMPRESSED = 3.5
STRONG_LEVEL_RMS = 0.05

UNKNOWN = "Unknown"


class AudioContentAnalyzer:
    """Sound type, dynamics and level of an audio recording."""

    @staticmethod
    def _mean_spectrum(samples: np.ndarray) -> np.ndarray:
        """Average magnitude spectrum over evenly spaced Hann-windowed frames covering the whole recording."""
        n_fft = min(FRAME, len(samples))
        count = max(1, min(MAX_FRAMES, len(samples) // n_fft))
        starts = np.linspace(0, len(samples) - n_fft, num=count).astype(int)
        frames = np.stack([samples[s:s + n_fft] for s in starts]) * np.hanning(n_fft)
        return np.abs(np.fft.rfft(frames, axis=1)).mean(axis=0)

    def analyze_audio_scene(self, samples: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Dominant sound type, delivery style and level of a recording; ``Unknown`` values when it is too short to measure."""
        if samples is None or len(samples) < 500 or sample_rate <= 0:
            return {"signal_level": UNKNOWN, "dominant_modality": UNKNOWN, "delivery_style": UNKNOWN, "speech_ratio": None,
                    "rms_energy": None, "crest_factor": None}

        rms = float(np.sqrt(np.mean(samples ** 2)))
        crest_factor = float(np.max(np.abs(samples)) / max(1e-6, rms))
        mag = self._mean_spectrum(samples)
        freqs = np.fft.rfftfreq(min(FRAME, len(samples)), d=1.0 / sample_rate)
        total = float(np.sum(mag)) + 1e-10
        speech_ratio = float(np.clip(np.sum(mag[(freqs >= SPEECH_BAND[0]) & (freqs <= SPEECH_BAND[1])]) / total, 0.0, 1.0))
        high_ratio = float(np.sum(mag[freqs > HIGH_BAND_FROM]) / total)
        flatness = float(np.exp(np.mean(np.log(mag + 1e-10))) / (np.mean(mag) + 1e-10))

        if crest_factor > CREST_DYNAMIC:
            delivery = "Dynamic (peaks far above the average level)"
        elif crest_factor < CREST_COMPRESSED:
            delivery = "Heavily compressed (peaks close to the average level)"
        else:
            delivery = "Natural dynamics"

        if speech_ratio > SPEECH_RATIO_MIN:
            dominant = "Speech-like (most energy in the voice band)"
        elif high_ratio > HIGH_BAND_RATIO_MIN and flatness < FLATNESS_NOISE_MIN:
            dominant = "Music-like (strong high-frequency content)"
        else:
            dominant = "Ambient / noise-like"

        return {
            "signal_level": "Strong signal" if rms > STRONG_LEVEL_RMS else "Low signal",
            "dominant_modality": dominant,
            "delivery_style": delivery,
            "speech_ratio": round(speech_ratio, 2),
            "rms_energy": round(rms, 4),
            "crest_factor": round(crest_factor, 2),
        }
