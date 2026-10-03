"""
audio_detector.profiler: Technical Media Profiler and Signal Forensic Inspector for Audio.
Extracts:
1. Cryptographic identity (SHA-256, MD5, size in bytes).
2. Audio stream specifications (sample rate, channels, duration).
3. Acoustic signal metrics (RMS energy, peak amplitude, crest factor, dynamic range).
4. Digital clipping and zero-silence inspection.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from core.hashing import file_digests
import numpy as np

from audio_detector.validator import AudioValidator

logger = logging.getLogger("audio_detector.profiler")


def compute_file_hashes(file_path: str | Path) -> Tuple[str, str, int]:
    """Calculates SHA-256, MD5, and exact file size in bytes (shared cached implementation)."""
    return file_digests(file_path)


class AudioProfiler:
    """Profiles acoustic stream metrics and dynamic range of audio recordings."""

    def __init__(self):
        self.validator = AudioValidator()

    def profile_audio(self, file_path: str | Path) -> Dict[str, Any]:
        """Extracts complete signal specifications and hashes from an audio file."""
        path = Path(file_path)
        if not path.is_file():
            return {"valid": False, "error": f"File not found: {path}"}

        sha256, md5, size_bytes = compute_file_hashes(path)
        size_mb = size_bytes / (1024.0 * 1024.0)

        samples, sr, duration = self.validator.extract_pcm_samples(path)
        if samples is None or len(samples) == 0:
            return {
                "valid": False,
                "filename": path.name,
                "file_size_mb": round(size_mb, 3),
                "sha256": sha256,
                "md5": md5,
                "error": "Failed to decode audio samples.",
            }

        abs_s = np.abs(samples)
        peak = float(np.max(abs_s))
        rms = float(np.sqrt(np.mean(samples ** 2)))
        crest_factor = float(peak / max(1e-6, rms))

        # Dynamic range (dB)
        dr_db = 20.0 * np.log10(max(1e-5, peak) / max(1e-5, rms)) if rms > 0 else 0.0

        is_clipped = peak >= 0.999
        is_silent = rms < 1e-4

        return {
            "valid": True,
            "filename": path.name,
            "file_size_bytes": size_bytes,
            "file_size_mb": round(size_mb, 3),
            "sha256": sha256,
            "md5": md5,
            "sample_rate": sr,
            "duration_seconds": round(duration, 2),
            "total_samples": len(samples),
            "peak_amplitude": round(peak, 3),
            "rms_energy": round(rms, 4),
            "crest_factor": round(crest_factor, 2),
            "dynamic_range_db": round(dr_db, 1),
            "is_clipped": is_clipped,
            "is_silent": is_silent,
            "format": path.suffix.lstrip(".").upper(),
        }
