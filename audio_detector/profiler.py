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

import json
import logging
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.ffmpeg import ffprobe_input
from core.hashing import file_digests
import numpy as np

from audio_detector.validator import AudioValidator

logger = logging.getLogger("audio_detector.profiler")


def compute_file_hashes(file_path: str | Path) -> Tuple[str, str, int]:
    """Calculates SHA-256, MD5, and exact file size in bytes (shared cached implementation)."""
    return file_digests(file_path)


def _positive_int(value: Any) -> Optional[int]:
    """``int(value)`` when it is a positive number (ffprobe reports numbers as strings, and "0" for "unknown"), else None."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def read_stream_info(path: Path) -> Dict[str, Optional[Any]]:
    """What the file itself says about its audio stream: ``sample_rate`` (native), ``channels``, ``bit_depth`` and ``codec``.

    Everything the engine analyses is decoded to 16 kHz mono, so those numbers are not the file's. A PCM WAV is read from its own
    header; any other format is asked of ``ffprobe`` when it is installed. A value that cannot be determined is ``None``: it is
    shown as "not recorded", never guessed."""
    info: Dict[str, Optional[Any]] = {"sample_rate": None, "channels": None, "bit_depth": None, "codec": None}
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                info.update(sample_rate=w.getframerate(), channels=w.getnchannels(), bit_depth=8 * w.getsampwidth(), codec="pcm")
            return info
        except (wave.Error, EOFError, OSError):
            pass                                          # not plain PCM (float, extensible, damaged): ask ffprobe
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return info
    try:
        out = subprocess.run(ffprobe_input(path, ffprobe) + ["-select_streams", "a:0", "-show_entries",
                             "stream=sample_rate,channels,bits_per_raw_sample,bits_per_sample,codec_name", "-of", "json"],
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15, check=False)
        stream = (json.loads(out.stdout or "{}").get("streams") or [{}])[0]
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.debug("ffprobe failed for %s: %s", path, exc)
        return info
    bits = next((b for b in (_positive_int(stream.get(k)) for k in ("bits_per_raw_sample", "bits_per_sample")) if b), None)
    info.update(sample_rate=_positive_int(stream.get("sample_rate")), channels=_positive_int(stream.get("channels")),
                bit_depth=bits, codec=stream.get("codec_name"))
    return info


def _short_term_dynamic_range_db(samples: np.ndarray, sr: int, frame_ms: float = 50.0) -> float:
    """Spread between loud (95th percentile) and quiet (10th percentile) short-term RMS levels, in dB.

    Ignores digital silence. This is the dynamic range; peak-to-RMS (crest factor) is reported separately.
    """
    frame = max(1, int(sr * frame_ms / 1000.0))
    n = len(samples) // frame
    if n < 2:
        return 0.0
    rms = np.sqrt(np.mean(samples[: n * frame].reshape(n, frame) ** 2, axis=1))
    rms = rms[rms > 1e-5]
    if len(rms) < 2:
        return 0.0
    return float(20.0 * np.log10(np.percentile(rms, 95) / max(1e-9, np.percentile(rms, 10))))


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

        # Crest factor (peak vs. average level) and true loudness dynamic range (spread of short-term levels)
        crest_factor_db = 20.0 * np.log10(max(1e-5, peak) / max(1e-5, rms)) if rms > 0 else 0.0
        dr_db = _short_term_dynamic_range_db(samples, sr)

        is_clipped = peak >= 0.999
        is_silent = rms < 1e-4
        stream = read_stream_info(path)

        return {
            "valid": True,
            "filename": path.name,
            "file_size_bytes": size_bytes,
            "file_size_mb": round(size_mb, 3),
            "sha256": sha256,
            "md5": md5,
            "sample_rate": sr,                                  # the rate the engine analyses at (decoded), not the file's
            "native_sample_rate": stream["sample_rate"],
            "channels": stream["channels"],
            "bit_depth": stream["bit_depth"],
            "codec": stream["codec"],
            "geometry": {"sample_rate": stream["sample_rate"], "channels": stream["channels"], "bit_depth": stream["bit_depth"],
                         "duration_seconds": round(duration, 2)},
            "cryptographic_hashes": {"sha256": sha256, "md5": md5},
            "duration_seconds": round(duration, 2),
            "total_samples": len(samples),
            "peak_amplitude": round(peak, 3),
            "rms_energy": round(rms, 4),
            "crest_factor": round(crest_factor, 2),
            "dynamic_range_db": round(dr_db, 1),
            "crest_factor_db": round(crest_factor_db, 1),
            "is_clipped": is_clipped,
            "is_silent": is_silent,
            "format": path.suffix.lstrip(".").upper(),
        }
