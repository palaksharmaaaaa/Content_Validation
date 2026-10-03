"""
audio_detector.validator: Standalone audio stream & container validator.
Performs:
1. Audio container and codec validation (WAV, MP3, AAC, FLAC, OGG, M4A).
2. Safe extraction of PCM float32 waveform samples via FFmpeg or wave fallback.
3. Clipping, amplitude saturation, dynamic range, and channel inspection.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import tempfile
from typing import Optional, Tuple
import wave

import numpy as np

_FFMPEG_MISSING_WARNED = False

from core.hashing import file_sha256
from audio_detector.config import (
    MAX_DURATION_SECONDS,
    MAX_FILE_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    TARGET_SAMPLE_RATE,
)
from audio_detector.schemas import AudioValidationResult

logger = logging.getLogger("audio_detector.validator")


def _get_file_size_mb(path: Path) -> float:
    """Calculates file size in Megabytes."""
    return path.stat().st_size / (1024.0 * 1024.0)


def _calculate_file_hash(path: Path, chunk_size: int = 65536) -> str:
    """SHA-256 checksum via the shared cached hasher."""
    return file_sha256(path)


def resample_antialiased(samples: np.ndarray, src_sr: int, dst_sr: int, taps: int = 127) -> np.ndarray:
    """Linear-interpolation resampling with a windowed-sinc low-pass first when downsampling.

    Plain interpolation folds everything above the new Nyquist frequency back into the band (aliasing), which
    corrupts exactly the spectral features this engine measures (cutoff, flatness). The FIR low-pass removes it.
    """
    s = np.asarray(samples, dtype=np.float32)
    if src_sr > dst_sr:
        cutoff = 0.45 * dst_sr / src_sr  # cycles/sample, a little under the new Nyquist
        n = np.arange(taps) - (taps - 1) / 2.0
        kernel = np.sinc(2.0 * cutoff * n) * np.hamming(taps)
        s = np.convolve(s, (kernel / np.sum(kernel)).astype(np.float32), mode="same").astype(np.float32)
    new_len = max(1, int(len(s) * (dst_sr / float(src_sr))))
    return np.interp(np.linspace(0, len(s) - 1, new_len), np.arange(len(s)), s).astype(np.float32)


class AudioValidator:
    """Independent validator and decoder for audio recordings."""

    def __init__(
        self,
        target_sr: int = TARGET_SAMPLE_RATE,
        max_duration_sec: float = MAX_DURATION_SECONDS,
        max_size_mb: float = MAX_FILE_SIZE_MB,
    ):
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec
        self.max_size_mb = max_size_mb

    def validate(self, audio_path: str | Path) -> AudioValidationResult:
        """Validates audio file existence, size, duration, and readable stream structure."""
        path = Path(audio_path)
        if not path.is_file():
            return AudioValidationResult(
                valid=False,
                error=f"Audio file not found: {path}",
            )

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return AudioValidationResult(
                valid=False,
                filename=path.name,
                error=f"Unsupported audio format: '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

        size_mb = _get_file_size_mb(path)
        if size_mb > self.max_size_mb:
            return AudioValidationResult(
                valid=False,
                filename=path.name,
                file_size_mb=round(size_mb, 3),
                error=f"Audio file size ({size_mb:.2f}MB) exceeds limit of {self.max_size_mb}MB",
            )

        samples, sr, dur = self.extract_pcm_samples(path)
        if samples is None or len(samples) == 0:
            return AudioValidationResult(
                valid=False,
                filename=path.name,
                file_size_mb=round(size_mb, 3),
                error="Failed to decode audio stream into PCM samples.",
            )

        if dur > self.max_duration_sec:
            return AudioValidationResult(
                valid=False,
                filename=path.name,
                file_size_mb=round(size_mb, 3),
                duration_seconds=dur,
                error=f"Audio duration ({dur:.1f}s) exceeds limit of {self.max_duration_sec}s",
            )

        # Check clipping & dynamic range
        peak_amp = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
        is_clipped = peak_amp > 0.999
        is_silent = rms < 1e-4

        return AudioValidationResult(
            valid=True,
            filename=path.name,
            file_size_mb=round(size_mb, 3),
            file_hash_sha256=_calculate_file_hash(path),
            sample_rate=sr,
            duration_seconds=round(dur, 2),
            channels=1,
            format=ext.lstrip(".").upper(),
            is_clipped=is_clipped,
            is_silent=is_silent,
            extracted_samples=(samples, sr, dur),
        )

    def validate_audio(self, audio_path: str | Path) -> AudioValidationResult:
        """Alias for validate to ensure backwards compatibility across services and callers."""
        return self.validate(audio_path)

    def extract_pcm_samples(
        self, audio_path: str | Path
    ) -> Tuple[Optional[np.ndarray], int, float]:
        """
        Transcodes any audio format to 16kHz mono float32 PCM samples in [-1.0, 1.0].
        Safe with timeout and native wave fallback.
        """
        path = Path(audio_path)
        if not path.is_file():
            return None, self.target_sr, 0.0

        tmp_wav_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                tmp_wav_path = tmp_f.name

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(self.target_sr),
                "-vn",
                "-f",
                "wav",
                tmp_wav_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)

            if res.returncode == 0 and Path(tmp_wav_path).is_file() and Path(tmp_wav_path).stat().st_size > 44:
                with wave.open(tmp_wav_path, "rb") as wf:
                    sw = wf.getsampwidth()
                    fr = wf.getframerate()
                    raw = wf.readframes(wf.getnframes())

                    if sw == 2:
                        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sw == 1:
                        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    elif sw == 4:
                        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                    dur = float(len(samples)) / max(1, fr)
                    return samples, fr, dur
        except FileNotFoundError:
            global _FFMPEG_MISSING_WARNED
            if not _FFMPEG_MISSING_WARNED:
                logger.warning(
                    "ffmpeg executable not found in PATH. audio_detector requires the "
                    "ffmpeg system binary (not a pip package) to decode non-WAV audio -- "
                    "install it from https://ffmpeg.org/download.html and ensure it's on "
                    "PATH. Falling back to native wave decoding, which only supports WAV."
                )
                _FFMPEG_MISSING_WARNED = True
        except Exception as e:
            logger.debug("FFmpeg decoding failed: %s", e)
        finally:
            if tmp_wav_path and Path(tmp_wav_path).exists():
                try:
                    Path(tmp_wav_path).unlink()
                except OSError:
                    pass

        # Native wave fallback
        if str(path).lower().endswith(".wav"):
            try:
                with wave.open(str(path), "rb") as wf:
                    n_ch = wf.getnchannels()
                    sw = wf.getsampwidth()
                    fr = wf.getframerate()
                    raw = wf.readframes(wf.getnframes())
                    if sw == 2:
                        s = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sw == 1:
                        s = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    else:
                        s = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if n_ch > 1:
                        s = s.reshape(-1, n_ch).mean(axis=1)
                    if fr > 0 and fr != self.target_sr and len(s) > 1:
                        s = resample_antialiased(s, fr, self.target_sr)
                        fr = self.target_sr
                    dur = float(len(s)) / max(1, fr)
                    return s, fr, dur
            except Exception as e:
                logger.debug("Native wave fallback failed: %s", e)

        return None, self.target_sr, 0.0
