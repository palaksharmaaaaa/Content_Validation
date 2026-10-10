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

from core.ffmpeg import ffmpeg_input, pcm_to_float
from core.hashing import file_sha256
from audio_detector.config import (
    MAX_DURATION_SECONDS,
    MAX_FILE_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    TARGET_SAMPLE_RATE,
)
from audio_detector.schemas import AudioValidationResult

logger = logging.getLogger("audio_detector.validator")

_FFMPEG_MISSING_WARNED = False


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


MAX_NATIVE_SAMPLE_RATE = 768_000          # above the highest rate any real audio format uses (DXD is 352.8 kHz)


def _is_near_silent(samples: np.ndarray) -> bool:
    """True when the RMS level is below the threshold the validator calls silent."""
    return len(samples) == 0 or float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) < 1e-4


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


    def _ffmpeg_decode(self, path: Path, tmp_wav_path: str, downmix: list) -> Optional[Tuple[np.ndarray, int]]:
        """Decode ``path`` to 16 kHz mono PCM with the given channel option; None if ffmpeg fails or writes nothing."""
        cmd = ffmpeg_input(path, max_seconds=self.max_duration_sec + 1.0) + ["-y", *downmix, "-ar", str(self.target_sr), "-vn", "-f", "wav", tmp_wav_path]
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if res.returncode != 0 or not Path(tmp_wav_path).is_file() or Path(tmp_wav_path).stat().st_size <= 44:
            return None
        with wave.open(tmp_wav_path, "rb") as wf:
            sw, fr = wf.getsampwidth(), wf.getframerate()
            return pcm_to_float(wf.readframes(wf.getnframes()), sw), fr

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

            # Decode at most one second past the limit: a file longer than that is rejected anyway, so the rest is never decoded.
            decoded = self._ffmpeg_decode(path, tmp_wav_path, downmix=["-ac", "1"])
            if decoded is not None and _is_near_silent(decoded[0]):
                # Averaging the channels cancels a stereo file whose channels are opposite in phase, and a hollow-sounding mix would then be
                # read as "no audio". Take the first channel alone before concluding the file is silent.
                first = self._ffmpeg_decode(path, tmp_wav_path, downmix=["-af", "pan=mono|c0=c0"])
                if first is not None and not _is_near_silent(first[0]):
                    decoded = first
            if decoded is not None:
                samples, fr = decoded
                return samples, fr, float(len(samples)) / max(1, fr)
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
                    if not (1 <= fr <= MAX_NATIVE_SAMPLE_RATE):
                        return None, self.target_sr, 0.0              # a header sample rate no real recording has
                    # Read no more than the duration limit allows (plus the one second that proves the file is too long).
                    raw = wf.readframes(min(wf.getnframes(), int((self.max_duration_sec + 1.0) * fr)))
                    s = pcm_to_float(raw, sw)
                    if n_ch > 1:
                        frames = s[: len(s) // n_ch * n_ch].reshape(-1, n_ch)
                        s = frames.mean(axis=1)
                        if _is_near_silent(s) and not _is_near_silent(frames[:, 0]):
                            s = frames[:, 0]                          # channels in opposite phase cancel in the mean; use one on its own
                    if fr > 0 and fr != self.target_sr and len(s) > 1:
                        s = resample_antialiased(s, fr, self.target_sr)
                        fr = self.target_sr
                    dur = float(len(s)) / max(1, fr)
                    return s, fr, dur
            except Exception as e:
                logger.debug("Native wave fallback failed: %s", e)

        return None, self.target_sr, 0.0
