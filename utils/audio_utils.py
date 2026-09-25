"""
Audio forensic processing utilities: Audio stream extraction from video/audio containers via FFmpeg,
spectral cutoff profiling, Wiener entropy, silence/room tone consistency, and temporal audio segmentation.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import wave

import numpy as np

from utils.logging_utils import get_logger

logger = get_logger("audio_utils")


def extract_audio_samples(
    media_path: str | Path, target_sr: int = 16000
) -> Tuple[Optional[np.ndarray], int, float]:
    """
    Extracts audio from any video container (.mp4, .mov, .mkv, .webm) or converts any audio format
    (.mp3, .m4a, .aac, .flac, .ogg, .wav) to 16kHz mono float32 PCM using FFmpeg.
    Returns: (samples_float32, sample_rate, duration_seconds)
    """
    media_path = str(media_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name

    try:
        # Run ffmpeg to extract/transcode to 16kHz 16-bit mono PCM WAV
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            media_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(target_sr),
            "-ac",
            "1",
            tmp_wav_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0 or not Path(tmp_wav_path).is_file() or Path(tmp_wav_path).stat().st_size == 0:
            return None, target_sr, 0.0

        with wave.open(tmp_wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return None, target_sr, 0.0
            raw_bytes = wf.readframes(n_frames)
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            duration = float(n_frames) / target_sr
            return samples, target_sr, duration

    except Exception as exc:
        logger.info("Could not extract audio via ffmpeg: %s", exc)
        return None, target_sr, 0.0

    finally:
        if Path(tmp_wav_path).exists():
            try:
                Path(tmp_wav_path).unlink()
            except OSError:
                pass


def compute_spectral_features(samples: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Computes acoustic features indicative of neural vocoders and synthetic speech:
    1. Vocoder high-frequency brick-wall cutoff (> 7.5kHz vs < 4kHz).
    2. Wiener spectral flatness (unnatural harmonic smoothness).
    3. Digital zero silence gaps vs natural acoustic room tone.
    """
    if len(samples) < 1000 or sample_rate <= 0:
        return {
            "has_vocoder_cutoff": False,
            "cutoff_freq_hz": sample_rate // 2,
            "spectral_flatness": 0.05,
            "digital_silence_ratio": 0.0,
            "high_freq_ratio": 0.05,
            "p_audio_ai": 0.5,
        }

    # Short-Time Fourier Transform
    n_fft = min(1024, len(samples))
    hop_length = n_fft // 2
    window = np.hanning(n_fft)
    num_frames = (len(samples) - n_fft) // hop_length + 1

    if num_frames <= 0:
        return {"has_vocoder_cutoff": False, "p_audio_ai": 0.5}

    specs = []
    for i in range(num_frames):
        chunk = samples[i * hop_length : i * hop_length + n_fft] * window
        mag = np.abs(np.fft.rfft(chunk))
        specs.append(mag)

    spec_mat = np.array(specs)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # 1. High-frequency Cutoff Analysis
    # Neural vocoders (HiFi-GAN, MelGAN, older ElevenLabs/Bark) drop sharply above 7.5kHz or 11kHz
    hf_mask = freqs > 7500
    lf_mask = (freqs >= 300) & (freqs <= 3500)

    hf_energy = float(np.mean(spec_mat[:, hf_mask])) if np.any(hf_mask) else 0.0
    lf_energy = float(np.mean(spec_mat[:, lf_mask])) if np.any(lf_mask) else 1e-6
    hf_ratio = hf_energy / (lf_energy + 1e-9)

    # Sharp brick-wall cutoff typical of 16kHz/22kHz-upsampled TTS
    is_vocoder_cutoff = bool(hf_ratio < 0.007 if np.any(hf_mask) else False)

    # 2. Wiener Spectral Flatness
    power_spec = spec_mat ** 2 + 1e-12
    geom_mean = np.exp(np.mean(np.log(power_spec), axis=1))
    arith_mean = np.mean(power_spec, axis=1)
    flatness = float(np.mean(geom_mean / (arith_mean + 1e-9)))

    # 3. Digital Zero Silence Detection
    # AI generated audio frequently contains literal 0.0 or near-zero samples between phonemes
    near_zero = np.abs(samples) < 1e-4
    digital_silence_ratio = float(np.sum(near_zero) / float(len(samples)))

    # Continuous probabilistic synthetic audio scoring
    p_cutoff = 0.90 if is_vocoder_cutoff else 0.20
    # Overly smooth or extreme flatness
    p_flatness = 0.85 if (flatness < 0.002 or flatness > 0.35) else 0.25
    # High digital silence
    p_silence = 0.80 if digital_silence_ratio > 0.18 else 0.30

    p_audio_ai = float((p_cutoff * 0.45) + (p_flatness * 0.35) + (p_silence * 0.20))

    return {
        "has_vocoder_cutoff": is_vocoder_cutoff,
        "high_freq_ratio": round(hf_ratio, 5),
        "spectral_flatness": round(flatness, 5),
        "digital_silence_ratio": round(digital_silence_ratio, 4),
        "p_audio_ai": round(p_audio_ai, 3),
    }


def segment_audio_temporal(
    samples: np.ndarray, sample_rate: int, window_sec: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Evaluates audio in sliding temporal windows and groups into contiguous intervals:
    e.g. [00:00 - 00:04 REAL], [00:04 - 00:16 AI GENERATED]
    """
    total_duration = len(samples) / float(sample_rate)
    if total_duration < 1.0:
        return []

    window_size = int(window_sec * sample_rate)
    hop_size = window_size
    num_windows = max(1, len(samples) // hop_size)

    window_results = []
    for i in range(num_windows):
        start_t = i * window_sec
        end_t = min(total_duration, (i + 1) * window_sec)
        chunk = samples[i * hop_size : min(len(samples), (i + 1) * hop_size)]
        feats = compute_spectral_features(chunk, sample_rate)
        p_ai = feats.get("p_audio_ai", 0.5)

        if p_ai >= 0.60:
            lbl = "LIKELY AI-GENERATED"
        elif p_ai <= 0.35:
            lbl = "LIKELY REAL"
        else:
            lbl = "UNDECIDED"

        window_results.append({
            "start_seconds": round(start_t, 2),
            "end_seconds": round(end_t, 2),
            "label": lbl,
            "ai_prob": p_ai,
        })

    # Group adjacent windows with identical label
    segments = []
    if not window_results:
        return segments

    curr_lbl = window_results[0]["label"]
    curr_start = window_results[0]["start_seconds"]
    curr_end = window_results[0]["end_seconds"]

    for w in window_results[1:]:
        if w["label"] == curr_lbl:
            curr_end = w["end_seconds"]
        else:
            segments.append({
                "start_seconds": curr_start,
                "end_seconds": curr_end,
                "duration_seconds": round(max(0.0, curr_end - curr_start), 2),
                "label": curr_lbl,
            })
            curr_lbl = w["label"]
            curr_start = w["start_seconds"]
            curr_end = w["end_seconds"]

    segments.append({
        "start_seconds": curr_start,
        "end_seconds": curr_end,
        "duration_seconds": round(max(0.0, curr_end - curr_start), 2),
        "label": curr_lbl,
    })
    return segments


def generate_spectrogram_image(samples: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """
    Generates an RGB spectrogram heatmap image from audio PCM samples.
    Visualizes frequency distribution over time to expose vocoder cutoff lines
    and synthetic harmonic smoothing.
    """
    if len(samples) < 512 or sample_rate <= 0:
        return None
    try:
        import cv2

        n_fft = min(1024, len(samples))
        hop_length = n_fft // 4
        window = np.hanning(n_fft)
        num_frames = (len(samples) - n_fft) // hop_length + 1
        if num_frames < 2:
            return None

        specs = []
        for i in range(num_frames):
            chunk = samples[i * hop_length : i * hop_length + n_fft] * window
            mag = np.abs(np.fft.rfft(chunk))
            specs.append(mag)

        spec_mat = np.array(specs).T  # (freq_bins, time_frames)
        spec_mat = np.flipud(spec_mat)  # lowest frequency at the bottom

        spec_db = 20 * np.log10(np.maximum(spec_mat, 1e-6))
        min_db = float(np.percentile(spec_db, 5))
        max_db = float(np.percentile(spec_db, 95))
        if max_db > min_db:
            spec_norm = np.clip((spec_db - min_db) / (max_db - min_db) * 255.0, 0, 255).astype(np.uint8)
        else:
            spec_norm = np.zeros_like(spec_db, dtype=np.uint8)

        spec_resized = cv2.resize(spec_norm, (640, 220), interpolation=cv2.INTER_LINEAR)
        heatmap_bgr = cv2.applyColorMap(spec_resized, cv2.COLORMAP_MAGMA)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        return heatmap_rgb
    except Exception as exc:
        logger.debug(f"Spectrogram generation failed: {exc}")
        return None
