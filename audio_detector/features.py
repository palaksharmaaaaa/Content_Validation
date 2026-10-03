"""
audio_detector.features: Acoustic feature extraction, vocoder cutoff analysis, and spectrogram rendering.
Analyzes:
1. Neural vocoder high-frequency brick-wall cutoffs (e.g. 7.5kHz / 16kHz).
2. Wiener spectral flatness (synthetic voice harmonic over-smoothing).
3. Digital zero silence gaps vs natural room tone.
4. Spectrogram visualization rendering.
5. Temporal audio segmentation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


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
        return {
            "has_vocoder_cutoff": False,
            "cutoff_freq_hz": sample_rate // 2,
            "spectral_flatness": 0.05,
            "digital_silence_ratio": 0.0,
            "high_freq_ratio": 0.05,
            "p_audio_ai": 0.5,
        }

    specs = []
    for i in range(num_frames):
        chunk = samples[i * hop_length : i * hop_length + n_fft] * window
        mag = np.abs(np.fft.rfft(chunk))
        specs.append(mag)

    spec_mat = np.array(specs)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # 1. High-frequency Cutoff Analysis
    # Neural vocoders (HiFi-GAN, MelGAN, WaveGlow) often terminate sharply at 7.5kHz or 16kHz
    mean_power_by_freq = np.mean(spec_mat, axis=0)
    total_power = np.sum(mean_power_by_freq) + 1e-10

    # Cumulative energy distribution
    cum_energy = np.cumsum(mean_power_by_freq) / total_power
    cutoff_idx = np.searchsorted(cum_energy, 0.985)
    cutoff_idx = min(cutoff_idx, len(freqs) - 1)
    cutoff_freq = freqs[cutoff_idx]

    has_vocoder_cutoff = bool(
        (6500 <= cutoff_freq <= 8200 and sample_rate >= 16000)
        or (15000 <= cutoff_freq <= 16500 and sample_rate >= 44100)
    )

    # 2. Wiener Spectral Flatness: Geometric Mean / Arithmetic Mean of Power
    # Natural human speech has rich formants and harmonic irregularity (lower flatness)
    # Neural synthesizers often have hyper-regular or unnaturally flat spectral noise
    power_spectrum = mean_power_by_freq + 1e-12
    geom_mean = float(np.exp(np.mean(np.log(power_spectrum))))
    arith_mean = float(np.mean(power_spectrum))
    spectral_flatness = float(geom_mean / max(1e-12, arith_mean))

    # 3. Digital Zero Silence Gaps vs Natural Room Tone
    # AI voice generators frequently splice phonemes with mathematical zero amplitude (silence = 0.0)
    # Microphones in physical rooms always record ambient baseline noise (entropy > 0)
    abs_samples = np.abs(samples)
    digital_zero_count = np.sum(abs_samples < 1e-5)
    digital_silence_ratio = float(digital_zero_count / max(1, len(samples)))

    # High frequency energy ratio (> 5kHz)
    hf_mask = freqs >= 5000
    high_freq_ratio = float(np.sum(mean_power_by_freq[hf_mask]) / total_power)

    # Heuristic combined indicator
    p_score = 0.10
    if has_vocoder_cutoff:
        p_score += 0.50
    if spectral_flatness < 0.002:
        p_score += 0.25
    elif spectral_flatness > 0.45:
        p_score += 0.15
    if digital_silence_ratio > 0.12:
        p_score += 0.30

    p_score = float(np.clip(p_score, 0.05, 0.95))

    return {
        "has_vocoder_cutoff": has_vocoder_cutoff,
        "cutoff_freq_hz": round(float(cutoff_freq), 1),
        "spectral_flatness": round(spectral_flatness, 6),
        "digital_silence_ratio": round(digital_silence_ratio, 4),
        "high_freq_ratio": round(high_freq_ratio, 4),
        "p_audio_ai": round(p_score, 3),
    }


def generate_spectrogram_image(samples: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """Generates an RGB spectral heatmap image representing frequency energy over time."""
    if len(samples) < 500 or sample_rate <= 0:
        return None

    n_fft = min(512, len(samples))
    hop_length = n_fft // 2
    window = np.hanning(n_fft)
    num_frames = (len(samples) - n_fft) // hop_length + 1

    if num_frames <= 0:
        return None

    specs = []
    for i in range(num_frames):
        chunk = samples[i * hop_length : i * hop_length + n_fft] * window
        mag = np.abs(np.fft.rfft(chunk))
        specs.append(mag)

    spec_mat = np.array(specs).T  # Shape: (freq_bins, time_frames)
    spec_db = 20.0 * np.log10(np.maximum(spec_mat, 1e-5))

    norm = cv2.normalize(spec_db, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm = np.uint8(norm)
    norm = np.flipud(norm)  # High frequencies at the top

    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def segment_audio_temporal(
    samples: np.ndarray, sample_rate: int, window_sec: float = 3.0
) -> List[Dict[str, Any]]:
    """Divides audio into contiguous temporal evaluation chunks."""
    if len(samples) == 0 or sample_rate <= 0:
        return []

    win_len = int(window_sec * sample_rate)
    total_len = len(samples)
    total_dur = float(total_len) / sample_rate

    # If audio is shorter than window, evaluate entire clip as single window
    if total_len <= win_len:
        feats = compute_spectral_features(samples, sample_rate)
        label = "LIKELY AI-GENERATED" if feats["p_audio_ai"] >= 0.55 else "LIKELY REAL"
        return [{
            "start_seconds": 0.0,
            "end_seconds": round(total_dur, 2),
            "duration_seconds": round(total_dur, 2),
            "label": label,
            "ai_prob": feats["p_audio_ai"],
            "has_vocoder_cutoff": feats["has_vocoder_cutoff"],
        }]

    segments = []
    step = win_len // 2  # 50% overlap
    for start in range(0, total_len, step):
        chunk = samples[start : start + win_len]
        if len(chunk) < sample_rate // 2:  # skip chunks under 0.5s
            break
        t_start = float(start) / sample_rate
        t_end = min(total_dur, float(start + len(chunk)) / sample_rate)
        feats = compute_spectral_features(chunk, sample_rate)
        label = "LIKELY AI-GENERATED" if feats["p_audio_ai"] >= 0.55 else "LIKELY REAL"
        segments.append({
            "start_seconds": round(t_start, 2),
            "end_seconds": round(t_end, 2),
            "duration_seconds": round(t_end - t_start, 2),
            "label": label,
            "ai_prob": feats["p_audio_ai"],
            "has_vocoder_cutoff": feats["has_vocoder_cutoff"],
        })

    return segments
