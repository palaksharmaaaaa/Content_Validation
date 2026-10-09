"""
audio_detector.explain: Plain-English Newbie Explanation & 9-Dimensions Audio Forensic Dossier Generator.
Transforms complex acoustic telemetry, Wiener spectral flatness, and neural vocoder frequency cutoffs
into clear, engaging narrative explanations for beginners and comprehensive audits for professionals.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, NamedTuple

from audio_detector.config import DIGITAL_SILENCE_RATIO_THRESHOLD, SYNTHETIC_FLATNESS_HIGH_THRESHOLD, SYNTHETIC_FLATNESS_LOW_THRESHOLD

logger = logging.getLogger("audio_detector.explain")


class _AudioDossierContext(NamedTuple):
    """Values derived once from the inputs and shared by the per-dimension builders."""

    prof: Any
    geom: Any
    aud_res: Any
    prov: Any
    attr: Any
    inv: Any
    acoustics: Any
    sr: Any
    duration: Any
    channels: Any
    cutoff_hz: Any


def _channels_text(channels: Any) -> str:
    """Channel layout from the file's own header, or 'Not recorded' when it could not be read."""
    if channels == 1:
        return "Mono (1 channel)"
    if channels == 2:
        return "Stereo (2 channels)"
    return f"{channels} channels" if isinstance(channels, int) and channels > 2 else "Not recorded"


def _audio_dimension_1(c: _AudioDossierContext) -> Dict[str, Any]:
    """1. Hardware & Container Provenance"""
    has_c2pa = c.prov.get("c2pa_present", False)
    d1 = {
        "dimension_id": 1,
        "title": "Dimension 1: Acoustic Hardware & Container Provenance",
        "description": "Inspects audio container headers (WAV/RIFF, MP3 frame sync), metadata tags, and C2PA manifests.",
        "container_format": c.prof.get("format") or "Not recorded",
        "c2pa_status": "Present (markers only; not cryptographically verified)" if has_c2pa else "Absent (Neutral)",
        "provenance_verdict": c.prov.get("provenance_status", "PROVENANCE_UNKNOWN"),
    }
    return d1


def _audio_dimension_2(c: _AudioDossierContext) -> Dict[str, Any]:
    """2. Acoustic Temporal Geometry"""
    d2 = {
        "dimension_id": 2,
        "title": "Dimension 2: Acoustic Temporal Geometry & Bandwidth Sampling",
        "description": "Examines sampling rate, bit depth, duration, and channel spatial configuration.",
        "sample_rate": f"{c.sr:,} Hz" if c.sr else "Not recorded",
        "duration": f"{c.duration:.2f} seconds",
        "channels": _channels_text(c.channels),
        "bit_depth": c.prof.get("bit_depth") or "Not recorded",
    }
    return d2


def _measured(c: _AudioDossierContext) -> bool:
    return bool(c.acoustics) and c.acoustics.get("measured", True) is not False


def _audio_dimension_3(c: _AudioDossierContext) -> Dict[str, Any]:
    """3. Vocoder High-Frequency Cutoff (Brickwalling)"""
    d3 = {
        "dimension_id": 3,
        "title": "Dimension 3: Neural Vocoder High-Frequency Cutoff (Brickwalling)",
        "description": "Finds the frequency below which 98.5 % of the energy lies and flags a cutoff in the bands typical of neural speech synthesizers.",
    }
    if not _measured(c):
        d3.update(detected_cutoff_hz=None, is_brickwalled=None, diagnosis="Not measured: the recording is too short to analyse.")
        return d3
    has_cutoff = bool(c.acoustics.get("has_vocoder_cutoff"))
    d3.update(
        detected_cutoff_hz=f"{c.cutoff_hz:.0f} Hz",
        is_brickwalled=has_cutoff,
        diagnosis=(
            f"Energy stops sharply at {c.cutoff_hz:.0f} Hz, a band where many neural vocoders cut off; low-bitrate and telephone audio do the same."
            if has_cutoff
            else f"No vocoder-style cutoff: 98.5 % of the energy lies below {c.cutoff_hz:.0f} Hz."
        ),
    )
    return d3


def _audio_dimension_4(c: _AudioDossierContext) -> Dict[str, Any]:
    """4. Wiener Spectral Flatness & Harmonic Over-Smoothing"""
    d4 = {
        "dimension_id": 4,
        "title": "Dimension 4: Wiener Spectral Flatness & Harmonic Continuity",
        "description": "Measures how noise-like (flat) the spectrum is; extremely low values mean an unnaturally pure, smooth signal.",
    }
    if not _measured(c):
        d4.update(spectral_flatness=None, is_unnaturally_smooth=None, diagnosis="Not measured: the recording is too short to analyse.")
        return d4
    flatness = float(c.acoustics.get("spectral_flatness", 0.0))
    is_flat = flatness < SYNTHETIC_FLATNESS_LOW_THRESHOLD
    d4.update(
        spectral_flatness=round(flatness, 5),
        is_unnaturally_smooth=is_flat,
        diagnosis=(
            f"Spectrum is extremely smooth (flatness {flatness:.5f}); typical of algorithmic generation, also of pure test tones."
            if is_flat
            else f"Spectrum is almost flat (flatness {flatness:.5f}), like white noise; the detector counts this as a synthetic cue."
            if flatness > SYNTHETIC_FLATNESS_HIGH_THRESHOLD
            else f"Spectral flatness {flatness:.5f} is between the synthetic extremes."
        ),
    )
    return d4


def _audio_dimension_5(c: _AudioDossierContext) -> Dict[str, Any]:
    """5. Silence Profile & Acoustic Room Ambience"""
    d5 = {
        "dimension_id": 5,
        "title": "Dimension 5: Silence Profile & Environmental Room Ambience",
        "description": "Measures the share of samples that are exactly zero (digital silence), as opposed to the low noise of a real room.",
    }
    if not _measured(c):
        d5.update(silence_ratio=None, has_digital_dead_silence=None, diagnosis="Not measured: the recording is too short to analyse.")
        return d5
    ratio = float(c.acoustics.get("digital_silence_ratio", 0.0))
    dead = ratio > DIGITAL_SILENCE_RATIO_THRESHOLD
    d5.update(
        silence_ratio=f"{ratio * 100:.1f}%",
        has_digital_dead_silence=dead,
        diagnosis=(
            "Long stretches of exact digital silence; common in chunked text-to-speech, but also in edited recordings."
            if dead
            else "No notable digital silence."
        ),
    )
    return d5


def _audio_dimension_6(c: _AudioDossierContext) -> Dict[str, Any]:
    """6. Sound type and delivery (what the recording is, as far as a spectrum and a level can say)"""
    d6 = {
        "dimension_id": 6,
        "title": "Dimension 6: Sound Type & Delivery",
        "description": "Classifies the whole recording by where its energy lies (voice band, high frequencies) and describes its dynamics and level. It does not count speakers or identify rooms.",
        "dominant_type": c.inv.get("dominant_modality") or "Not determined",
        "vocal_tone": c.inv.get("delivery_style") or "Not determined",
        "signal_level": c.inv.get("signal_level") or "Not determined",
    }
    return d6


def _audio_dimension_7(c: _AudioDossierContext) -> Dict[str, Any]:
    """7. Audio Synthesis Medium"""
    d7 = {
        "dimension_id": 7,
        "title": "Dimension 7: Audio Synthesis Medium & Generation Paradigm",
        "description": "Distinguishes human physical vocal cords from neural text-to-speech, voice conversion, or singing synthesis.",
        "synthesis_medium": c.aud_res.get("synthesis_medium") or "Not determined",
        "is_synthetic_voice": c.aud_res.get("is_synthetic", False),
    }
    return d7


def _audio_dimension_8(c: _AudioDossierContext) -> Dict[str, Any]:
    """8. Frequency Spectrum Bandwidth & Nyquist Coverage"""
    d8 = {
        "dimension_id": 8,
        "title": "Dimension 8: Frequency Spectrum Bandwidth & Nyquist Coverage",
        "description": "Assesses frequency range from sub-bass (20Hz) through presence (10kHz-20kHz).",
        "nyquist_frequency": f"{c.sr // 2:,} Hz" if c.sr else "Not recorded",
        "effective_bandwidth": f"{min(c.sr // 2, c.cutoff_hz if c.cutoff_hz > 0 else c.sr // 2):,.0f} Hz" if c.sr else "Not recorded",
    }
    return d8


def _audio_dimension_9(c: _AudioDossierContext) -> Dict[str, Any]:
    """9. Foundation Voice Model Attribution & Watermarking"""
    d9 = {
        "dimension_id": 9,
        "title": "Dimension 9: Foundation Voice Model Attribution & Watermarking",
        "description": "Names a voice-generation tool only when the file's own metadata declares one; otherwise reports none.",
        "attributed_generator": c.attr.get("attributed_model") or "Not attributable",
        "attribution_confidence": f"{int(c.attr.get('attribution_confidence', 0.0) * 100)}%",
        "watermark_detected": c.attr.get("watermark_detected", False),
        "suspicious_duration_pct": f"{c.aud_res.get('ai_duration_pct', 0.0):.1f}%",
    }
    return d9


def build_audio_nine_dimensions_dossier(
    profile_data: Dict[str, Any],
    audio_result: Dict[str, Any],
    content_inventory: Optional[Dict[str, Any]] = None,
    provenance_result: Optional[Dict[str, Any]] = None,
    attribution_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Constructs an exhaustive 9-dimensional forensic analysis dossier for audio recordings."""
    prof = profile_data or {}
    geom = prof.get("geometry", {})
    aud_res = audio_result or {}
    prov = provenance_result or {}
    attr = attribution_result or {}
    inv = content_inventory or {}
    acoustics = aud_res.get("acoustic_features", {})

    # The file's own sample rate, channel count and duration (the profile's "sample_rate" is the 16 kHz the engine decodes to).
    sr = prof.get("native_sample_rate") or geom.get("sample_rate")
    duration = prof.get("duration_seconds") or geom.get("duration_seconds") or 0.0
    channels = prof.get("channels") or geom.get("channels")

    cutoff_hz = float(acoustics.get("cutoff_freq_hz", 0.0))
    c = _AudioDossierContext(prof=prof, geom=geom, aud_res=aud_res, prov=prov, attr=attr, inv=inv, acoustics=acoustics, sr=sr, duration=duration, channels=channels, cutoff_hz=cutoff_hz)
    return {
        "dimension_1": _audio_dimension_1(c),
        "dimension_2": _audio_dimension_2(c),
        "dimension_3": _audio_dimension_3(c),
        "dimension_4": _audio_dimension_4(c),
        "dimension_5": _audio_dimension_5(c),
        "dimension_6": _audio_dimension_6(c),
        "dimension_7": _audio_dimension_7(c),
        "dimension_8": _audio_dimension_8(c),
        "dimension_9": _audio_dimension_9(c),
    }


def generate_audio_newbie_explanation(
    filename: str,
    profile_data: Dict[str, Any],
    content_inventory: Dict[str, Any],
    audio_result: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    """Generates an engaging, accessible narrative explanation of audio forensics for non-technical users."""
    duration = profile_data.get("duration_seconds") or 0.0
    sr = profile_data.get("native_sample_rate")

    probs = decision.get("authenticity_probabilities", {})
    p_ai = probs.get("p_ai", 0.0)
    p_real = probs.get("p_real", 0.0)
    final_status = decision.get("final_status", "")

    inv = content_inventory or {}
    audio_type = inv.get("dominant_modality")
    delivery = inv.get("delivery_style")
    known = [x for x in (audio_type, delivery) if x and x != "Unknown"]
    content_phrase = f"Acoustically, it is **{'; '.join(x[0].lower() + x[1:] for x in known)}**." if known else ""

    section_what = (
        f"### 🎙️ What We Identified in this Audio Track\n\n"
        f"This file (`{filename}`) is a **{duration:.1f}-second audio recording**" + (f" sampled at **{sr:,} Hz**" if sr else "") + ". "
        + content_phrase + "\n\n"
    )

    section_newbie = "### 💡 How Would You Explain This to a Newbie?\n\n"

    cues = [str(c) for c in (audio_result.get("forensic_cues") or []) if c][:4]
    flatness, silence = audio_result.get("spectral_flatness"), audio_result.get("digital_silence_ratio")
    measured = []
    if audio_result.get("cutoff_freq_hz"):
        measured.append(f"the spectrum {'stops sharply' if audio_result.get('has_vocoder_cutoff') else 'falls off naturally'} near **{float(audio_result['cutoff_freq_hz']):,.0f} Hz**")
    if flatness is not None:
        measured.append(f"spectral flatness **{float(flatness):.4f}**")
    if silence is not None:
        measured.append(f"**{float(silence) * 100:.1f}%** of the samples are exact digital zeros")
    in_this_file = ("In this file " + ", ".join(measured) + ".") if measured else "No spectral measurement was available for this file."
    cue_text = (" The cues raised were: " + "; ".join(cues) + ".") if cues else ""

    if final_status == "BLANK_OR_DEGRADED":
        body = "**The Simple Takeaway:** The recording is silent or has no usable signal, so no verdict is given."
    elif final_status == "UNDETERMINED":
        body = (
            f"**The Simple Takeaway:** The evidence does not settle it (heuristic, uncalibrated AI-likelihood **{p_ai:.1f}%**). "
            f"{in_this_file} Treat this as inconclusive."
        )
    elif "SYNTHETIC" in final_status or p_ai >= 55.0:
        body = (
            f"**The Simple Takeaway:** Our heuristic (uncalibrated) score is **{p_ai:.1f}% AI-likelihood**, so this audio leans towards a "
            f"**synthetic or cloned voice**. That is a ranking aid, not proof.\n\n"
            f"**What that is based on:** {in_this_file}{cue_text} Voice generators tend to produce a very smooth spectrum, a sharp ceiling "
            f"in frequency and silences that are exact zeros instead of room noise; low-bitrate encoders, phone audio and noise gates can leave the same marks."
        )
    else:
        body = (
            f"**The Simple Takeaway:** This audio is **consistent with a recording of a live source** (heuristic, uncalibrated estimate **{p_real:.1f}%**; "
            f"a ranking aid, not proof).\n\n"
            f"**What that is based on:** {in_this_file} No clear sign of a synthetic voice was found, but high-quality voice clones can pass these checks."
        )

    return section_what + section_newbie + body
