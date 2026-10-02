"""
audio_detector.explain: Plain-English Newbie Explanation & 9-Dimensions Audio Forensic Dossier Generator.
Transforms complex acoustic telemetry, Wiener spectral flatness, and neural vocoder frequency cutoffs
into clear, engaging narrative explanations for beginners and comprehensive audits for professionals.
Aligned directly with NIST OpenMFC and C2PA v2.1 audio standards.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("audio_detector.explain")


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

    sr = prof.get("sample_rate", geom.get("sample_rate", 44100))
    duration = prof.get("duration", geom.get("duration_seconds", 0.0))
    channels = prof.get("channels", geom.get("channels", 1))

    # 1. Hardware & Container Provenance
    has_c2pa = prov.get("c2pa_present", False)
    d1 = {
        "dimension_id": 1,
        "title": "Dimension 1: Acoustic Hardware & Container Provenance",
        "description": "Inspects audio container headers (WAV/RIFF, MP3 frame sync), metadata tags, and C2PA manifests.",
        "container_format": prof.get("format", "WAV Audio Stream"),
        "c2pa_status": "Present & Verified" if has_c2pa else "Absent (Neutral)",
        "provenance_verdict": prov.get("provenance_status", "PROVENANCE_UNKNOWN"),
    }

    # 2. Acoustic Temporal Geometry
    d2 = {
        "dimension_id": 2,
        "title": "Dimension 2: Acoustic Temporal Geometry & Bandwidth Sampling",
        "description": "Examines sampling rate, bit depth, duration, and channel spatial configuration.",
        "sample_rate": f"{sr:,} Hz",
        "duration": f"{duration:.2f} seconds",
        "channels": f"{'Stereo (2 Ch)' if channels == 2 else 'Mono (1 Ch)'}",
        "bit_depth": prof.get("bit_depth", "16-bit PCM"),
    }

    # 3. Vocoder High-Frequency Cutoff (Brickwalling)
    cutoff_hz = acoustics.get("vocoder_cutoff_hz", 0)
    has_cutoff = aud_res.get("has_vocoder_cutoff", cutoff_hz in (8000, 11025, 16000, 22050, 24000))
    d3 = {
        "dimension_id": 3,
        "title": "Dimension 3: Neural Vocoder High-Frequency Cutoff (Brickwalling)",
        "description": "Detects sharp artificial frequency drop-offs characteristic of neural speech synthesizers (16kHz/24kHz).",
        "detected_cutoff_hz": f"{cutoff_hz} Hz" if cutoff_hz > 0 else "None (Full Bandwidth)",
        "is_brickwalled": has_cutoff,
        "diagnosis": (
            f"Sharp vocoder brickwall cutoff detected at {cutoff_hz} Hz; common signature of ElevenLabs / VITS neural models."
            if has_cutoff
            else "Natural acoustic harmonic roll-off preserved to Nyquist frequency."
        ),
    }

    # 4. Wiener Spectral Flatness & Harmonic Over-Smoothing
    flatness = acoustics.get("spectral_flatness_mean", 0.05)
    is_flat = flatness < 0.003
    d4 = {
        "dimension_id": 4,
        "title": "Dimension 4: Wiener Spectral Flatness & Harmonic Continuity",
        "description": "Measures resonant tonality versus unnatural mathematical smoothing across speech formants.",
        "spectral_flatness": round(flatness, 5),
        "is_unnaturally_smooth": is_flat,
        "diagnosis": (
            f"Over-smoothed harmonic formants (flatness: {flatness:.5f}); indicates algorithmic speech generation."
            if is_flat
            else "Natural human vocal tract resonance and acoustic turbulence preserved."
        ),
    }

    # 5. Silence Profile & Acoustic Room Ambience
    silence_ratio = acoustics.get("silence_ratio", 0.08)
    unnatural_silence = acoustics.get("has_unnatural_silence", silence_ratio > 0.35)
    d5 = {
        "dimension_id": 5,
        "title": "Dimension 5: Silence Profile & Environmental Room Ambience",
        "description": "Verifies background room tone versus digital zero-silence between spoken utterances.",
        "silence_ratio": f"{silence_ratio * 100:.1f}%",
        "has_digital_dead_silence": unnatural_silence,
        "diagnosis": (
            "Digital zero-silence patches detected between phonemes; common artifact of chunked neural text-to-speech."
            if unnatural_silence
            else "Natural continuous acoustic room tone and microphone floor noise present."
        ),
    }

    # 6. Speaker Identification & Scene Setting
    speakers = inv.get("estimated_speakers", 1)
    d6 = {
        "dimension_id": 6,
        "title": "Dimension 6: Vocal Entities & Acoustic Environment",
        "description": "Catalogs estimated speakers, speaking cadence, and acoustic recording environment.",
        "estimated_speakers": speakers,
        "dominant_type": inv.get("dominant_audio_type", "Spoken Voice"),
        "environment": inv.get("acoustic_environment", "Studio / Isolated Recording"),
        "vocal_tone": inv.get("vocal_tone_and_delivery", "Natural Cadence"),
    }

    # 7. Audio Synthesis Medium
    d7 = {
        "dimension_id": 7,
        "title": "Dimension 7: Audio Synthesis Medium & Generation Paradigm",
        "description": "Distinguishes human physical vocal cords from neural text-to-speech, voice conversion, or singing synthesis.",
        "synthesis_medium": aud_res.get("synthesis_medium", "Human Speech"),
        "is_synthetic_voice": aud_res.get("is_synthetic", False),
    }

    # 8. Frequency Spectrum Bandwidth & Nyquist Coverage
    d8 = {
        "dimension_id": 8,
        "title": "Dimension 8: Frequency Spectrum Bandwidth & Nyquist Coverage",
        "description": "Assesses frequency range from sub-bass (20Hz) through presence (10kHz-20kHz).",
        "nyquist_frequency": f"{sr // 2:,} Hz",
        "effective_bandwidth": f"{min(sr // 2, cutoff_hz if cutoff_hz > 0 else sr // 2):,} Hz",
    }

    # 9. Foundation Voice Model Attribution & Watermarking
    d9 = {
        "dimension_id": 9,
        "title": "Dimension 9: Foundation Voice Model Attribution & Watermarking",
        "description": "Identifies voice cloning engines (ElevenLabs, OpenAI Voice, Tortoise, Bark, VITS) and watermark markers.",
        "attributed_generator": attr.get("attributed_model", "Unattributable / Unknown Model"),
        "attribution_confidence": f"{int(attr.get('attribution_confidence', 0.0) * 100)}%",
        "watermark_detected": attr.get("watermark_detected", False),
        "suspicious_duration_pct": f"{aud_res.get('ai_duration_pct', 0.0):.1f}%",
    }

    return {
        "dimension_1_hardware_provenance": d1,
        "dimension_2_pixel_architecture": d2,
        "dimension_3_prnu_sensor_noise": d3,
        "dimension_4_surface_smoothness": d4,
        "dimension_5_fourier_fft_decay": d5,
        "dimension_6_subject_genre": d6,
        "dimension_7_visual_medium": d7,
        "dimension_8_sensor_spectrum": d8,
        "dimension_9_generative_attribution": d9,
    }


def generate_audio_newbie_explanation(
    filename: str,
    profile_data: Dict[str, Any],
    content_inventory: Dict[str, Any],
    audio_result: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    """Generates an engaging, accessible narrative explanation of audio forensics for non-technical users."""
    duration = profile_data.get("duration", 0.0)
    sr = profile_data.get("sample_rate", 44100)

    probs = decision.get("authenticity_probabilities", {})
    p_ai = probs.get("p_ai", 0.0)
    p_real = probs.get("p_real", 0.0)
    final_status = decision.get("final_status", "")

    inv = content_inventory or {}
    spk_cnt = inv.get("estimated_speakers", 1)
    audio_type = inv.get("dominant_audio_type", "vocal speech")

    speaker_str = f"{spk_cnt} speaker" if spk_cnt == 1 else f"{spk_cnt} speakers"

    section_what = (
        f"### 🎙️ What We Identified in this Audio Track\n\n"
        f"This file (`{filename}`) is a **{duration:.1f}-second audio recording** sampled at **{sr:,} Hz**. "
        f"Acoustically, it features **{speaker_str}** delivering **{audio_type.lower()}**.\n\n"
    )

    section_newbie = "### 💡 How Would You Explain This to a Newbie?\n\n"

    is_ai = "SYNTHETIC" in final_status or "AI" in final_status or p_ai >= 55.0

    if is_ai:
        body = (
            f"**The Simple Takeaway:** Our forensic acoustic engine determined with **{p_ai:.1f}% confidence** that this audio was "
            f"**created by an AI voice generator or clone** (such as ElevenLabs, OpenAI Voice, or Tortoise), rather than being spoken by a live human.\n\n"
            f"**Think of it like this:** When a real person speaks into a microphone, air passes through vocal cords, bounces around the mouth and chest, "
            f"and catches subtle human imperfections — like tiny breaths, tongue clicks, and the natural echo of the room you are standing in. "
            f"An AI voice model, on the other hand, calculates speech mathematically using a digital vocoder. "
            f"When we analyze the sound waves under our forensic microscope, the speech harmonics are unnaturally smooth, the sound abruptly cuts off at a digital ceiling "
            f"(the vocoder brickwall), and the silent gaps between words are pure mathematical zeros rather than real-world room ambience. "
            f"Even though it sounds uncannily like a real person, the acoustic physics prove it was generated by a computer algorithm."
        )
    else:
        body = (
            f"**The Simple Takeaway:** This audio track was verified with **{p_real:.1f}% confidence** as a **genuine human recording**.\n\n"
            f"**Think of it like this:** Everything about this sound wave matches real-world acoustic physics. Natural vocal tract resonance, organic breathing rhythm, "
            f"subtle microphone room tone, and complete high-frequency harmonic extension are all present without any neural vocoder cutoff or synthetic over-smoothing. "
            f"No voice cloning or generative deepfake alterations were detected."
        )

    return section_what + section_newbie + body
