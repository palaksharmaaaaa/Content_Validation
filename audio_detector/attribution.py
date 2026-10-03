"""
audio_detector.attribution: Generative AI Voice & Audio Synthesizer Attribution Engine.
Profiles audio characteristics against major speech and music synthesis models:
- ElevenLabs (Multilingual v1/v2, Flash)
- Alibaba CosyVoice (1.0 / 2.0)
- 2Noise ChatTTS
- Suno AI (v3 / v3.5)
- Udio Music Generation
- OpenAI (Voice Engine / Realtime API / Advanced Voice Mode)
- Google Gemini native audio / Lyria
- Meta AudioCraft (MusicGen / Voicebox)
- Hume AI (Octave)
- Cartesia (Sonic)
- PlayHT
- Resemble AI
- Coqui XTTS-v2 (open source)
Completely self-contained with zero outside dependencies.

NOTE on scoring confidence per entry: only ElevenLabs, CosyVoice, ChatTTS, Suno, Udio, and
OpenAI have dedicated spectral/vocoder-cutoff signatures below (calibrated against observed
acoustic characteristics). The 2025/2026 additions (Gemini/Lyria, AudioCraft, Hume, Cartesia,
PlayHT, Resemble, Coqui) are currently metadata-signature-only -- matched when a tool name
literally appears in file metadata, with no acoustic/spectral fingerprint yet. Do not assume
these can be identified from audio content alone the way the original six can; add real
spectral calibration for them only once backed by actual sample analysis, not guessed numbers.
"""
from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.shared_results import unknown_attribution

logger = logging.getLogger("audio_detector.attribution")

KNOWN_AUDIO_GENERATORS = {
    "elevenlabs": {
        "name": "ElevenLabs Speech Synthesis",
        "provider": "ElevenLabs",
        "telltales": ["Characteristic 7.5kHz / 16kHz brick-wall vocoder termination", "ultrasmooth formant transitions"],
    },
    "cosyvoice": {
        "name": "Alibaba CosyVoice (1.0 / 2.0)",
        "provider": "Alibaba Tongyi Lab",
        "telltales": ["Flow-matching diffusion vocoder harmonics", "high-frequency roll-off at 8kHz or 12kHz"],
    },
    "chat_tts": {
        "name": "2Noise ChatTTS",
        "provider": "Open Source",
        "telltales": ["Conversational pause markers", "mild vocoder phase distortion in sibilants"],
    },
    "suno_ai": {
        "name": "Suno AI (v3 / v3.5)",
        "provider": "Suno Inc.",
        "telltales": ["Synthetic multi-instrument stem compression", "characteristic mid-frequency phase swirl"],
    },
    "udio": {
        "name": "Udio Music Generation",
        "provider": "Uncharted Labs",
        "telltales": ["High dynamic range synthetic diffusion music stems", "vocal formant stereo artifacting"],
    },
    "openai_voice": {
        "name": "OpenAI Voice Engine / Realtime API / Advanced Voice Mode",
        "provider": "OpenAI",
        "telltales": ["Consistent natural room tone emulation", "smooth pitch intonation curve"],
    },
    "google_gemini_audio": {
        "name": "Google Gemini Native Audio / Lyria",
        "provider": "Google DeepMind",
        "telltales": ["Metadata-signature match only (no spectral fingerprint calibrated yet)"],
    },
    "meta_audiocraft": {
        "name": "Meta AudioCraft (MusicGen / Voicebox)",
        "provider": "Meta AI",
        "telltales": ["Metadata-signature match only (no spectral fingerprint calibrated yet)"],
    },
    "hume_ai": {
        "name": "Hume AI (Octave)",
        "provider": "Hume AI",
        "telltales": ["Metadata-signature match only (no spectral fingerprint calibrated yet)"],
    },
    "cartesia": {
        "name": "Cartesia (Sonic)",
        "provider": "Cartesia",
        "telltales": ["Metadata-signature match only (no spectral fingerprint calibrated yet)"],
    },
    "playht": {
        "name": "PlayHT",
        "provider": "Play.ht",
        "telltales": ["Metadata-signature match only (no spectral fingerprint calibrated yet)"],
    },
    "resemble_ai": {
        "name": "Resemble AI",
        "provider": "Resemble AI",
        "telltales": ["Metadata-signature match only (no spectral fingerprint calibrated yet)"],
    },
    "coqui_xtts": {
        "name": "Coqui XTTS-v2",
        "provider": "Open Source",
        "telltales": ["Metadata-signature match only (no spectral fingerprint calibrated yet)"],
    },
}


def _word(token: str, text: str) -> bool:
    """Whole-word match, so short vendor names don't fire inside unrelated words (\"udio\" in \"audio\", \"sonic\" in \"supersonic\")."""
    return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None


class AudioModelAttributionEngine:
    """Attributes synthetic voice clones and AI audio to specific foundation models."""

    def __init__(self):
        pass

    def attribute_audio(
        self,
        audio_path: str | Path,
        acoustic_data: Optional[Dict[str, Any]] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        provenance_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attributes audio to likely generative synthesizer."""
        path = Path(audio_path)
        if not path.is_file():
            return self._unknown_attribution("File not found")

        scores: Dict[str, float] = {k: 0.05 for k in KNOWN_AUDIO_GENERATORS}
        cues: List[str] = []

        # 1. Metadata / Provenance inspection
        if provenance_data:
            meta = provenance_data.get("metadata", {})
            for k, v in meta.items():
                val_str = str(v).lower()
                if "elevenlabs" in val_str:
                    scores["elevenlabs"] += 0.85
                    cues.append("Metadata declares ElevenLabs audio generation")
                elif _word("suno", val_str):
                    scores["suno_ai"] += 0.85
                    cues.append("Metadata declares Suno AI generation")
                elif _word("udio", val_str):
                    scores["udio"] += 0.85
                    cues.append("Metadata declares Udio generation")
                elif "lyria" in val_str or ("gemini" in val_str and "audio" in val_str):
                    scores["google_gemini_audio"] += 0.85
                    cues.append("Metadata declares Google Gemini/Lyria audio generation")
                elif "audiocraft" in val_str or "musicgen" in val_str or "voicebox" in val_str:
                    scores["meta_audiocraft"] += 0.85
                    cues.append("Metadata declares Meta AudioCraft generation")
                elif _word("hume", val_str):
                    scores["hume_ai"] += 0.85
                    cues.append("Metadata declares Hume AI (Octave) generation")
                elif "cartesia" in val_str or _word("sonic", val_str):
                    scores["cartesia"] += 0.85
                    cues.append("Metadata declares Cartesia (Sonic) generation")
                elif "play.ht" in val_str or "playht" in val_str:
                    scores["playht"] += 0.85
                    cues.append("Metadata declares PlayHT generation")
                elif "resemble" in val_str:
                    scores["resemble_ai"] += 0.85
                    cues.append("Metadata declares Resemble AI generation")
                elif "coqui" in val_str or "xtts" in val_str:
                    scores["coqui_xtts"] += 0.85
                    cues.append("Metadata declares Coqui XTTS generation")
                elif "realtime api" in val_str or "advanced voice mode" in val_str or ("gpt-4o" in val_str and "voice" in val_str):
                    scores["openai_voice"] += 0.85
                    cues.append("Metadata declares OpenAI Realtime API / Advanced Voice Mode generation")

        # 2. Spectral & Vocoder Cutoff signatures
        if acoustic_data:
            spec = acoustic_data.get("spectral_features", acoustic_data)
            has_cutoff = spec.get("has_vocoder_cutoff", False)
            cutoff_freq = float(spec.get("cutoff_freq_hz", 0.0))
            flatness = float(spec.get("spectral_flatness", 0.05))
            silence = float(spec.get("digital_silence_ratio", 0.0))

            if has_cutoff:
                if 7200 <= cutoff_freq <= 8200:
                    scores["elevenlabs"] += 0.35
                    scores["cosyvoice"] += 0.30
                    cues.append(f"Sharp brick-wall vocoder cutoff at {cutoff_freq:.0f} Hz (typical of ElevenLabs / CosyVoice)")
                elif 15000 <= cutoff_freq <= 16500:
                    scores["elevenlabs"] += 0.40
                    cues.append(f"16kHz high-tier vocoder boundary detected ({cutoff_freq:.0f} Hz)")

            if flatness < 0.002:
                scores["elevenlabs"] += 0.20
                scores["openai_voice"] += 0.20
                cues.append("Hyper-regular Wiener spectral flatness matching modern neural vocoders")

            if silence > 0.12:
                scores["elevenlabs"] += 0.15
                scores["chat_tts"] += 0.25
                cues.append("Digital zero inter-phoneme splicing gaps isolated")

        # Normalize candidate scores
        total = sum(scores.values())
        norm = {k: v / total for k, v in scores.items()} if total > 0 else scores
        top_candidates = sorted(norm.items(), key=lambda x: x[1], reverse=True)

        best_key, best_score = top_candidates[0]
        if best_score < 0.25:
            return {
                "attributed_model": "Unknown / Generic Neural Vocoder",
                "model_key": "unknown",
                "confidence": round(best_score, 2),
                "cues": cues or ["No distinctive audio synthesizer fingerprints isolated."],
                "top_candidates": [{"model": KNOWN_AUDIO_GENERATORS[k]["name"], "confidence": round(s, 2)} for k, s in top_candidates[:3]],
            }

        best_info = KNOWN_AUDIO_GENERATORS[best_key]
        conf = round(best_score, 2)
        region = (
            "China" if best_key in ("cosyvoice", "chat_tts")
            else "United States" if best_key in (
                "elevenlabs", "suno_ai", "suno", "udio", "openai_voice",
                "google_gemini_audio", "meta_audiocraft", "hume_ai", "cartesia",
                "playht", "resemble_ai",
            )
            else "Global"
        )
        return {
            "attributed_model": best_info["name"],
            "model_key": best_key,
            "provider": best_info["provider"],
            "confidence": conf,
            "attribution_confidence": conf,
            "region_of_origin": region,
            "watermark_detected": False,
            "cues": cues or [f"Acoustic vocoder profile matches {best_info['name']}"],
            "top_candidates": [{"model": KNOWN_AUDIO_GENERATORS[k]["name"], "confidence": round(s, 2)} for k, s in top_candidates[:3]],
        }

    def attribute_media(
        self,
        media_path: str | Path,
        modality: str = "audio",
        forensic_data: Optional[Dict[str, Any]] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        provenance_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Unified attribution interface for multi-modal workflows."""
        return self.attribute_audio(
            audio_path=media_path,
            acoustic_data=forensic_data,
            profile_data=profile_data,
            provenance_data=provenance_data,
        )

    def _unknown_attribution(self, reason: str) -> Dict[str, Any]:
        return unknown_attribution(reason)
