"""
audio_detector.attribution: Generative AI Voice & Audio Synthesizer Attribution Engine.
Profiles audio characteristics against major speech and music synthesis models:
- ElevenLabs (Multilingual v1/v2, Flash)
- Alibaba CosyVoice (1.0 / 2.0)
- 2Noise ChatTTS
- Suno AI (v3 / v3.5)
- Udio Music Generation
- OpenAI Voice Engine
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "name": "OpenAI Voice Engine",
        "provider": "OpenAI",
        "telltales": ["Consistent natural room tone emulation", "smooth pitch intonation curve"],
    },
}


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
                elif "suno" in val_str:
                    scores["suno_ai"] += 0.85
                    cues.append("Metadata declares Suno AI generation")
                elif "udio" in val_str:
                    scores["udio"] += 0.85
                    cues.append("Metadata declares Udio generation")

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
        region = "China" if best_key in ("cosyvoice", "chat_tts") else ("United States" if best_key in ("elevenlabs", "suno_ai", "suno", "udio", "openai_voice") else "Global")
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
        return {
            "attributed_model": "Unknown",
            "model_key": "unknown",
            "confidence": 0.0,
            "attribution_confidence": 0.0,
            "region_of_origin": "Unknown",
            "watermark_detected": False,
            "cues": [reason],
            "top_candidates": [],
        }
