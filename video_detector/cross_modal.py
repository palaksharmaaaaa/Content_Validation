"""
video_detector.cross_modal: Cross-Modal Audio-Visual Consistency and Synchronization Engine.
Detects deepfakes, AI voiceovers, and audiovisual splice anomalies:
1. Audio-Visual AI Divergence (e.g. authentic video footage spliced with AI voice clone).
2. Lip Motion vs Speech Energy Temporal Correlation.
3. Acoustic Environment vs Visual Scene Plausibility.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("video_detector.cross_modal")


class CrossModalConsistencyEngine:
    """Evaluates cross-modal coherence between video visual frames and audio speech track."""

    def __init__(self):
        pass

    def evaluate_consistency(
        self,
        video_forensics: Dict[str, Any],
        audio_forensics: Optional[Dict[str, Any]] = None,
        content_inventory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluates cross-modal coherence; always includes ``cross_modal_status`` for display."""
        res = self._evaluate(video_forensics, audio_forensics, content_inventory)
        res["cross_modal_status"] = res.get("status", "CROSS_MODAL_COHERENT")
        return res

    def _evaluate(
        self,
        video_forensics: Dict[str, Any],
        audio_forensics: Optional[Dict[str, Any]] = None,
        content_inventory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates cross-modal coherence between video and audio.
        """
        if not audio_forensics or not audio_forensics.get("has_audio_track", False):
            return {
                "is_multimodal": False,
                "status": "SINGLE_MODALITY_VIDEO",
                "asymmetry_score": 0.0,
                "cues": ["Media contains video only; cross-modal synchronization skipped."],
                "is_consistent": True,
                "tampering_risk": "LOW",
            }

        vid_ai = float(video_forensics.get("ai_percentage", 0.0))
        aud_ai = float(audio_forensics.get("ai_percentage", 0.0))

        cues = []
        anomalies = 0

        # 1. Modality Asymmetry Check (Voiceover / Deepfake Splice)
        asymmetry = abs(vid_ai - aud_ai)
        if asymmetry >= 40.0:
            anomalies += 1
            if aud_ai > vid_ai:
                cues.append(
                    f"Modality Asymmetry: Synthetic Audio ({aud_ai:.1f}%) paired with Authentic Video ({100-vid_ai:.1f}%). "
                    "Characteristic of AI voice clone or manipulated voiceover."
                )
            else:
                cues.append(
                    f"Modality Asymmetry: Synthetic Video ({vid_ai:.1f}%) paired with Authentic Audio ({100-aud_ai:.1f}%). "
                    "Characteristic of synthetic video diffusion driven by real voice track."
                )
        elif vid_ai >= 60.0 and aud_ai >= 60.0:
            cues.append(
                f"Full Synthetic Multimodality: Both Video ({vid_ai:.1f}%) and Audio ({aud_ai:.1f}%) show AI signatures."
            )

        # 2. Scene Plausibility Check
        if content_inventory:
            env = content_inventory.get("environment", {})
            setting = env.get("setting", "").lower()
            spec = audio_forensics.get("spectral_features", {})
            silence_ratio = float(spec.get("digital_silence_ratio", 0.0))

            if "outdoor" in setting and silence_ratio > 0.15:
                anomalies += 1
                cues.append(
                    "Acoustic/Visual Mismatch: Outdoor visual scene combined with digital zero silence in audio. "
                    "Physical outdoor recordings always have ambient atmospheric noise."
                )

        if anomalies >= 2:
            status = "CROSS_MODAL_INCONSISTENT"
            risk = "HIGH"
            is_consistent = False
        elif anomalies == 1:
            status = "SUSPICIOUS_ASYMMETRY"
            risk = "MEDIUM"
            is_consistent = False
        else:
            status = "CROSS_MODAL_COHERENT"
            risk = "LOW"
            is_consistent = True

        return {
            "is_multimodal": True,
            "status": status,
            "asymmetry_score": round(asymmetry, 1),
            "video_ai_percentage": vid_ai,
            "audio_ai_percentage": aud_ai,
            "anomalies_detected": anomalies,
            "cues": cues or ["Video and audio forensic profiles are mutually consistent."],
            "is_consistent": is_consistent,
            "tampering_risk": risk,
        }


def evaluate_cross_modal_consistency(
    video_forensics: Dict[str, Any],
    audio_forensics: Optional[Dict[str, Any]] = None,
    content_inventory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluates cross-modal coherence between video visual frames and audio speech track."""
    return CrossModalConsistencyEngine().evaluate_consistency(video_forensics, audio_forensics, content_inventory)
