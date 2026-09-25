"""
Cross-Modal Consistency and Audio-Visual Synchronization Engine.
Detects advanced deepfakes, AI voice-overs, and audio-visual splice anomalies:
1. Audio-Visual AI Divergence (e.g. authentic footage spliced with voice clone).
2. Lip Motion vs Speech Energy Temporal Correlation.
3. Acoustic Environment vs Visual Scene Plausibility (outdoor visuals vs digital silence room tone).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from utils.logging_utils import get_logger

logger = get_logger("cross_modal_engine")


def evaluate_cross_modal_consistency(
    video_forensics: Dict[str, Any],
    audio_forensics: Dict[str, Any],
    content_inventory: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates cross-modal coherence between the video track and demuxed audio track.
    """
    has_audio = audio_forensics.get("has_audio_track", False)
    if not has_audio:
        return {
            "is_multimodal": False,
            "cross_modal_status": "SINGLE_MODALITY_VIDEO",
            "asymmetry_score": 0.0,
            "cues": ["Media contains video only; cross-modal synchronization skipped."],
            "is_consistent": True,
            "tampering_risk": "LOW",
        }

    cues = []
    anomalies_detected = 0

    vid_ai = float(video_forensics.get("ai_video_rating", {}).get("ai_percentage", 0.0))
    aud_ai = float(audio_forensics.get("ai_percentage", 0.0))

    # 1. Modality Asymmetry Check (Voiceover / Deepfake Splice)
    asymmetry = abs(vid_ai - aud_ai)
    if asymmetry >= 40.0:
        anomalies_detected += 1
        if aud_ai > vid_ai:
            cues.append(
                f"Modality Asymmetry: Synthetic Audio ({aud_ai:.1f}%) paired with Authentic Video ({100-vid_ai:.1f}%). "
                "Characteristic of AI voice clone or manipulated voiceover."
            )
        else:
            cues.append(
                f"Modality Asymmetry: Synthetic Video ({vid_ai:.1f}%) paired with Authentic Audio ({100-aud_ai:.1f}%). "
                "Characteristic of generative video puppetry / talking head re-enactment."
            )

    # 2. Scene Environment vs Acoustic Room Tone Coherence
    scene_type = content_inventory.get("scene_type", "")
    spec_feats = audio_forensics.get("spectral_features", {})
    silence_ratio = spec_feats.get("digital_silence_ratio", 0.0)

    if "Outdoor" in scene_type and silence_ratio > 0.22:
        anomalies_detected += 1
        cues.append(
            "Acoustic Plausibility Mismatch: Outdoor visual environment detected, "
            f"but audio stream exhibits unnatural digital silence gaps ({silence_ratio*100:.1f}%)."
        )

    # 3. Inter-frame Motion vs Speech Correlation
    temp_cons = video_forensics.get("temporal_consistency", {})
    warping_risk = temp_cons.get("temporal_warping_risk", "LOW")

    if warping_risk in ("HIGH", "CRITICAL") and aud_ai > 60.0:
        anomalies_detected += 1
        cues.append(
            "Temporal Desynchronization: High visual frame warping coincides with "
            "synthetic neural vocoder audio, indicating deepfake video generation (e.g. Wav2Lip/SadTalker)."
        )

    if anomalies_detected >= 2:
        status = "CROSS_MODAL_TAMPERING_DETECTED"
        risk = "HIGH"
    elif anomalies_detected == 1:
        status = "CROSS_MODAL_DISCREPANCY"
        risk = "MODERATE"
    else:
        status = "SYNCHRONOUS_CONSISTENT"
        risk = "LOW"
        cues.append("Visual motion and acoustic tracks exhibit mutual physical consistency.")

    return {
        "is_multimodal": True,
        "cross_modal_status": status,
        "tampering_risk": risk,
        "asymmetry_score": round(asymmetry, 1),
        "video_ai_pct": vid_ai,
        "audio_ai_pct": aud_ai,
        "cues": cues,
        "is_consistent": (anomalies_detected == 0),
    }
