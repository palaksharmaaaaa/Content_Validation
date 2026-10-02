"""
audio_detector.pipeline: End-to-End Audio Forensic Analysis Orchestrator.
Executes the comprehensive NIST-aligned forensic inspection:
1. Audio Stream & Codec Validation.
2. Signal Profiling & Dynamic Range / Crest Factor Analysis.
3. Cryptographic Provenance & ID3/RIFF Chunk Inspection.
4. Vocoder Cutoff, Spectral Flatness & Digital Silence Gap AI Detection.
5. Neural Acoustic Classifier Inference.
6. Scene, Setting & Vocal Delivery Tone Intelligence.
7. Generative Voice & Music Synthesizer Attribution.
8. Calibrated Multi-Modal Decision & Audit Trail Generation.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from audio_detector.attribution import AudioModelAttributionEngine
from audio_detector.content import AudioContentAnalyzer
from audio_detector.detector import AudioAIDetector
from audio_detector.explain import build_audio_nine_dimensions_dossier, generate_audio_newbie_explanation
from audio_detector.profiler import AudioProfiler
from audio_detector.provenance import AudioProvenanceValidator
from audio_detector.validator import AudioValidator

logger = logging.getLogger("audio_detector.pipeline")


class AudioForensicPipeline:
    """Unified end-to-end forensic analysis pipeline for audio recordings."""

    def __init__(
        self,
        detector: Optional[AudioAIDetector] = None,
        validator: Optional[AudioValidator] = None,
        profiler: Optional[AudioProfiler] = None,
        provenance_validator: Optional[AudioProvenanceValidator] = None,
        content_analyzer: Optional[AudioContentAnalyzer] = None,
        attribution_engine: Optional[AudioModelAttributionEngine] = None,
    ):
        self.detector = detector or AudioAIDetector()
        self.detector.load()
        self.validator = validator or AudioValidator()
        self.profiler = profiler or AudioProfiler()
        self.provenance = provenance_validator or AudioProvenanceValidator()
        self.content_analyzer = content_analyzer or AudioContentAnalyzer()
        self.attribution_engine = attribution_engine or AudioModelAttributionEngine()

    def analyze(
        self,
        audio_path: str | Path,
        sensitivity: str = "balanced",
    ) -> Dict[str, Any]:
        """Runs the entire end-to-end audio forensic analysis pipeline."""
        path = Path(audio_path)
        if not path.is_file():
            return {
                "content_valid": False,
                "final_status": "INVALID_FILE",
                "reason": f"File does not exist: {path}",
                "ai_detected": False,
                "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
                "evidence_trail": ["File not found on filesystem."],
            }

        # 1. Audio Container & Stream Validation
        val_res = self.validator.validate(path)
        if not val_res.valid:
            return {
                "content_valid": False,
                "final_status": "CORRUPT_OR_UNREADABLE",
                "reason": val_res.error or "Audio stream unreadable.",
                "ai_detected": False,
                "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
                "evidence_trail": [val_res.error or "Audio validation failure."],
            }

        # 2. Technical Signal Profiling
        profile_res = self.profiler.profile_audio(path)

        # 3. Provenance & Chunk Inspection
        provenance_res = self.provenance.analyze_provenance(path)

        # 4. Deep Acoustic & Statistical AI Detection
        if hasattr(val_res, "extracted_samples") and val_res.extracted_samples is not None:
            samples, sr, dur = val_res.extracted_samples
        else:
            samples, sr, dur = self.validator.extract_pcm_samples(path)
        ai_res = self.detector.analyze_audio_file(
            path,
            sensitivity=sensitivity,
            pre_extracted=(samples, sr, dur),
            generate_spectrogram=True,
        )

        # 5. Scene & Delivery Style Intelligence
        scene_res = self.content_analyzer.analyze_audio_scene(samples, sr) if samples is not None else {}

        # 6. Audio Synthesizer Attribution
        attribution_res = self.attribution_engine.attribute_audio(
            path,
            acoustic_data=ai_res,
            profile_data=profile_res,
            provenance_data=provenance_res,
        )

        # 7. Assemble Unified Forensic Dossier
        ai_percentage = float(ai_res.get("ai_percentage", 0.0))
        real_percentage = float(ai_res.get("real_percentage", 0.0))
        undecided_percentage = float(ai_res.get("undecided_percentage", 0.0))
        final_label = ai_res.get("label", "UNDECIDED")
        is_ai = final_label == "LIKELY AI-GENERATED"

        evidence_trail = []
        if provenance_res.get("c2pa_present"):
            evidence_trail.append("Cryptographic C2PA Content Credentials found in audio stream.")
        for cue in ai_res.get("forensic_cues", []):
            evidence_trail.append(cue)
        if attribution_res.get("attributed_model") != "Unknown":
            evidence_trail.append(f"Voice Synthesizer Fingerprint: {attribution_res.get('attributed_model')} ({attribution_res.get('confidence', 0)*100:.0f}% confidence)")

        decision_payload = {
            "final_status": final_label,
            "authenticity_probabilities": {
                "p_ai": ai_percentage,
                "p_real": real_percentage,
                "p_undecided": undecided_percentage,
            },
            "evidence_trail": evidence_trail,
        }
        nine_dims = build_audio_nine_dimensions_dossier(
            profile_data=profile_res,
            audio_result=ai_res,
            content_inventory=scene_res,
            provenance_result=provenance_res,
            attribution_result=attribution_res,
        )
        newbie_expl = generate_audio_newbie_explanation(
            filename=path.name,
            profile_data=profile_res,
            content_inventory=scene_res,
            audio_result=ai_res,
            decision=decision_payload,
        )

        return {
            "content_valid": True,
            "filename": path.name,
            "path": str(path),
            "final_status": final_label,
            "ai_detected": is_ai,
            "confidence": ai_res.get("confidence", 0.0),
            "authenticity_probabilities": {
                "p_ai": ai_percentage,
                "p_real": real_percentage,
                "p_undecided": undecided_percentage,
            },
            "file_profile": profile_res,
            "stream_validation": val_res.to_dict(),
            "provenance": provenance_res,
            "acoustic_forensics": ai_res,
            "scene_and_tone": scene_res,
            "model_attribution": attribution_res,
            "evidence_trail": evidence_trail,
            "nine_dimensions_dossier": nine_dims,
            "newbie_explanation": newbie_expl,
        }
