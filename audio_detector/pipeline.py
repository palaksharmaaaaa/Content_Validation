"""
audio_detector.pipeline: End-to-End Audio Forensic Analysis Orchestrator.
Executes the comprehensive forensic inspection:
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_detector.attribution import AudioModelAttributionEngine
from audio_detector.content import AudioContentAnalyzer
from audio_detector.detector import AudioAIDetector
from audio_detector.dimension_checks import (
    AudioDimensionAnalysis,
    check_audio_gates,
    gate_short_circuit_result,
    summarize_for_evidence_trail,
)
from audio_detector.explain import build_audio_nine_dimensions_dossier, generate_audio_newbie_explanation
from audio_detector.profiler import AudioProfiler
from audio_detector.provenance import AudioProvenanceValidator
from audio_detector.validator import AudioValidator
from core.decision import generate_final_decision

logger = logging.getLogger("audio_detector.pipeline")


@dataclass
class AudioRun:
    """Every intermediate of one audio analysis; consumed by the headless report and by the Streamlit adapter."""

    path: Path
    gates: dict[str, Any]
    samples: Any
    sample_rate: int
    duration: float
    profile: dict[str, Any]
    provenance: dict[str, Any]
    ai_result: dict[str, Any]
    scene: dict[str, Any]
    attribution: dict[str, Any]
    dimension_report: dict[str, Any]
    decision: dict[str, Any]
    nine_dimensions: dict[str, Any]
    newbie_explanation: str


def _terminal_report(status: str, reason: str, trail_line: str) -> dict[str, Any]:
    return {
        "content_valid": False,
        "final_status": status,
        "reason": reason,
        "ai_detected": False,
        "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
        "evidence_trail": [trail_line],
    }


def _evidence_trail(provenance: dict[str, Any], ai_res: dict[str, Any], attribution: dict[str, Any], dimension_report: dict[str, Any]) -> list[str]:
    trail: list[str] = []
    if provenance.get("c2pa_present"):
        trail.append("C2PA Content Credentials markers found in audio stream (presence only; not cryptographically verified).")
    trail.extend(ai_res.get("forensic_cues", []))
    if attribution.get("model_key", "unknown") != "unknown":
        trail.append(f"The file declares the voice generator: {attribution.get('attributed_model')} (unauthenticated claim)")
    trail.extend(summarize_for_evidence_trail(dimension_report))
    return trail


class AudioForensicPipeline:
    """Unified end-to-end forensic analysis pipeline for audio recordings."""

    def __init__(
        self,
        detector: AudioAIDetector | None = None,
        validator: AudioValidator | None = None,
        profiler: AudioProfiler | None = None,
        provenance_validator: AudioProvenanceValidator | None = None,
        content_analyzer: AudioContentAnalyzer | None = None,
        attribution_engine: AudioModelAttributionEngine | None = None,
    ):
        self.detector = detector or AudioAIDetector()
        self.detector.load()
        self.validator = validator or AudioValidator()
        self.profiler = profiler or AudioProfiler()
        self.provenance = provenance_validator or AudioProvenanceValidator()
        self.content_analyzer = content_analyzer or AudioContentAnalyzer()
        self.attribution_engine = attribution_engine or AudioModelAttributionEngine()

    def run(
        self,
        audio_path: str | Path,
        *,
        sensitivity: str = "balanced",
        filename: str | None = None,
        gates: dict[str, Any] | None = None,
        decoded: tuple[Any, int, float] | None = None,
    ) -> AudioRun:
        """The one analysis sequence. Callers handle gate/validation short-circuits before calling this."""
        path = Path(audio_path)
        gates = gates if gates is not None else check_audio_gates(path)

        samples, sr, dur = decoded if decoded is not None else self.validator.extract_pcm_samples(path)
        profile = self.profiler.profile_audio(path, decoded=(samples, sr, dur))      # one decode serves the profile and the detector
        provenance = self.provenance.analyze_provenance(path)

        # Dimension checks (integrity / container / signal) -> capped score terms
        dim_analysis = AudioDimensionAnalysis(path, profile=profile, provenance=provenance, samples=(samples, sr, dur))
        dim_terms = dim_analysis.run_pre()
        ai_result = self.detector.analyze_audio_file(
            path, sensitivity=sensitivity, pre_extracted=(samples, sr, dur), generate_spectrogram=True, extra_log_lrs=dim_terms
        )
        scene = self.content_analyzer.analyze_audio_scene(samples, sr) if samples is not None else {}
        attribution = self.attribution_engine.attribute_audio(
            path, acoustic_data=ai_result, profile_data=profile, provenance_data=provenance
        )
        # Post-detector dimension checks (context / legal / lifecycle / reliability), band, OOD, open-set
        dimension_report = dim_analysis.run_post(content=scene, ai_result=ai_result, attribution=attribution, gates=gates)

        decision = generate_final_decision(
            file_validation={"readable": True},
            quality_result={"is_blank": bool(profile.get("is_silent"))},          # no signal at all: nothing to judge
            audio_result=ai_result,
            content_inventory=scene,
            provenance_result=provenance,
            attribution_result=attribution,
        )
        nine_dimensions = build_audio_nine_dimensions_dossier(
            profile_data=profile, audio_result=ai_result, content_inventory=scene,
            provenance_result=provenance, attribution_result=attribution,
        )
        newbie = generate_audio_newbie_explanation(
            filename=filename or path.name, profile_data=profile, content_inventory=scene,
            audio_result=ai_result, decision=decision,
        )
        return AudioRun(path, gates, samples, sr, dur, profile, provenance, ai_result, scene, attribution,
                        dimension_report, decision, nine_dimensions, newbie)

    def analyze(
        self,
        audio_path: str | Path,
        sensitivity: str = "balanced",
    ) -> dict[str, Any]:
        """Runs the entire end-to-end audio forensic analysis pipeline and returns the headless report."""
        path = Path(audio_path)
        if not path.is_file():
            return _terminal_report("INVALID_FILE", f"File does not exist: {path}", "File not found on filesystem.")

        # Pre-analysis gates (before decoding): hard-block hash list + symbolic-music recognition
        gates = check_audio_gates(path)
        if gates["triggered"]:
            return gate_short_circuit_result(path, gates)

        val_res = self.validator.validate(path)
        if not val_res.valid:
            return _terminal_report(
                "CORRUPT_OR_UNREADABLE", val_res.error or "Audio stream unreadable.",
                val_res.error or "Audio validation failure.",
            )
        extracted = getattr(val_res, "extracted_samples", None)
        r = self.run(path, sensitivity=sensitivity, gates=gates, decoded=extracted)
        return self._report(r, val_res.to_dict())

    @staticmethod
    def _report(r: AudioRun, stream_validation: dict[str, Any]) -> dict[str, Any]:
        probs = r.decision["authenticity_probabilities"]
        return {
            "content_valid": True,
            "filename": r.path.name,
            "path": str(r.path),
            "final_status": r.decision["final_status"],
            "ai_detected": r.decision["ai_detected"],
            "decision": r.decision,
            "confidence": r.ai_result.get("confidence", 0.0),
            "authenticity_probabilities": {"p_ai": probs["p_ai"], "p_real": probs["p_real"], "p_undecided": probs["p_undecided"]},
            "file_profile": r.profile,
            "stream_validation": stream_validation,
            "provenance": r.provenance,
            "acoustic_forensics": r.ai_result,
            "scene_and_tone": r.scene,
            "model_attribution": r.attribution,
            "evidence_trail": _evidence_trail(r.provenance, r.ai_result, r.attribution, r.dimension_report),
            "nine_dimensions_dossier": r.nine_dimensions,
            "newbie_explanation": r.newbie_explanation,
            "dimension_report": r.dimension_report,
            "confidence_band": r.dimension_report.get("confidence_band"),
            "ood": r.dimension_report.get("ood"),
            "attribution_open_set": r.dimension_report.get("attribution_open_set"),
            "gate": r.gates,
        }
