"""
video_detector.pipeline: End-to-End Video Forensic Analysis Orchestrator.
Executes the comprehensive forensic inspection:
1. Container & Codec Validation.
2. Signal Profiling & Stream Parameter Extraction.
3. Cryptographic Provenance & MP4 Atom Inspection.
4. Temporal Motion Continuity & Diffusion Flickering Detection.
5. Facial Deepfake Texture Analysis across Keyframes.
6. Scene, Entity & Environmental Content Intelligence.
7. Generative Video Model Attribution & Fingerprinting.
8. Cross-Modal Audio-Visual Synchronization (when demuxed audio present).
9. Calibrated Multi-Modal Decision & Audit Trail Generation.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.decision import generate_final_decision
from video_detector.attribution import VideoModelAttributionEngine
from video_detector.content import VideoContentAnalyzer
from video_detector.cross_modal import CrossModalConsistencyEngine
from video_detector.detector import VideoAIDetector
from video_detector.dimension_checks import (
    VideoDimensionAnalysis,
    check_video_gates,
    gate_short_circuit_result,
    summarize_for_evidence_trail,
)
from video_detector.explain import build_video_nine_dimensions_dossier, generate_video_newbie_explanation
from video_detector.profiler import VideoProfiler
from video_detector.provenance import VideoProvenanceValidator
from video_detector.validator import VideoValidator

logger = logging.getLogger("video_detector.pipeline")


@dataclass
class VideoRun:
    """Every intermediate of one video analysis; consumed by the headless report and by the Streamlit adapter."""

    path: Path
    gates: Dict[str, Any]
    profile: Dict[str, Any]
    provenance: Dict[str, Any]
    video_result: Dict[str, Any]
    audio_result: Optional[Dict[str, Any]]
    content: Dict[str, Any]
    keyframe: Optional[np.ndarray]
    keyframe_path: Optional[str]
    keyframe_ai: Optional[Dict[str, Any]]
    cross_modal: Dict[str, Any]
    attribution: Dict[str, Any]
    dimension_report: Dict[str, Any]
    decision: Dict[str, Any]
    nine_dimensions: Dict[str, Any]
    newbie_explanation: str


def _terminal_report(status: str, reason: str, trail_line: str) -> Dict[str, Any]:
    return {
        "content_valid": False,
        "final_status": status,
        "reason": reason,
        "ai_detected": False,
        "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
        "evidence_trail": [trail_line],
    }


def _evidence_trail(
    provenance: Dict[str, Any], ai_res: Dict[str, Any], cross_modal: Dict[str, Any],
    attribution: Dict[str, Any], dimension_report: Dict[str, Any],
) -> List[str]:
    trail: List[str] = []
    if provenance.get("c2pa_present"):
        trail.append("C2PA Content Credentials markers found in video container (presence only; not cryptographically verified).")
    trail.extend(ai_res.get("forensic_cues", []))
    trail.extend(f"Audio-Visual Cross-Modal: {cue}" for cue in cross_modal.get("cues", []) if "skipped" not in cue.lower())
    if attribution.get("attributed_model") != "Unknown":
        trail.append(f"Video Generator Fingerprint: {attribution.get('attributed_model')} ({attribution.get('confidence', 0)*100:.0f}% confidence)")
    trail.extend(summarize_for_evidence_trail(dimension_report))
    return trail


class VideoForensicPipeline:
    """Unified end-to-end forensic analysis pipeline for video recordings."""

    def __init__(
        self,
        detector: Optional[VideoAIDetector] = None,
        validator: Optional[VideoValidator] = None,
        profiler: Optional[VideoProfiler] = None,
        provenance_validator: Optional[VideoProvenanceValidator] = None,
        content_analyzer: Optional[VideoContentAnalyzer] = None,
        attribution_engine: Optional[VideoModelAttributionEngine] = None,
        cross_modal_engine: Optional[CrossModalConsistencyEngine] = None,
        audio_detector: Optional[Any] = None,
    ):
        """``audio_detector``: optional object with ``analyze_audio_file(path, sensitivity=...)``; when given, the
        file's audio track is analysed too and fused / cross-checked (injected by the services layer)."""
        self.detector = detector or VideoAIDetector()
        self.detector.load()
        self.validator = validator or VideoValidator()
        self.profiler = profiler or VideoProfiler()
        self.provenance = provenance_validator or VideoProvenanceValidator()
        self.content_analyzer = content_analyzer or VideoContentAnalyzer()
        self.attribution_engine = attribution_engine or VideoModelAttributionEngine()
        self.cross_modal_engine = cross_modal_engine or CrossModalConsistencyEngine()
        self.audio_detector = audio_detector

    def run(
        self,
        video_path: str | Path,
        *,
        sensitivity: str = "balanced",
        filename: Optional[str] = None,
        gates: Optional[Dict[str, Any]] = None,
        audio_forensics: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[Path] = None,
    ) -> VideoRun:
        """The one analysis sequence. Callers handle gate/validation short-circuits before calling this.

        ``audio_forensics`` overrides the pipeline's own audio analysis. ``cache_dir``: where the representative
        keyframe JPEG is written for display (omit to skip writing it).
        """
        path = Path(video_path)
        gates = gates if gates is not None else check_video_gates(path)

        profile = self.profiler.profile_video(path)
        provenance = self.provenance.analyze_provenance(path)

        # Dimension checks (integrity / container / signal) -> capped score terms
        dim_analysis = VideoDimensionAnalysis(path, profile=profile, provenance=provenance)
        dim_terms = dim_analysis.run_pre()
        video_result = self.detector.analyze_video(path, sensitivity=sensitivity, extra_log_lrs=dim_terms)

        if audio_forensics is None and self.audio_detector is not None:
            audio_forensics = self.audio_detector.analyze_audio_file(path, sensitivity=sensitivity)

        frames = video_result.get("keyframes")
        if not frames:
            frames, _, _ = self.detector.extractor.extract_sampled_frames(path, max_frames=12)
        content = self.content_analyzer.analyze_video_frames(frames, video_path=path)
        keyframe = self.content_analyzer.representative_frame(frames) if frames else None
        keyframe_path = self._write_keyframe(keyframe, cache_dir, filename or path.name)
        frame_detector = getattr(self.detector, "frame_detector", None)
        keyframe_ai = (
            frame_detector.predict(keyframe, sensitivity=sensitivity)
            if keyframe is not None and frame_detector is not None and hasattr(frame_detector, "predict") else None
        )

        cross_modal = self.cross_modal_engine.evaluate_consistency(
            video_forensics=video_result, audio_forensics=audio_forensics, content_inventory=content
        )
        attribution = self.attribution_engine.attribute_video(
            path, temporal_data=video_result, profile_data=profile, provenance_data=provenance
        )
        # Post-detector dimension checks (context / legal / lifecycle / reliability), band, OOD, open-set
        dimension_report = dim_analysis.run_post(content=content, ai_result=video_result, attribution=attribution, gates=gates)

        decision = generate_final_decision(
            file_validation={"readable": True},
            quality_result=video_result,
            video_result=video_result,
            audio_result=audio_forensics,
            content_inventory=content,
            provenance_result=provenance,
            cross_modal_result=cross_modal,
            attribution_result=attribution,
        )
        nine_dimensions = build_video_nine_dimensions_dossier(
            profile_data=profile, video_result=video_result, content_inventory=content,
            provenance_result=provenance, attribution_result=attribution, cross_modal_result=cross_modal,
        )
        newbie = generate_video_newbie_explanation(
            filename=filename or path.name, profile_data=profile, content_inventory=content,
            video_result=video_result, decision=decision,
        )
        return VideoRun(path, gates, profile, provenance, video_result, audio_forensics, content, keyframe, keyframe_path,
                        keyframe_ai, cross_modal, attribution, dimension_report, decision, nine_dimensions, newbie)

    @staticmethod
    def _write_keyframe(keyframe: Optional[np.ndarray], cache_dir: Optional[Path], filename: str) -> Optional[str]:
        if keyframe is None or cache_dir is None:
            return None
        target = Path(cache_dir) / f"kf_{Path(filename).stem}.jpg"
        cv2.imwrite(str(target), keyframe)
        return str(target)

    def analyze(
        self,
        video_path: str | Path,
        sensitivity: str = "balanced",
        audio_forensics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs the entire end-to-end video forensic analysis pipeline and returns the headless report."""
        path = Path(video_path)
        if not path.is_file():
            return _terminal_report("INVALID_FILE", f"File does not exist: {path}", "File not found on filesystem.")

        # Pre-analysis gates (before decoding): hard-block hash list + scientific-format recognition
        gates = check_video_gates(path)
        if gates["triggered"]:
            return gate_short_circuit_result(path, gates)

        val_res = self.validator.validate(path)
        if not val_res.valid:
            return _terminal_report(
                "CORRUPT_OR_UNREADABLE", val_res.error or "Video container unreadable.",
                val_res.error or "Container validation failure.",
            )
        r = self.run(path, sensitivity=sensitivity, gates=gates, audio_forensics=audio_forensics)
        return self._report(r, val_res.to_dict())

    @staticmethod
    def _report(r: VideoRun, container_validation: Dict[str, Any]) -> Dict[str, Any]:
        probs = r.decision["authenticity_probabilities"]
        return {
            "content_valid": True,
            "filename": r.path.name,
            "path": str(r.path),
            "final_status": r.decision["final_status"],
            "ai_detected": r.decision["ai_detected"],
            "decision": r.decision,
            "confidence": r.video_result.get("confidence", 0.0),
            "authenticity_probabilities": {"p_ai": probs["p_ai"], "p_real": probs["p_real"], "p_undecided": probs["p_undecided"]},
            "file_profile": r.profile,
            "container_validation": container_validation,
            "provenance": r.provenance,
            "temporal_forensics": r.video_result,
            "content_inventory": r.content,
            "cross_modal_consistency": r.cross_modal,
            "model_attribution": r.attribution,
            "evidence_trail": _evidence_trail(r.provenance, r.video_result, r.cross_modal, r.attribution, r.dimension_report),
            "nine_dimensions_dossier": r.nine_dimensions,
            "newbie_explanation": r.newbie_explanation,
            "dimension_report": r.dimension_report,
            "confidence_band": r.dimension_report.get("confidence_band"),
            "ood": r.dimension_report.get("ood"),
            "attribution_open_set": r.dimension_report.get("attribution_open_set"),
            "gate": r.gates,
        }
