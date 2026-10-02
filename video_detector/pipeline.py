"""
video_detector.pipeline: End-to-End Video Forensic Analysis Orchestrator.
Executes the comprehensive NIST-aligned forensic inspection:
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
from pathlib import Path
from typing import Any, Dict, Optional

from video_detector.attribution import VideoModelAttributionEngine
from video_detector.content import VideoContentAnalyzer
from video_detector.cross_modal import CrossModalConsistencyEngine
from video_detector.detector import VideoAIDetector
from video_detector.explain import build_video_nine_dimensions_dossier, generate_video_newbie_explanation
from video_detector.face import VideoFaceDeepfakeDetector
from video_detector.profiler import VideoProfiler
from video_detector.provenance import VideoProvenanceValidator
from video_detector.validator import VideoValidator

logger = logging.getLogger("video_detector.pipeline")


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
    ):
        self.detector = detector or VideoAIDetector()
        self.detector.load()
        self.validator = validator or VideoValidator()
        self.profiler = profiler or VideoProfiler()
        self.provenance = provenance_validator or VideoProvenanceValidator()
        self.content_analyzer = content_analyzer or VideoContentAnalyzer()
        self.attribution_engine = attribution_engine or VideoModelAttributionEngine()
        self.cross_modal_engine = cross_modal_engine or CrossModalConsistencyEngine()

    def analyze(
        self,
        video_path: str | Path,
        sensitivity: str = "high",
        audio_forensics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs the entire end-to-end video forensic analysis pipeline."""
        path = Path(video_path)
        if not path.is_file():
            return {
                "content_valid": False,
                "final_status": "INVALID_FILE",
                "reason": f"File does not exist: {path}",
                "ai_detected": False,
                "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
                "evidence_trail": ["File not found on filesystem."],
            }

        # 1. Video Container & Codec Validation
        val_res = self.validator.validate(path)
        if not val_res.valid:
            return {
                "content_valid": False,
                "final_status": "CORRUPT_OR_UNREADABLE",
                "reason": val_res.error or "Video container unreadable.",
                "ai_detected": False,
                "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
                "evidence_trail": [val_res.error or "Container validation failure."],
            }

        # 2. Technical Stream Profiling
        profile_res = self.profiler.profile_video(path)

        # 3. Provenance & MP4 Atom Inspection
        provenance_res = self.provenance.analyze_provenance(path)

        # 4. Temporal Motion & Diffusion Flickering AI Detection
        ai_res = self.detector.analyze_video(path, sensitivity=sensitivity)

        # 5. Keyframe Extraction for Content & Face Analysis
        frames = ai_res.get("keyframes")
        if not frames:
            frames, _, _ = self.detector.extractor.extract_frames(path, max_frames=12)

        # 6. Scene & Content Intelligence
        content_res = self.content_analyzer.analyze_video_frames(frames)

        # 7. Video Generative Model Attribution
        attribution_res = self.attribution_engine.attribute_video(
            path,
            temporal_data=ai_res,
            profile_data=profile_res,
            provenance_data=provenance_res,
        )

        # 8. Cross-Modal Audio-Visual Consistency
        cross_modal_res = self.cross_modal_engine.evaluate_consistency(
            video_forensics=ai_res,
            audio_forensics=audio_forensics,
            content_inventory=content_res,
        )

        # 9. Assemble Unified Forensic Dossier
        ai_percentage = float(ai_res.get("ai_percentage", 0.0))
        real_percentage = float(ai_res.get("real_percentage", 0.0))
        undecided_percentage = float(ai_res.get("undecided_percentage", 0.0))
        final_label = ai_res.get("label", "UNDECIDED")
        is_ai = final_label == "LIKELY AI-GENERATED"

        evidence_trail = []
        if provenance_res.get("c2pa_present"):
            evidence_trail.append("Cryptographic C2PA Content Credentials found in video container.")
        for cue in ai_res.get("forensic_cues", []):
            evidence_trail.append(cue)
        for cue in cross_modal_res.get("cues", []):
            if "skipped" not in cue.lower():
                evidence_trail.append(f"Audio-Visual Cross-Modal: {cue}")
        if attribution_res.get("attributed_model") != "Unknown":
            evidence_trail.append(f"Video Generator Fingerprint: {attribution_res.get('attributed_model')} ({attribution_res.get('confidence', 0)*100:.0f}% confidence)")

        decision_payload = {
            "final_status": final_label,
            "authenticity_probabilities": {
                "p_ai": ai_percentage,
                "p_real": real_percentage,
                "p_undecided": undecided_percentage,
            },
            "evidence_trail": evidence_trail,
        }
        nine_dims = build_video_nine_dimensions_dossier(
            profile_data=profile_res,
            video_result=ai_res,
            content_inventory=content_res,
            provenance_result=provenance_res,
            attribution_result=attribution_res,
            cross_modal_result=cross_modal_res,
        )
        newbie_expl = generate_video_newbie_explanation(
            filename=path.name,
            profile_data=profile_res,
            content_inventory=content_res,
            video_result=ai_res,
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
            "container_validation": val_res.to_dict(),
            "provenance": provenance_res,
            "temporal_forensics": ai_res,
            "content_inventory": content_res,
            "cross_modal_consistency": cross_modal_res,
            "model_attribution": attribution_res,
            "evidence_trail": evidence_trail,
            "nine_dimensions_dossier": nine_dims,
            "newbie_explanation": newbie_expl,
        }
