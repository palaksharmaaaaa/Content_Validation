"""
image_detector.pipeline: End-to-End Linear Image Forensic Analysis Orchestrator.
Executes the strictly linear forensic inspection:
Step 1: File Ingestion & Pre-Analysis Details Extraction (Dimensions, DPI, Pixel stats, EXIF, Colors, Noise).
Step 2: 9-Dimensions Forensic Analyzer.
Step 3: Image Type & Category Identification (Ontology states, IPTC mapping, Genre, Medium, Spectrum).
Step 4: Algorithmic Detection & Quantified Inventory (Percentages, Counts of faces/objects/text/colors/noise).
Step 5: Result Generation & Plain-English Newbie Narrative Explanation.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.decision import generate_final_decision
from image_detector.attribution import ImageModelAttributionEngine
from image_detector.content import ImageContentAnalyzer
from image_detector.detector import ImageAIDetector
from image_detector.dimension_checks import (
    ImageDimensionAnalysis,
    check_image_gates,
    gate_short_circuit_result,
    summarize_for_evidence_trail,
)
from image_detector.explain import build_nine_dimensions_dossier, generate_newbie_explanation
from image_detector.profiler import ImageProfiler
from image_detector.provenance import ImageProvenanceValidator
from image_detector.validator import ImageValidator

logger = logging.getLogger("image_detector.pipeline")


def _quantified_inventory(
    ai_res: Dict[str, Any], content_res: Dict[str, Any], profile_res: Dict[str, Any],
    ai_percentage: float, real_percentage: float, undecided_percentage: float,
) -> Dict[str, Any]:
    """Quantified inventory summary (percentages, counts, pixel-physics metrics)."""
    return {
        "authenticity_probabilities": {
            "p_ai_percentage": ai_percentage,
            "p_real_percentage": real_percentage,
            "p_undecided_percentage": undecided_percentage,
        },
        "spatial_anomaly_manipulated_area_pct": ai_res.get("ai_spatial_area_pct", 0.0),
        "living_entities": {
            "persons_count": content_res.get("persons_count", 0),
            "faces_count": content_res.get("faces_count", 0),
            "is_stylized_character": content_res.get("living_entities", {}).get("humans", {}).get("is_stylized_character", False),
            "animals_count": content_res.get("living_entities", {}).get("animals", {}).get("count", 0),
            "animal_types": content_res.get("living_entities", {}).get("animals", {}).get("animal_types", []),
        },
        "vehicles": {
            "vehicles_count": content_res.get("vehicles", {}).get("count", 0),
            "vehicle_types": content_res.get("vehicles", {}).get("types", []),
        },
        "objects_and_items": {
            "items_count": len(content_res.get("contents_and_items", {}).get("identified_items", [])),
            "identified_items": content_res.get("contents_and_items", {}).get("identified_items", []),
        },
        "text_and_typography": {
            "text_regions_count": content_res.get("contents_and_items", {}).get("text_regions_count", 0),
        },
        "color_palette_percentages": profile_res.get("pixel_color_profile", {}).get("dominant_palette", []),
        "pixel_physics_metrics": {
            "prnu_noise_mean": profile_res.get("raw_physical_signals", {}).get("prnu_noise_mean", 0.0),
            "flat_region_noise": profile_res.get("raw_physical_signals", {}).get("flat_region_noise_mean", 0.0),
            "surface_smoothness": profile_res.get("raw_physical_signals", {}).get("surface_smoothness_index", 0.0),
            "fourier_fft_alpha": profile_res.get("raw_physical_signals", {}).get("fft_decay_alpha", 2.05),
            "highlight_clipped_pct": profile_res.get("pixel_color_profile", {}).get("highlight_clipped_pct", 0.0),
            "shadow_crushed_pct": profile_res.get("pixel_color_profile", {}).get("shadow_crushed_pct", 0.0),
        },
    }


def _evidence_trail(
    provenance_res: Dict[str, Any], ai_res: Dict[str, Any], attribution_res: Dict[str, Any], dimension_report: Dict[str, Any]
) -> List[str]:
    evidence_trail: List[str] = []
    if provenance_res.get("c2pa_present"):
        evidence_trail.append("C2PA Content Credentials markers found in file (presence only; not cryptographically verified).")
    if provenance_res.get("has_camera_hardware"):
        make = provenance_res.get("camera_make", "")
        model = provenance_res.get("camera_model", "")
        evidence_trail.append(f"Camera hardware EXIF tags present (unauthenticated metadata): {make} {model}")
    for cue in ai_res.get("forensic_cues", []):
        evidence_trail.append(cue)
    attr_model = attribution_res.get("attributed_model", "")
    attr_conf = float(attribution_res.get("confidence", 0.0))
    if attr_model and not attr_model.startswith("None") and not attr_model.startswith("Unknown") and attr_conf > 0.0:
        evidence_trail.append(f"Generative fingerprint matched: {attr_model} ({int(attr_conf * 100)}% match)")

    evidence_trail.extend(summarize_for_evidence_trail(dimension_report))
    return evidence_trail


@dataclass
class ImageRun:
    """Every intermediate of one image analysis; consumed by the headless report and by the Streamlit adapter."""

    path: Path
    gates: Dict[str, Any]
    profile: Dict[str, Any]
    provenance: Dict[str, Any]
    quality: Dict[str, Any]
    ai_result: Dict[str, Any]
    content: Dict[str, Any]
    attribution: Dict[str, Any]
    dimension_report: Dict[str, Any]
    decision: Dict[str, Any]
    nine_dimensions: Dict[str, Any]
    newbie_explanation: str


def _flat_quality(validation: Any) -> Dict[str, Any]:
    """Validator result as one flat dict (the nested ``quality`` block is merged up, as the decision layer expects)."""
    flat = validation.to_dict()
    if isinstance(flat.get("quality"), dict):
        flat.update(flat["quality"])
    return flat


FACE_LED_WEIGHT = 0.65     # share of the AI score taken from the face classifier when a face is the main subject
FACE_LED_MIN_P = 0.95      # ... and only when it is this sure


def _face_led_verdict(ai_result: Dict[str, Any], face_p_ai: float) -> None:
    """Portrait-style images: the face classifier (trained on exactly that kind of image) leads the verdict.
    The AI score becomes a weighted blend of the pixel detector and the face score, and the category follows the score.
    One-directional: a face that looks real never raises confidence, because an unseen generator would also look real."""
    if face_p_ai < FACE_LED_MIN_P or ai_result.get("taxonomy_state") in ("AI_GENERATED_SCREENSHOT", "AI_ENHANCED_SCREENSHOT"):
        return
    from image_detector.schemas import ImageTaxonomyState as S

    old = float(ai_result.get("ai_percentage", 0.0))
    blended = round((1 - FACE_LED_WEIGHT) * old + FACE_LED_WEIGHT * face_p_ai * 100.0, 1)
    if blended <= old:
        return
    undecided = min(float(ai_result.get("undecided_percentage", 0.0)), 100.0 - blended)
    ai_result.update(ai_percentage=blended, real_percentage=round(100.0 - blended - undecided, 1), undecided_percentage=round(undecided, 1))
    reason = f"The main face in the image looks AI-generated ({face_p_ai * 100:.0f}% by the face classifier)"
    ai_result["taxonomy_reasons"] = [reason] + list(ai_result.get("taxonomy_reasons", []))
    if blended >= 65.0 and ai_result.get("taxonomy_state") in (None, S.AUTHENTIC_REAL_PHOTOGRAPH, S.AUTHENTIC_EDITED, S.AUTHENTIC_RECAPTURED_SCREEN):
        ai_result.update(taxonomy_state=S.FULLY_AI_GENERATED, taxonomy_label=S.get_label(S.FULLY_AI_GENERATED),
                         taxonomy_description=S.get_description(S.FULLY_AI_GENERATED), label="LIKELY AI-GENERATED")


class ImageForensicPipeline:
    """Unified end-to-end linear forensic analysis pipeline for images."""

    def __init__(
        self,
        detector: Optional[ImageAIDetector] = None,
        validator: Optional[ImageValidator] = None,
        profiler: Optional[ImageProfiler] = None,
        provenance_validator: Optional[ImageProvenanceValidator] = None,
        content_analyzer: Optional[ImageContentAnalyzer] = None,
        attribution_engine: Optional[ImageModelAttributionEngine] = None,
    ):
        self.detector = detector or ImageAIDetector()
        self.detector.load()
        self.validator = validator or ImageValidator()
        self.profiler = profiler or ImageProfiler()
        self.provenance = provenance_validator or ImageProvenanceValidator()
        self.content_analyzer = content_analyzer or ImageContentAnalyzer()
        self.attribution_engine = attribution_engine or ImageModelAttributionEngine()

    def extract_pre_analysis_details(
        self, image_path: str | Path, source: str = "User Upload"
    ) -> Dict[str, Any]:
        """
        Stage 1: Extracts each and every detail (dimensions, DPI, pixel-by-pixel information,
        EXIF details, colors, noise metrics) before running any AI predictions.
        """
        return self.profiler.profile_image(image_path, source=source)

    def run(
        self,
        image_path: str | Path,
        *,
        sensitivity: str = "balanced",
        source: str = "User Upload",
        filename: Optional[str] = None,
        gates: Optional[Dict[str, Any]] = None,
        quality: Optional[Dict[str, Any]] = None,
    ) -> ImageRun:
        """The one analysis sequence. Callers handle gate/validation short-circuits before calling this."""
        path = Path(image_path)
        gates = gates if gates is not None else check_image_gates(path)

        profile = self.extract_pre_analysis_details(path, source=source)
        provenance = self.provenance.analyze_provenance(path)

        # Dimension checks (integrity / format / metadata) -> capped score terms
        dim_analysis = ImageDimensionAnalysis(path, profile=profile, provenance=provenance)
        dim_terms = dim_analysis.run_pre()

        ai_result = self.detector.predict(path, sensitivity=sensitivity, provenance=provenance, extra_log_lrs=dim_terms)
        face = dim_analysis.pre_finding("face_authenticity")
        if face is not None and face.data.get("face_dominant"):
            ai_result["face_check"] = {"worst_p_ai": face.data.get("worst_p_ai")}
        if face is not None and face.data.get("ai_like"):
            ai_result["face_ai_like"] = {"worst_p_ai": face.data.get("worst_p_ai")}
            if face.data.get("face_dominant"):
                _face_led_verdict(ai_result, float(face.data["worst_p_ai"]))
        content = self.content_analyzer.analyze_image_content(path)
        attribution = self.attribution_engine.attribute_image(
            path, forensic_data=ai_result, profile_data=profile, provenance_data=provenance
        )
        # Post-detector dimension checks (legal / context / lifecycle / reliability), band, OOD, open-set
        dimension_report = dim_analysis.run_post(content=content, ai_result=ai_result, attribution=attribution, gates=gates)

        quality = quality if quality is not None else _flat_quality(self.validator.validate(path))
        decision = generate_final_decision(
            file_validation={"readable": True},
            quality_result=quality,
            ai_result=ai_result,
            content_inventory=content,
            provenance_result=provenance,
            attribution_result=attribution,
        )
        nine_dimensions = build_nine_dimensions_dossier(
            profile_data=profile, ai_result=ai_result, content_inventory=content,
            provenance_result=provenance, attribution_result=attribution,
        )
        newbie = generate_newbie_explanation(
            filename=filename or path.name, profile_data=profile, content_inventory=content,
            ai_result=ai_result, decision=decision,
        )
        return ImageRun(path, gates, profile, provenance, quality, ai_result, content, attribution,
                        dimension_report, decision, nine_dimensions, newbie)

    def analyze(
        self,
        image_path: str | Path,
        sensitivity: str = "balanced",
        source: str = "User Upload",
    ) -> Dict[str, Any]:
        """Runs the entire end-to-end linear image forensic analysis pipeline and returns the headless report."""
        path = Path(image_path)
        if not path.is_file():
            return _terminal_report("INVALID_FILE", f"File does not exist: {path}", "File not found on filesystem.")

        # Pre-analysis gates (before decoding): hard-block hash list + out-of-scope scientific formats.
        gates = check_image_gates(path)
        if gates["triggered"]:
            return gate_short_circuit_result(path, gates)

        val_res = self.validator.validate(path)
        if not val_res.valid:
            return _terminal_report(
                "CORRUPT_OR_UNREADABLE", val_res.error or "Image stream corrupted or unreadable.",
                val_res.error or "File format failure.",
            )

        r = self.run(path, sensitivity=sensitivity, source=source, gates=gates, quality=_flat_quality(val_res))
        return self._report(r, val_res.to_dict())

    @staticmethod
    def _report(r: ImageRun, quality_validation: Dict[str, Any]) -> Dict[str, Any]:
        ai_res, content_res, decision = r.ai_result, r.content, r.decision
        probs = decision["authenticity_probabilities"]
        category_identification = {
            "taxonomy_state": ai_res.get("taxonomy_state"),
            "taxonomy_label": ai_res.get("taxonomy_label"),
            "taxonomy_description": ai_res.get("taxonomy_description"),
            "iptc_digital_source_type": r.nine_dimensions["dimension_1"]["iptc_digital_source_type"],
            "primary_genre": content_res.get("purpose_and_depiction", {}).get("primary_genre") or ai_res.get("subject_genre") or "Not determined",
            "visual_medium": ai_res.get("visual_medium", "Photographic Capture"),
            "sensor_spectrum": ai_res.get("sensor_spectrum", "Visible light (colour RGB)"),
            "document_layout": content_res.get("purpose_and_depiction", {}).get("document_layout", "None (Standard Visual Content)"),
        }
        return {
            "content_valid": True,
            "filename": r.path.name,
            "path": str(r.path),
            "final_status": decision["final_status"],
            "ai_detected": decision["ai_detected"],
            "decision": decision,
            "confidence": ai_res.get("confidence", 0.0),
            "authenticity_probabilities": {"p_ai": probs["p_ai"], "p_real": probs["p_real"], "p_undecided": probs["p_undecided"]},
            "pre_analysis_details": r.profile,
            "file_profile": r.profile,
            "category_identification": category_identification,
            "taxonomy_state": ai_res.get("taxonomy_state"),
            "taxonomy_label": ai_res.get("taxonomy_label"),
            "taxonomy_description": ai_res.get("taxonomy_description"),
            "taxonomy_reasons": ai_res.get("taxonomy_reasons", []),
            "subject_genre": category_identification["primary_genre"],
            "visual_medium": category_identification["visual_medium"],
            "sensor_spectrum": category_identification["sensor_spectrum"],
            "document_layout": category_identification["document_layout"],
            "quantified_inventory": _quantified_inventory(
                ai_res, content_res, r.profile, probs["p_ai"], probs["p_real"], probs["p_undecided"]
            ),
            "nine_dimensions_dossier": r.nine_dimensions,
            "newbie_explanation": r.newbie_explanation,
            "watermark_detected": ai_res.get("watermark_detected", False),
            "background_cutout_detected": ai_res.get("background_cutout_detected", False),
            "screen_recapture_detected": ai_res.get("screen_recapture_detected", False),
            "screen_recapture_analysis": ai_res.get("screen_recapture_details", {}),
            "digital_art_detected": ai_res.get("digital_art_detected", False),
            "screenshot_detected": ai_res.get("screenshot_detected", False),
            "screenshot_analysis": ai_res.get("screenshot_details", {}),
            "inpainting_detected": ai_res.get("inpainting_detected", False),
            "inpainting_analysis": ai_res.get("inpainting_details", {}),
            "quality_validation": quality_validation,
            "provenance": r.provenance,
            "ai_detection": ai_res,
            "content_inventory": content_res,
            "model_attribution": r.attribution,
            "dimension_report": r.dimension_report,
            "confidence_band": r.dimension_report.get("confidence_band"),
            "ood": r.dimension_report.get("ood"),
            "attribution_open_set": r.dimension_report.get("attribution_open_set"),
            "gate": r.gates,
            "evidence_trail": _evidence_trail(r.provenance, ai_res, r.attribution, r.dimension_report),
        }


def _terminal_report(status: str, reason: str, trail_line: str) -> Dict[str, Any]:
    return {
        "content_valid": False,
        "final_status": status,
        "reason": reason,
        "ai_detected": False,
        "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
        "evidence_trail": [trail_line],
    }
