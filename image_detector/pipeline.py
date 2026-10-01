"""
image_detector.pipeline: End-to-End Linear Image Forensic Analysis Orchestrator.
Executes the strictly linear NIST-aligned forensic inspection:
Step 1: File Ingestion & Pre-Analysis Details Extraction (Dimensions, DPI, Pixel stats, EXIF, Colors, Noise).
Step 2: 9-Dimensions Forensic Analyzer (As described in GLOBAL_IMAGE_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md).
Step 3: Image Type & Category Identification (Ontology states, IPTC mapping, Genre, Medium, Spectrum).
Step 4: Algorithmic Detection & Quantified Inventory (Percentages, Counts of faces/objects/text/colors/noise).
Step 5: Result Generation & Plain-English Newbie Narrative Explanation.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from image_detector.attribution import ImageModelAttributionEngine
from image_detector.content import ImageContentAnalyzer
from image_detector.detector import ImageAIDetector
from image_detector.explain import build_nine_dimensions_dossier, generate_newbie_explanation
from image_detector.face import FaceDeepfakeDetector
from image_detector.profiler import ImageProfiler
from image_detector.provenance import ImageProvenanceValidator
from image_detector.validator import ImageValidator

logger = logging.getLogger("image_detector.pipeline")


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

    def analyze(
        self,
        image_path: str | Path,
        sensitivity: str = "high",
        source: str = "User Upload",
    ) -> Dict[str, Any]:
        """Runs the entire end-to-end linear image forensic analysis pipeline."""
        path = Path(image_path)
        if not path.is_file():
            return {
                "content_valid": False,
                "final_status": "INVALID_FILE",
                "reason": f"File does not exist: {path}",
                "ai_detected": False,
                "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
                "evidence_trail": ["File not found on filesystem."],
            }

        # 1. Quality & Format Validation
        val_res = self.validator.validate(path)
        if not val_res.valid:
            return {
                "content_valid": False,
                "final_status": "CORRUPT_OR_UNREADABLE",
                "reason": val_res.error or "Image stream corrupted or unreadable.",
                "ai_detected": False,
                "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
                "evidence_trail": [val_res.error or "File format failure."],
            }

        # 2. Stage 1: Pre-Analysis Feature & Metadata Extraction
        profile_res = self.extract_pre_analysis_details(path, source=source)

        # 3. Provenance & Cryptographic C2PA Verification
        provenance_res = self.provenance.analyze_provenance(path)

        # 4. Deep Learning & Statistical Sensor Noise AI Detection
        ai_res = self.detector.predict(path, sensitivity=sensitivity)

        # 5. Scene & Content Intelligence (Living entities, objects, text regions)
        content_res = self.content_analyzer.analyze_image_content(path)

        # 6. Foundation Model Attribution & Watermarking
        attribution_res = self.attribution_engine.attribute_image(
            path,
            forensic_data=ai_res,
            profile_data=profile_res,
            provenance_data=provenance_res,
        )

        ai_percentage = float(ai_res.get("ai_percentage", 0.0))
        real_percentage = float(ai_res.get("real_percentage", 0.0))
        undecided_percentage = float(ai_res.get("undecided_percentage", 0.0))
        final_label = ai_res.get("label", "UNDECIDED")
        is_ai = final_label == "LIKELY AI-GENERATED"

        # Evidence Trail Compilation
        evidence_trail = []
        if provenance_res.get("c2pa_present"):
            evidence_trail.append("Cryptographic C2PA Content Credentials found in file.")
        if provenance_res.get("has_camera_hardware"):
            make = provenance_res.get("camera_make", "")
            model = provenance_res.get("camera_model", "")
            evidence_trail.append(f"Authentic camera hardware tags verified: {make} {model}")
        for cue in ai_res.get("forensic_cues", []):
            evidence_trail.append(cue)
        attr_model = attribution_res.get("attributed_model", "")
        attr_conf = float(attribution_res.get("confidence", 0.0))
        if attr_model and not attr_model.startswith("None") and not attr_model.startswith("Unknown") and attr_conf > 0.0:
            evidence_trail.append(f"Generative fingerprint matched: {attr_model} ({int(attr_conf * 100)}% match)")

        # 7. Stage 2: 9-Dimensional NIST Forensic Dossier
        decision_stub = {
            "final_status": final_label,
            "taxonomy_label": ai_res.get("taxonomy_label"),
            "authenticity_probabilities": {
                "p_ai": ai_percentage,
                "p_real": real_percentage,
                "p_undecided": undecided_percentage,
            },
        }
        nine_dims = build_nine_dimensions_dossier(
            profile_data=profile_res,
            ai_result=ai_res,
            content_inventory=content_res,
            provenance_result=provenance_res,
            attribution_result=attribution_res,
        )

        # 8. Stage 5: Plain-English Newbie Narrative Explanation
        newbie_expl = generate_newbie_explanation(
            filename=path.name,
            profile_data=profile_res,
            content_inventory=content_res,
            ai_result=ai_res,
            decision=decision_stub,
        )

        # Quantified Inventory Summary (Percentages & Counts)
        quantified_inventory = {
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

        # Category and Type Identification
        category_identification = {
            "taxonomy_state": ai_res.get("taxonomy_state"),
            "taxonomy_label": ai_res.get("taxonomy_label"),
            "taxonomy_description": ai_res.get("taxonomy_description"),
            "iptc_digital_source_type": nine_dims["dimension_1_hardware_provenance"]["iptc_digital_source_type"],
            "primary_genre": content_res.get("purpose_and_depiction", {}).get("primary_genre", ai_res.get("subject_genre", "General Scene")),
            "visual_medium": ai_res.get("visual_medium", "Photographic Capture"),
            "sensor_spectrum": ai_res.get("sensor_spectrum", "Visible Spectrum (Bayer RGB)"),
            "document_layout": content_res.get("purpose_and_depiction", {}).get("document_layout", "None (Standard Visual Content)"),
        }

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
            "pre_analysis_details": profile_res,
            "file_profile": profile_res,
            "category_identification": category_identification,
            "taxonomy_state": ai_res.get("taxonomy_state"),
            "taxonomy_label": ai_res.get("taxonomy_label"),
            "taxonomy_description": ai_res.get("taxonomy_description"),
            "taxonomy_reasons": ai_res.get("taxonomy_reasons", []),
            "subject_genre": category_identification["primary_genre"],
            "visual_medium": category_identification["visual_medium"],
            "sensor_spectrum": category_identification["sensor_spectrum"],
            "document_layout": category_identification["document_layout"],
            "quantified_inventory": quantified_inventory,
            "nine_dimensions_dossier": nine_dims,
            "newbie_explanation": newbie_expl,
            "watermark_detected": ai_res.get("watermark_detected", False),
            "background_cutout_detected": ai_res.get("background_cutout_detected", False),
            "screen_recapture_detected": ai_res.get("screen_recapture_detected", False),
            "screen_recapture_analysis": ai_res.get("screen_recapture_details", {}),
            "face_swap_detected": ai_res.get("face_swap_detected", False),
            "digital_art_detected": ai_res.get("digital_art_detected", False),
            "screenshot_detected": ai_res.get("screenshot_detected", False),
            "screenshot_analysis": ai_res.get("screenshot_details", {}),
            "inpainting_detected": ai_res.get("inpainting_detected", False),
            "inpainting_analysis": ai_res.get("inpainting_details", {}),
            "quality_validation": val_res.to_dict(),
            "provenance": provenance_res,
            "ai_detection": ai_res,
            "content_inventory": content_res,
            "model_attribution": attribution_res,
            "evidence_trail": evidence_trail,
        }
