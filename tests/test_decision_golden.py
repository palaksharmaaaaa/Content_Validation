"""Golden regression: the decision layer's output for 14 representative evidence combinations must not drift."""
import json
from pathlib import Path

from core.decision import generate_final_decision as g
from tests.golden_support import assert_golden

OK = {"readable": True}
IMG_AUTH = {"ai_percentage": 12.0, "real_percentage": 80.0, "undecided_percentage": 8.0,
            "taxonomy_state": "AUTHENTIC_REAL_PHOTOGRAPH", "label": "LIKELY REAL", "forensic_cues": ["a"],
            "ai_spatial_area_pct": 3.0, "taxonomy_label": "L", "taxonomy_description": "D", "taxonomy_reasons": ["r"]}
IMG_AI = {"ai_percentage": 91.0, "real_percentage": 5.0, "undecided_percentage": 4.0,
          "taxonomy_state": "FULLY_AI_GENERATED", "label": "LIKELY AI-GENERATED", "neural_enhancer_detected": False}
IMG_PLAIN = {"ai_probability": 0.7}
VID = {"ai_percentage": 93.0, "real_percentage": 5.0, "forensic_cues": ["f"], "temporal_consistency": {"temporal_warping_risk": "HIGH", "mean_motion_delta": 1.2}}
AUD = {"ai_percentage": 30.0, "real_percentage": 60.0, "has_audio_track": True, "forensic_cues": ["x"], "ai_duration_pct": 10.0}
INV = {"entities": {"humans": {"persons_count": 2, "faces_count": 1}, "animals": {"animal_types": ["dog"]}},
       "vehicles": {"vehicle_types": ["car"]}, "contents_and_items": {"identified_items": ["cup"], "text_regions_count": 3},
       "environment_and_surroundings": {"setting_type": "Street"}, "lighting_and_daytime": {"estimated_daytime": "Night"},
       "purpose_and_depiction": {"photographic_purpose": "Portrait"}, "estimated_speakers": 2}
PROV_NEST = {"c2pa": {"c2pa_present": True, "ai_declaration": True}, "exif": {"camera_make": "Sony"}}
PROV_SIGN = {"c2pa": {"c2pa_present": True, "is_signed": True}}
PROV_EXIF = {"exif": {"camera_make": "Sony", "camera_model": "A7"}}
PROV_AIS = {"ai_signature_found": True, "signature_details": "tool"}
ATTR = {"attribution_confidence": 0.9, "model_key": "sora", "attributed_model": "Sora", "region_of_origin": "US", "attribution_cues": ["c"]}
CM = {"is_multimodal": True, "tampering_risk": "HIGH", "cues": ["cm"]}

cases = {
    "invalid": dict(file_validation={"readable": False, "error": "bad"}, quality_result={}),
    "blank": dict(file_validation=OK, quality_result={"is_blank": True}, content_inventory=INV),
    "empty": dict(file_validation=OK, quality_result={}),
    "img_auth": dict(file_validation=OK, quality_result={}, ai_result=IMG_AUTH, content_inventory=INV, provenance_result=PROV_EXIF, attribution_result=ATTR),
    "img_ai": dict(file_validation=OK, quality_result={}, ai_result=IMG_AI, attribution_result=ATTR),
    "img_plain": dict(file_validation=OK, quality_result={}, ai_result=IMG_PLAIN),
    "vid": dict(file_validation=OK, quality_result={}, video_result=VID, audio_result=AUD, cross_modal_result=CM, content_inventory=INV, attribution_result=ATTR),
    "vid_legacy": dict(file_validation=OK, quality_result={"ai_video_rating": {"ai_percentage": 70.0, "real_percentage": 25.0, "details": {"ai_duration_pct": 40}}}),
    "aud": dict(file_validation=OK, quality_result={}, audio_result=AUD, provenance_result=PROV_AIS),
    "prov_nest": dict(file_validation=OK, quality_result={}, video_result=VID, provenance_result=PROV_NEST),
    "prov_sign": dict(file_validation=OK, quality_result={}, video_result=VID, provenance_result=PROV_SIGN),
    "prov_exif": dict(file_validation=OK, quality_result={}, provenance_result=PROV_EXIF),
    "prov_ais": dict(file_validation=OK, quality_result={}, provenance_result=PROV_AIS),
    "img_and_aud": dict(file_validation=OK, quality_result={}, ai_result=IMG_AUTH, audio_result=AUD),
}


def test_decision_output_matches_golden():
    assert_golden("decision_golden", {k: g(**v) for k, v in cases.items()})
