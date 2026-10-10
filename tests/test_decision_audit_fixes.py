"""Decision-layer regressions from the independent audit (Oct 2026): every case below produced a wrong or crashing result before."""
from core.decision import generate_final_decision as g

OK = {"readable": True}


def test_a_modality_that_is_99_percent_undecided_does_not_decide():
    vid = {"ai_percentage": 0.5, "real_percentage": 0.1, "undecided_percentage": 99.4, "temporal_consistency": {}}
    out = g(file_validation=OK, quality_result={}, video_result=vid)
    assert out["authenticity_probabilities"]["p_ai"] < 65.0 and out["final_status"] != "LIKELY_SYNTHETIC"


def test_none_values_do_not_crash():
    assert g(file_validation=OK, quality_result={}, audio_result={"ai_percentage": 60.0, "real_percentage": 30.0, "has_audio_track": True, "ai_duration_pct": None})
    img = {"ai_percentage": 90.0, "real_percentage": 5.0, "taxonomy_state": "FULLY_AI_GENERATED", "label": "LIKELY AI-GENERATED",
           "face_check": {"worst_p_ai": None}}
    assert g(file_validation=OK, quality_result={}, ai_result=img)["content_valid"]
    vid = {"ai_percentage": 60.0, "real_percentage": 30.0, "temporal_consistency": {"temporal_warping_risk": None, "mean_motion_delta": None}}
    assert g(file_validation=OK, quality_result={}, video_result=vid)["content_valid"]


def test_an_undetermined_verdict_is_not_reported_as_ai_detected():
    img = {"ai_percentage": 90.0, "real_percentage": 5.0, "undecided_percentage": 5.0, "taxonomy_state": "FULLY_AI_GENERATED",
           "label": "LIKELY AI-GENERATED", "face_check": {"worst_p_ai": 0.01}, "taxonomy_label": "x", "taxonomy_description": "y"}
    out = g(file_validation=OK, quality_result={}, ai_result=img)
    assert out["final_status"] == "UNDETERMINED" and out["ai_detected"] is False


def test_a_declared_ai_manifest_cannot_vanish_behind_a_real_pixel_verdict():
    img = {"ai_percentage": 12.0, "real_percentage": 80.0, "undecided_percentage": 8.0, "taxonomy_state": "AUTHENTIC_REAL_PHOTOGRAPH",
           "label": "LIKELY REAL", "taxonomy_label": "x", "taxonomy_description": "y"}
    prov = {"c2pa": {"c2pa_present": True, "ai_declaration": True}}
    out = g(file_validation=OK, quality_result={}, ai_result=img, provenance_result=prov)
    assert out["final_status"] == "UNDETERMINED" and out["ai_detected"] is False
    assert any("declares AI" in r for r in out["taxonomy_reasons"])


def test_a_benign_c2pa_marker_does_not_hide_a_metadata_ai_declaration():
    prov = {"c2pa": {"c2pa_present": True, "is_signed": True}, "exif": {"ai_signature_found": True, "signature_details": "Midjourney"}}
    out = g(file_validation=OK, quality_result={}, provenance_result=prov)
    assert any("Midjourney" in line for line in out["evidence_trail"])
    assert out["authenticity_probabilities"]["p_ai"] > 50.0


def test_audio_video_disagreement_adds_no_ai_probability():
    base = g(file_validation=OK, quality_result={}, video_result={"ai_percentage": 20.0, "real_percentage": 70.0})
    with_cm = g(file_validation=OK, quality_result={}, video_result={"ai_percentage": 20.0, "real_percentage": 70.0},
                cross_modal_result={"is_multimodal": True, "tampering_risk": "HIGH", "cues": ["x"]})
    assert with_cm["authenticity_probabilities"] == base["authenticity_probabilities"]
