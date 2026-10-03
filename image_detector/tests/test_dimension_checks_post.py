"""Context / legal / lifecycle / reliability checks and the ImageDimensionAnalysis helper."""
import json

import pytest
from PIL import Image

from core.forensics.registry import CheckContext
from core.forensics.schemas import EvidenceClass, FindingStatus
from image_detector.dimension_checks import (
    ImageDimensionAnalysis,
    check_image_gates,
    context,
    legal,
    lifecycle,
    normalize_provenance,
    reliability,
)
from image_detector.dimension_checks import _common as C


def _ctx(path, profile=None, provenance=None, content=None, ai=None):
    return CheckContext(path=path, modality="image", profile=profile or {}, provenance=provenance or {},
                        content=content or {}, ai_result=ai or {})


def _img(tmp_path, name="a.jpg", size=(160, 120), color=(120, 130, 140), **save):
    p = tmp_path / name
    Image.new("RGB", size, color).save(p, **save) if save else Image.new("RGB", size, color).save(p)
    return p


# ---------------- hashes + context ----------------
def test_hashes_stable_and_discriminative():
    a = Image.linear_gradient("L").convert("RGB")
    b = Image.new("RGB", (256, 256), (255, 255, 255))
    assert C.dhash64(a) == C.dhash64(a.copy())
    assert C.phash64(a) == C.phash64(a.copy())
    assert C.hamming64(C.phash64(a), C.phash64(b)) > 0 or C.hamming64(C.dhash64(a), C.dhash64(b)) > 0


def test_context_no_index_is_info_with_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_CONTEXT_HASH_INDEX", str(tmp_path / "none.jsonl"))
    f = context.check_perceptual_hash(_ctx(_img(tmp_path, quality=90)))
    assert f.status == FindingStatus.INFO
    assert f.evidence_class == EvidenceClass.CONTEXT
    assert len(f.data["phash"]) == 16
    assert f.data["index_loaded"] is False


def test_context_index_hit_warns(tmp_path, monkeypatch):
    p = _img(tmp_path, quality=90)
    with Image.open(p) as im:
        ph = C.to_hex64(C.phash64(im))
    idx = tmp_path / "idx.jsonl"
    idx.write_text(json.dumps({"phash": ph, "label": "2018 conflict photo", "source": "archive"}) + "\n")
    monkeypatch.setenv("OMNI_CONTEXT_HASH_INDEX", str(idx))
    f = context.check_perceptual_hash(_ctx(p))
    assert f.status == FindingStatus.WARN
    assert f.data["matches"][0]["label"] == "2018 conflict photo"


# ---------------- legal ----------------
def test_gps_flags_location_privacy(tmp_path):
    ex = Image.Exif()
    ex.get_ifd(0x8825)[1] = "N"
    ex.get_ifd(0x8825)[2] = (12.0, 30.0, 0.0)
    p = tmp_path / "gps.jpg"
    Image.new("RGB", (64, 64)).save(p, "JPEG", exif=ex)
    findings = {f.check_id: f for f in legal.check_rights_and_privacy(_ctx(p))}
    assert findings["location_privacy"].status == FindingStatus.WARN
    assert all(f.evidence_class == EvidenceClass.LEGAL_FLAG for f in findings.values())


def test_no_gps_location_not_flagged(tmp_path):
    findings = {f.check_id: f for f in legal.check_rights_and_privacy(_ctx(_img(tmp_path, quality=90)))}
    assert findings["location_privacy"].status == FindingStatus.PASS


def test_copyright_notice_detected(tmp_path):
    ex = Image.Exif()
    ex[0x8298] = "(c) 2024 Jane Doe"
    p = tmp_path / "c.jpg"
    Image.new("RGB", (64, 64)).save(p, "JPEG", exif=ex)
    findings = {f.check_id: f for f in legal.check_rights_and_privacy(_ctx(p))}
    assert findings["rights_notice"].data["copyright"] == "(c) 2024 Jane Doe"


def test_biometric_notice_when_faces_present(tmp_path):
    findings = {f.check_id: f for f in legal.check_rights_and_privacy(_ctx(_img(tmp_path, quality=90), content={"faces_count": 2}))}
    assert findings["biometric_notice"].status == FindingStatus.INFO
    assert findings["biometric_notice"].data["faces_count"] == 2


def test_ai_disclosure_missing_label_warns(tmp_path):
    f = {x.check_id: x for x in legal.check_rights_and_privacy(_ctx(_img(tmp_path, quality=90), ai={"ai_percentage": 80.0}))}
    assert f["ai_disclosure_label"].status == FindingStatus.WARN


def test_ai_disclosure_present_with_c2pa_passes(tmp_path):
    f = {x.check_id: x for x in legal.check_rights_and_privacy(
        _ctx(_img(tmp_path, quality=90), provenance={"c2pa_present": True}, ai={"ai_percentage": 80.0}))}
    assert f["ai_disclosure_label"].status == FindingStatus.PASS


def test_ai_disclosure_not_applicable_for_real(tmp_path):
    f = {x.check_id: x for x in legal.check_rights_and_privacy(_ctx(_img(tmp_path, quality=90), ai={"ai_percentage": 10.0}))}
    assert f["ai_disclosure_label"].status == FindingStatus.NOT_APPLICABLE


# ---------------- lifecycle + reliability ----------------
def test_whatsapp_like_file_has_high_reencode_likelihood(tmp_path):
    p = _img(tmp_path, "wa.jpg", size=(1600, 1200), quality=72)
    res = lifecycle.assess_platform_reencode(_ctx(p, profile={"width": 1600, "height": 1200}))
    assert res["likelihood"] >= 0.6
    f = lifecycle.check_platform_reencode(_ctx(p, profile={"width": 1600, "height": 1200}))
    assert f.evidence_class == EvidenceClass.RELIABILITY
    assert f.llr is None


def test_camera_like_file_has_low_reencode_likelihood(tmp_path):
    ex = Image.Exif()
    ex[271], ex[272] = "Canon", "Canon EOS R5"
    p = tmp_path / "cam.jpg"
    Image.new("RGB", (4000, 3000), (1, 2, 3)).save(p, "JPEG", exif=ex, qtables=[[3] * 64, [4] * 64], subsampling=0)
    res = lifecycle.assess_platform_reencode(_ctx(p, profile={"width": 4000, "height": 3000}))
    assert res["likelihood"] < 0.35


def test_reliability_limiters(tmp_path):
    p = _img(tmp_path, "tiny.jpg", size=(120, 90), quality=40)
    f = reliability.check_confidence_limiters(_ctx(p, profile={"width": 120, "height": 90}))
    assert f.status == FindingStatus.WARN
    assert f.data["level"] in ("REDUCED", "LOW")
    assert any("resolution" in lim.lower() for lim in f.data["limiters"])
    assert any("compression" in lim.lower() for lim in f.data["limiters"])


def test_reliability_normal_for_clean_large_png(tmp_path):
    p = _img(tmp_path, "big.png", size=(1024, 768))
    f = reliability.check_confidence_limiters(_ctx(p, profile={"width": 1024, "height": 768}))
    assert f.status == FindingStatus.PASS
    assert f.data["level"] == "NORMAL"


# ---------------- helper ----------------
def test_normalize_provenance_handles_both_shapes():
    ui_shape = {"c2pa": {"c2pa_present": True}, "exif": {"camera_make": "Canon", "camera_model": "R5"}}
    n = normalize_provenance(ui_shape)
    assert n["c2pa_present"] and n["has_camera_hardware"]
    pipe_shape = {"c2pa_present": False, "has_camera_hardware": True}
    assert normalize_provenance(pipe_shape)["has_camera_hardware"]
    assert normalize_provenance(None) == {"c2pa_present": False, "has_camera_hardware": False}


def test_gates_clear_for_normal_image(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_HARDBLOCK_SHA256_FILE", str(tmp_path / "missing.txt"))
    g = check_image_gates(_img(tmp_path, quality=90))
    assert g["hard_block"]["status"] == "INACTIVE"
    assert g["recognition"] is None
    assert g["triggered"] is False


def test_run_pre_returns_clamped_terms(tmp_path):
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "x\nSteps: 20, Sampler: Euler, CFG scale: 7")
    p = tmp_path / "g.png"
    Image.new("RGB", (64, 64)).save(p, "PNG", pnginfo=info)
    analysis = ImageDimensionAnalysis(p, profile={}, provenance=None)
    terms = analysis.run_pre()
    assert terms["png_chunks"] == pytest.approx(0.40)
    assert abs(sum(terms.values())) <= 0.40 + 1e-9


def test_run_post_builds_full_report(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_CONTEXT_HASH_INDEX", str(tmp_path / "none.jsonl"))
    p = _img(tmp_path, "r.jpg", size=(1600, 1200), quality=72)
    analysis = ImageDimensionAnalysis(p, profile={"width": 1600, "height": 1200}, provenance={})
    analysis.run_pre()
    report = analysis.run_post(
        content={"faces_count": 0},
        ai_result={"ai_percentage": 99.7, "real_percentage": 0.3, "undecided_percentage": 0.0},
        attribution={"model_key": "unknown"},
    )
    assert report["confidence_band"]["band"] == "HIGH_CONFIDENCE_SYNTHETIC"
    assert report["ood"]["status"] == "NOT_CALIBRATED"
    assert report["attribution_open_set"]["unknown_source"] is True
    assert "file_integrity" in report["findings_by_stage"]
    assert "lifecycle" in report["findings_by_stage"]
    assert report["reliability"]["level"] in ("NORMAL", "REDUCED", "LOW")


def test_unknown_source_false_for_authentic_verdict(tmp_path):
    p = _img(tmp_path, "r.jpg", quality=90)
    a = ImageDimensionAnalysis(p, profile={}, provenance={})
    a.run_pre()
    report = a.run_post(content={}, ai_result={"ai_percentage": 5.0}, attribution={"model_key": "none_authentic"})
    assert report["attribution_open_set"]["unknown_source"] is False
    assert report["confidence_band"]["band"] == "LEANING_AUTHENTIC"
