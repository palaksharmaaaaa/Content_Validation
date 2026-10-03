"""Video context / legal / lifecycle / reliability checks and the VideoDimensionAnalysis helper."""
import json

import numpy as np
import pytest

from core.forensics.registry import CheckContext
from core.forensics.schemas import EvidenceClass, FindingStatus
from video_detector.dimension_checks import (
    VideoDimensionAnalysis, check_video_gates, context, legal, lifecycle, normalize_provenance, reliability, temporal_vector,
)
from video_detector.tests.video_fixtures import box, build_mp4, moving_square_frames, text_atom, trak, write_clip


def _ctx(path, content=None, ai=None, prov=None, extra=None):
    return CheckContext(path=path, modality="video", content=content or {}, ai_result=ai or {}, provenance=prov or {}, extra=extra or {})


def _clip(tmp_path, name="c.mp4", n=40, step=3, fps=25):
    return write_clip(tmp_path / name, moving_square_frames(n, step=step), fps=fps)


def _other_clip(tmp_path):
    rng = np.random.default_rng(5)
    frames = [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(40)]
    return write_clip(tmp_path / "other.mp4", frames)


# ---------------- fingerprint ----------------
def test_fingerprint_hashes_count_and_stability(tmp_path):
    p = _clip(tmp_path)
    h1, h2 = context.video_fingerprint_hashes(p), context.video_fingerprint_hashes(p)
    assert len(h1) >= 8 and h1 == h2
    assert all(len(x) == 16 for x in h1)


def test_context_no_index(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_VIDEO_FP_INDEX", str(tmp_path / "none.jsonl"))
    f = context.check_video_fingerprint(_ctx(_clip(tmp_path)))
    assert f.status == FindingStatus.INFO and f.evidence_class == EvidenceClass.CONTEXT and f.data["index_loaded"] is False


def test_context_index_hit_and_miss(tmp_path, monkeypatch):
    p = _clip(tmp_path)
    idx = tmp_path / "idx.jsonl"
    idx.write_text(json.dumps({"phashes": context.video_fingerprint_hashes(p), "label": "2018 exercise", "source": "archive"}) + "\n")
    monkeypatch.setenv("OMNI_VIDEO_FP_INDEX", str(idx))
    hit = context.check_video_fingerprint(_ctx(p))
    assert hit.status == FindingStatus.WARN and hit.data["matches"][0]["label"] == "2018 exercise"
    miss = context.check_video_fingerprint(_ctx(_other_clip(tmp_path)))
    assert miss.status == FindingStatus.INFO and miss.data["matches"] == []


# ---------------- legal ----------------
def _legal(tmp_path, **kw):
    p = tmp_path / "a.mp4"
    if not p.exists():
        p.write_bytes(build_mp4([trak()]))
    return {f.check_id: f for f in legal.check_rights_and_privacy(_ctx(p, **kw))}


def test_location_from_xyz_atom(tmp_path):
    extra = box(b"udta", text_atom(b"\xa9xyz", "+37.7749-122.4194/"))
    p = tmp_path / "loc.mp4"
    p.write_bytes(build_mp4([trak()], moov_extra=extra))
    out = {f.check_id: f for f in legal.check_rights_and_privacy(_ctx(p))}
    assert out["location_privacy"].status == FindingStatus.WARN
    assert all(f.evidence_class == EvidenceClass.LEGAL_FLAG for f in out.values())


def test_location_from_gps_telemetry_track(tmp_path):
    p = tmp_path / "gp.mp4"
    p.write_bytes(build_mp4([trak(), trak(handler=b"meta", fourcc=b"gpmd")]))
    out = {f.check_id: f for f in legal.check_rights_and_privacy(_ctx(p))}
    assert out["location_privacy"].status == FindingStatus.WARN


def test_no_location_passes(tmp_path):
    assert _legal(tmp_path)["location_privacy"].status == FindingStatus.PASS


def test_rights_notice_from_cpy(tmp_path):
    extra = box(b"udta", text_atom(b"\xa9cpy", "(c) 2024 ACME Films"))
    p = tmp_path / "r.mp4"
    p.write_bytes(build_mp4([trak()], moov_extra=extra))
    out = {f.check_id: f for f in legal.check_rights_and_privacy(_ctx(p))}
    assert out["rights_notice"].data["copyright"] == "(c) 2024 ACME Films"


def test_biometric_notice_both_content_shapes(tmp_path):
    assert _legal(tmp_path, content={"faces_count": 2})["biometric_notice"].status == FindingStatus.INFO
    nested = {"living_entities": {"humans": {"count": 1, "faces": 1}}}
    assert _legal(tmp_path, content=nested)["biometric_notice"].data["faces_count"] == 1
    assert _legal(tmp_path, content={})["biometric_notice"].status == FindingStatus.NOT_APPLICABLE


def test_ai_disclosure_label(tmp_path):
    assert _legal(tmp_path, ai={"ai_percentage": 85.0})["ai_disclosure_label"].status == FindingStatus.WARN
    assert _legal(tmp_path, ai={"ai_percentage": 85.0}, prov={"c2pa_present": True})["ai_disclosure_label"].status == FindingStatus.PASS
    assert _legal(tmp_path, ai={"ai_percentage": 5.0})["ai_disclosure_label"].status == FindingStatus.NOT_APPLICABLE


# ---------------- lifecycle / reliability ----------------
def test_cascade_high_for_vertical_low_bitrate_stripped(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(build_mp4([trak()], faststart=True, mdat_payload=b"\x00" * 3000))
    stream = {"fps": 30.0, "frames": 300, "width": 1080, "height": 1920, "duration": 10.0}
    res = lifecycle.assess_transcoding(_ctx(p, extra={"stream": stream}))
    assert res["likelihood"] >= 0.6
    f = lifecycle.check_transcoding_cascade(_ctx(p, extra={"stream": stream}))
    assert f.evidence_class == EvidenceClass.RELIABILITY and f.llr is None


def test_cascade_low_for_high_bitrate_camera_like(tmp_path):
    p = tmp_path / "cam.mp4"
    p.write_bytes(build_mp4([trak()], mdat_payload=b"\x00" * 3_000_000,
                            mvhd_args={"creation": 3_800_000_000, "modification": 3_800_000_000}))
    stream = {"fps": 25.0, "frames": 75, "width": 640, "height": 480, "duration": 3.0}
    assert lifecycle.assess_transcoding(_ctx(p, extra={"stream": stream}))["likelihood"] < 0.35


def test_reliability_limiters(tmp_path):
    p = tmp_path / "t.mp4"
    p.write_bytes(build_mp4([trak()], mdat_payload=b"\x00" * 500))
    stream = {"fps": 25.0, "frames": 25, "width": 160, "height": 120, "duration": 1.0}
    f = reliability.check_confidence_limiters(_ctx(p, extra={"stream": stream, "frames": []}))
    assert f.status == FindingStatus.WARN and f.data["level"] in ("REDUCED", "LOW")
    joined = " ".join(f.data["limiters"]).lower()
    assert "resolution" in joined and "short" in joined


def test_reliability_normal(tmp_path):
    p = tmp_path / "ok.mp4"
    p.write_bytes(build_mp4([trak()], mdat_payload=b"\x00" * 6_000_000,
                            mvhd_args={"creation": 3_800_000_000, "modification": 3_800_000_000}))
    stream = {"fps": 25.0, "frames": 250, "width": 1280, "height": 720, "duration": 10.0}
    f = reliability.check_confidence_limiters(_ctx(p, extra={"stream": stream, "frames": []}))
    assert f.status == FindingStatus.PASS and f.data["level"] == "NORMAL"


# ---------------- helper ----------------
def test_normalize_provenance():
    assert normalize_provenance(None) == {"c2pa_present": False}
    assert normalize_provenance({"c2pa": {"c2pa_present": True}}) == {"c2pa_present": True}


def test_temporal_vector():
    v = temporal_vector({"temporal_consistency": {"mean_motion_delta": 2.0, "motion_variance": 10.0},
                         "diffusion_flicker": {"flicker_score": 0.4, "flicker_ratio": 0.5, "mean_lum_jump": 3.0}})
    assert v.shape == (5,) and v[1] == pytest.approx(np.log1p(10.0))
    assert temporal_vector(None) is None and temporal_vector({}) is None


def test_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_HARDBLOCK_SHA256_FILE", str(tmp_path / "none.txt"))
    assert check_video_gates(_clip(tmp_path))["triggered"] is False
    aedat = tmp_path / "events.aedat"
    aedat.write_bytes(b"#!AER-DAT3.1\r\n" + b"\x00" * 50)
    g = check_video_gates(aedat)
    assert g["triggered"] and g["final_status"] == "RECOGNIZED_OUT_OF_SCOPE" and g["recognition"]["type"] == "AEDAT"


def test_run_pre_terms_and_post_report(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_VIDEO_FP_INDEX", str(tmp_path / "none.jsonl"))
    extra = box(b"udta", text_atom(b"\xa9cmt", "Created with Kling AI"))
    p = tmp_path / "g.mp4"
    p.write_bytes(build_mp4([trak()], moov_extra=extra))
    a = VideoDimensionAnalysis(p, profile={}, provenance=None, frames=[np.zeros((60, 80), np.uint8)] * 3)
    terms = a.run_pre()
    assert terms["mp4_metadata"] == pytest.approx(0.40)
    assert abs(sum(terms.values())) <= 0.40 + 1e-9

    clip = _clip(tmp_path)
    b = VideoDimensionAnalysis(clip)
    b.run_pre()
    report = b.run_post(content={"faces_count": 0}, ai_result={"ai_percentage": 99.7, "temporal_consistency": {
        "mean_motion_delta": 1.0, "motion_variance": 5.0}, "diffusion_flicker": {"flicker_score": 0.1, "flicker_ratio": 0.1,
                                                                                  "mean_lum_jump": 1.0}}, attribution={"model_key": "unknown"})
    assert report["confidence_band"]["band"] == "HIGH_CONFIDENCE_SYNTHETIC"
    assert report["ood"]["status"] == "NOT_CALIBRATED"
    assert report["attribution_open_set"]["unknown_source"] is True
    assert {"file_integrity", "container", "signal", "legal", "lifecycle", "reliability", "context"} <= set(report["findings_by_stage"])


def test_unknown_source_false_for_authentic(tmp_path):
    clip = _clip(tmp_path)
    a = VideoDimensionAnalysis(clip)
    a.run_pre()
    r = a.run_post(content={}, ai_result={"ai_percentage": 5.0}, attribution={"model_key": "unknown"})
    assert r["attribution_open_set"]["unknown_source"] is False
    assert r["confidence_band"]["band"] == "LEANING_AUTHENTIC"
