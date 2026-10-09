"""The video dossier must report what was measured, and say so when nothing was; no constant may stand in for a measurement."""
import cv2
import numpy as np

from video_detector.detector import VideoAIDetector
from video_detector.explain import build_video_nine_dimensions_dossier


def _clip(tmp_path, name, frame_at, seconds=6, fps=10, size=(160, 120)):
    path = str(tmp_path / name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        vw.write(frame_at(i))
    vw.release()
    return path


def _dossier(result):
    return build_video_nine_dimensions_dossier({"geometry": {}}, result)


def test_blank_footage_is_not_reported_as_authentic_sensor_noise(tmp_path):
    flat = _clip(tmp_path, "flat.mp4", lambda i: np.full((120, 160, 3), 128, np.uint8))
    result = VideoAIDetector().analyze_video(flat)
    d3 = _dossier(result)["dimension_3"]
    assert "Authentic camera" not in d3["diagnosis"]
    assert d3["noise_score"] is None and "Not measured" in d3["diagnosis"]      # blank frames are skipped, so nothing was measured


def test_noisy_and_flat_clips_get_different_noise_scores(tmp_path):
    rng = np.random.default_rng(1)
    noisy = _clip(tmp_path, "noisy.mp4", lambda i: np.clip(128 + rng.normal(0, 25, (120, 160, 3)), 0, 255).astype(np.uint8))
    ramp = np.tile(np.linspace(40, 200, 160, dtype=np.uint8), (120, 1))
    smooth = _clip(tmp_path, "smooth.mp4", lambda i: cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR))
    n = _dossier(VideoAIDetector().analyze_video(noisy))["dimension_3"]
    s = _dossier(VideoAIDetector().analyze_video(smooth))["dimension_3"]
    assert n["noise_score"] > s["noise_score"] + 1.0
    assert n["is_natural_noise"] is True and s["is_natural_noise"] is False


def test_flickering_clip_reports_its_flicker(tmp_path):
    flick = _clip(tmp_path, "flick.mp4", lambda i: np.full((120, 160, 3), 60 if (i // 2) % 2 else 200, np.uint8))   # the extractor samples every 2nd frame
    steady = _clip(tmp_path, "steady.mp4", lambda i: np.full((120, 160, 3), 120, np.uint8))
    f = _dossier(VideoAIDetector().analyze_video(flick))["dimension_5"]
    s = _dossier(VideoAIDetector().analyze_video(steady))["dimension_5"]
    assert f["has_diffusion_flicker"] is True and f["flicker_score"] > 0.5
    assert s["has_diffusion_flicker"] is False and s["flicker_score"] < f["flicker_score"]


def test_missing_measurements_are_stated_not_invented():
    d = _dossier({})
    assert d["dimension_3"]["noise_score"] is None and "Not measured" in d["dimension_3"]["diagnosis"]
    assert d["dimension_4"]["motion_variance"] is None and "Not measured" in d["dimension_4"]["diagnosis"]
    assert d["dimension_5"]["flicker_score"] is None and "Not measured" in d["dimension_5"]["diagnosis"]
    two_frames = _dossier({"diffusion_flicker": {"flicker_score": 0.0, "has_diffusion_flicker": False, "frames_compared": 2}})
    assert two_frames["dimension_5"]["flicker_score"] is None


def test_the_app_path_measures_frame_noise_too(tmp_path):
    """ForensicService scores video frames with the image detector's frame scorer; its noise value must reach the dossier."""
    from services.forensic_service import ForensicService

    rng = np.random.default_rng(3)
    noisy = _clip(tmp_path, "noisy_app.mp4", lambda i: np.clip(128 + rng.normal(0, 25, (120, 160, 3)), 0, 255).astype(np.uint8))
    r = ForensicService.get_instance().video_pipeline.analyze(noisy)
    d3 = r["nine_dimensions_dossier"]["dimension_3"]
    assert d3["noise_score"] is not None and d3["is_natural_noise"] is True and "Not measured" not in d3["diagnosis"]


def test_video_is_named_only_when_its_metadata_says_so(tmp_path):
    from video_detector.attribution import VideoModelAttributionEngine

    p = tmp_path / "v.mp4"
    p.write_bytes(b"x")
    eng = VideoModelAttributionEngine()
    temporal = {"temporal_consistency": {"motion_variance": 200.0}, "diffusion_flicker": {"has_diffusion_flicker": True}}
    out = eng.attribute_video(p, temporal_data=temporal, provenance_data={"vendor_signatures_found": []})
    assert out["model_key"] == "unknown" and out["top_candidates"] == []
    named = eng.attribute_video(p, temporal_data=temporal, provenance_data={"vendor_signatures_found": ["Made with Pika Labs"]})
    assert named["model_key"] == "pika"


def test_a_video_too_small_to_judge_is_rejected_with_the_reason(tmp_path):
    import cv2

    from video_detector.validator import VideoValidator

    p = tmp_path / "tiny.mp4"
    rng = np.random.default_rng(0)
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    for _ in range(10):
        w.write(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
    w.release()
    r = VideoValidator().validate(p)
    assert not r.valid and "16x16" in r.error and "minimum" in r.error


def test_the_video_narrative_is_grounded_in_what_was_measured():
    from video_detector.explain import generate_video_newbie_explanation

    vr = {"temporal_consistency": {"temporal_warping_risk": "HIGH_WARPING_DETECTED", "motion_variance": 211.5},
          "diffusion_flicker": {"has_diffusion_flicker": True}, "mean_frame_noise": 0.42}
    prof = {"geometry": {"width": 640, "height": 360, "fps": 24.0, "duration_seconds": 5.0, "total_frames": 120}}
    ai = generate_video_newbie_explanation("a.mp4", prof, {}, vr, {"final_status": "LIKELY_SYNTHETIC", "authenticity_probabilities": {"p_ai": 80.0, "p_real": 10.0}})
    assert "211.5" in ai and "0.42" in ai and "was seen" in ai and "Sora" not in ai and "not proof" in ai
    real = generate_video_newbie_explanation("a.mp4", prof, {}, {**vr, "diffusion_flicker": {"has_diffusion_flicker": False}},
                                             {"final_status": "LIKELY_AUTHENTIC", "authenticity_probabilities": {"p_ai": 10.0, "p_real": 80.0}})
    assert "was not seen" in real and "zero" not in real
    blank = generate_video_newbie_explanation("a.mp4", prof, {}, vr, {"final_status": "BLANK_OR_DEGRADED", "authenticity_probabilities": {}})
    assert "no verdict" in blank
    unknown_rate = generate_video_newbie_explanation("a.mp4", {"geometry": {"width": 640, "height": 360, "fps": 0.0}}, {}, vr, {"final_status": "UNDETERMINED", "authenticity_probabilities": {"p_ai": 50.0}})
    assert "not recorded" in unknown_rate


def test_video_dossier_reports_only_what_was_measured():
    from video_detector.explain import build_video_nine_dimensions_dossier

    vr = {"label": "UNDECIDED", "ai_percentage": 50.0, "mean_frame_noise": 1.3, "ai_duration_pct": 40.0,
          "temporal_consistency": {}, "diffusion_flicker": {}}
    d = build_video_nine_dimensions_dossier({"geometry": {}}, vr, {}, {}, {}, {})
    assert d["dimension_7"]["is_ai_video"] is None and d["dimension_7"]["visual_medium"] == "Not determined"
    assert d["dimension_8"]["mean_frame_noise"] == 1.3 and "color_channels" not in d["dimension_8"]
    assert d["dimension_9"]["watermark_detected"] is None and d["dimension_9"]["suspicious_duration_pct"] == "40.0%"
    assert d["dimension_2"]["frame_rate"] == "Not recorded" and d["dimension_2"]["aspect_ratio"] == "Not recorded"


def test_a_real_analysis_carries_the_share_of_the_timeline_flagged(tmp_path):
    import cv2

    from video_detector.detector import VideoAIDetector

    rng = np.random.default_rng(4)
    p = tmp_path / "t.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 72))
    for _ in range(30):
        w.write(rng.integers(0, 256, (72, 96, 3), dtype=np.uint8))
    w.release()
    r = VideoAIDetector().analyze_video(p)
    assert r["ai_duration_pct"] is not None and 0.0 <= r["ai_duration_pct"] <= 100.0


def test_a_vendor_name_must_stand_alone_in_the_file(tmp_path):
    import os

    from video_detector.provenance import VideoProvenanceValidator

    rng = np.random.default_rng(11)
    noise = bytearray(rng.integers(0, 256, 200_000, dtype=np.uint8).tobytes())
    noise[5000:5003] = b"VEO"                      # three letters inside compressed data
    noise[7000:7003] = b"dji"
    noise[9000:9004] = b"kling"[:4]                # a fragment glued to other bytes
    p = tmp_path / "noise.mp4"
    p.write_bytes(bytes(noise))
    out = VideoProvenanceValidator().analyze_provenance(p)
    assert out["vendor_signatures_found"] == [] and out["camera_make"] is None
    named = tmp_path / "named.mp4"
    named.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8 + b" encoder=Made with Google Veo \x00" + b"\x00" * 50)
    out2 = VideoProvenanceValidator().analyze_provenance(named)
    assert "google veo" in out2["vendor_signatures_found"]
