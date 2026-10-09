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
