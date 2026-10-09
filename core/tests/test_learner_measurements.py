"""Feedback moves calibration only by what was measured; it never falls back to a stand-in number."""
import pytest

from audio_detector.learner import AudioSelfImprover
from image_detector.learner import ImageSelfImprover
from video_detector.learner import VideoSelfImprover


def _image(tmp_path):
    return ImageSelfImprover(memory_file=tmp_path / "m.json", calibration_file=tmp_path / "c.json")


def test_image_reads_measurements_nested_under_forensic_metrics_as_the_app_passes_them(tmp_path):
    cal = _image(tmp_path).record_feedback("x.jpg", "AI", {"forensic_metrics": {"noise_residual_mean": 1.0, "surface_smoothness": 1.0}})
    assert cal["feature_weights"]["noise_residual"] > 0.35 and cal["feature_weights"]["surface_smoothness"] > 0.30
    assert cal["sensitivity_offsets"]["noise_center_offset"] < 0 < cal["sensitivity_offsets"]["smooth_center_offset"]


def test_image_flat_measurements_still_work(tmp_path):
    cal = _image(tmp_path).record_feedback("x.jpg", "AI", {"noise_residual_mean": 1.0})
    assert cal["feature_weights"]["noise_residual"] > 0.35


@pytest.mark.parametrize("label", ["AI", "REAL"])
def test_image_without_measurements_changes_nothing(tmp_path, label):
    cal = _image(tmp_path).record_feedback("x.jpg", label, {"label": "X"})
    assert cal["feature_weights"] == {"noise_residual": 0.35, "surface_smoothness": 0.30, "fft_decay": 0.20, "facial_shading": 0.25, "ela_discrepancy": 0.15}
    assert all(v == 0.0 for v in cal["sensitivity_offsets"].values())


def test_video_without_motion_measurement_changes_nothing(tmp_path):
    imp = VideoSelfImprover(memory_file=tmp_path / "m.json", calibration_file=tmp_path / "c.json")
    cal = imp.record_feedback("x.mp4", "AI", {"label": "X"})
    assert cal["sensitivity_offsets"]["video_ai_offset"] == 0.0
    cal = imp.record_feedback("x.mp4", "AI", {"temporal_consistency": {"motion_variance": 10.0}})
    assert cal["sensitivity_offsets"]["video_ai_offset"] > 0


def test_audio_without_flatness_leaves_the_flatness_threshold(tmp_path):
    imp = AudioSelfImprover(memory_file=tmp_path / "m.json", calibration_file=tmp_path / "c.json")
    cal = imp.record_feedback("x.wav", "AI", {"label": "X"})
    assert cal["thresholds"]["flatness_synthetic_max"] == 0.002
    cal = imp.record_feedback("x.wav", "AI", {"spectral_flatness": 0.05})
    assert cal["thresholds"]["flatness_synthetic_max"] > 0.002
