"""Feedback moves calibration only by what was measured; it never falls back to a stand-in number."""
import numpy as np
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


def test_trainer_reads_a_folder_and_does_not_reseed_the_global_rng(tmp_path):
    import random

    import numpy as np
    from PIL import Image

    from image_detector.trainer import ImageDetectorTrainer

    for sub in ("ai_generated", "real"):
        (tmp_path / sub).mkdir()
        for i in range(3):
            Image.fromarray(np.full((40, 40, 3), 10 * i + 5, np.uint8)).save(tmp_path / sub / f"{i}.png")
    tr = ImageDetectorTrainer.__new__(ImageDetectorTrainer)
    random.seed(123)
    expected = random.random()
    random.seed(123)
    train, val = tr.prepare_data(tmp_path, batch_size=2, val_split=0.5)
    assert len(train.dataset) + len(val.dataset) == 6
    assert random.random() == expected
    out = tmp_path / "ck" / "m.pt"
    tr.architecture, tr.checkpoint_path = "resnet18", out
    import torch.nn as nn
    tr.model = nn.Linear(2, 2)
    tr.save_checkpoint()
    assert out.is_file() and not list(out.parent.glob("*.partial"))


def test_audio_training_and_scoring_use_the_same_feature_vector(tmp_path):
    from audio_detector.features import compute_spectral_features, feature_vector
    from audio_detector.tests.audio_fixtures import tone, write_wav
    from audio_detector.trainer import AudioDetectorTrainer

    for sub in ("ai_generated", "real"):
        (tmp_path / sub).mkdir()
        write_wav(tmp_path / sub / "a.wav", tone(2.0, noise=0.05, seed=1))
    tr = AudioDetectorTrainer(checkpoint_path=tmp_path / "ck.pt")
    X, y = tr.prepare_data_from_directory(tmp_path)
    assert sorted(y.tolist()) == [0, 1] and X.shape == (2, 5)
    f = compute_spectral_features(tone(2.0, noise=0.05, seed=1), 16000)
    assert list(X[0]) == pytest.approx(feature_vector(f), abs=1e-3)
    report = tr.export_feature_dataset(tmp_path, tmp_path / "f.npz")
    assert report["success"] and report["samples_processed"] == 2


def test_video_trainer_pairs_come_downscaled_and_labelled(tmp_path):
    import cv2

    from video_detector.trainer import VideoDetectorTrainer

    rng = np.random.default_rng(0)
    for sub in ("ai_generated", "real"):
        (tmp_path / sub).mkdir()
        w = cv2.VideoWriter(str(tmp_path / sub / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
        for _ in range(20):
            w.write(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
        w.release()
    tr = VideoDetectorTrainer.__new__(VideoDetectorTrainer)
    from video_detector.extractor import VideoFrameExtractor

    tr.extractor = VideoFrameExtractor(max_frames=20)
    pairs = tr.prepare_data_from_videos(tmp_path)
    assert pairs and {p[2] for p in pairs} == {0, 1}
    assert pairs[0][0].shape[:2] == (224, 224)
