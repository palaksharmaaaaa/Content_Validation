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


def test_feature_store_never_invents_an_embedding(tmp_path, monkeypatch):
    from image_detector.feature_store import FeatureStore

    store = FeatureStore.__new__(FeatureStore)
    store.backbone, store.fingerprint, store.device = None, "no-backbone", "cpu"
    img = np.full((40, 40, 3), 90, np.uint8)
    assert store.extract_features(img) is None
    p = tmp_path / "a.png"
    from PIL import Image

    Image.fromarray(img).save(p)
    assert store.build_feature_bank([(p, 0)], tmp_path / "bank.npz")["success"] is False
    assert not (tmp_path / "bank.npz").exists()


def test_resubmitting_the_same_file_does_not_step_calibration_again(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"not really a jpeg, but the same bytes")
    learner = _image(tmp_path)
    once = learner.record_feedback(str(f), "AI", {"noise_residual_mean": 1.0})
    for _ in range(20):
        again = learner.record_feedback(str(f), "AI", {"noise_residual_mean": 1.0})
    assert again == once and len(learner.load_memory()) == 1
    other = tmp_path / "b.jpg"
    other.write_bytes(b"different bytes")
    assert learner.record_feedback(str(other), "AI", {"noise_residual_mean": 1.0})["feature_weights"]["noise_residual"] > once["feature_weights"]["noise_residual"]


@pytest.mark.parametrize("label", ["Not sure", "", "maybe"])
def test_an_answer_that_is_not_ai_or_real_records_nothing(tmp_path, label):
    learner = _image(tmp_path)
    learner.record_feedback("x.jpg", label, {"noise_residual_mean": 1.0})
    assert learner.load_memory() == []


def test_a_tampered_calibration_file_cannot_feed_nan_or_huge_numbers_to_the_scoring(tmp_path):
    import json

    cal = tmp_path / "c.json"
    cal.write_text(json.dumps({"samples_processed": -4, "feature_weights": {"noise_residual": 1e9, "surface_smoothness": "big", "fft_decay": 0.25},
                               "sensitivity_offsets": {"noise_center_offset": 7.0}, "note": "kept"}).replace("1000000000.0", "1e9"))
    learner = ImageSelfImprover(memory_file=tmp_path / "m.json", calibration_file=cal)
    got = learner.load_calibration()
    assert got["feature_weights"]["noise_residual"] == 0.35 and got["feature_weights"]["surface_smoothness"] == 0.30
    assert got["feature_weights"]["fft_decay"] == 0.25 and got["sensitivity_offsets"]["noise_center_offset"] == 0.0
    assert got["samples_processed"] == 0 and got["note"] == "kept"
    cal.write_text('{"feature_weights": {"noise_residual": NaN}}')
    assert learner.load_calibration()["feature_weights"]["noise_residual"] == 0.35
