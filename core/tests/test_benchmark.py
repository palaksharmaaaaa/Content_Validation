import numpy as np
import pytest
from PIL import Image

from core.benchmark import AI_LABEL, REAL_LABEL, evaluate_folder, roc_auc


def test_auc_counts_ties_half_and_ignores_order():
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    a = roc_auc([0.3, 0.7, 0.7, 0.2, 0.9], [1, 0, 1, 0, 1])
    assert a == roc_auc([0.9, 0.2, 0.7, 0.7, 0.3], [1, 0, 1, 0, 1])
    assert roc_auc([0.4, 0.6], [1, 1]) is None


def _folder(tmp_path, n_ai=3, n_real=3):
    for sub, n in (("ai_generated", n_ai), ("real", n_real)):
        (tmp_path / sub).mkdir()
        for i in range(n):
            (tmp_path / sub / f"{i}.jpg").write_bytes(b"x")
    (tmp_path / "real" / "notes.txt").write_text("ignored")
    return tmp_path


def test_abstentions_and_failures_are_reported_not_scored_as_real(tmp_path):
    root = _folder(tmp_path)

    def predict(p):
        if p.parent.name == "ai_generated":
            return {"label": AI_LABEL, "ai_percentage": 90.0} if p.stem != "2" else {"label": "UNDECIDED", "ai_percentage": 50.0}
        if p.stem == "2":
            raise RuntimeError("unreadable")
        return {"label": REAL_LABEL, "ai_percentage": 10.0}

    r = evaluate_folder(root, {".jpg"}, predict)
    assert (r["files"], r["abstained"], r["failed"], r["decided"]) == (6, 1, 1, 4)
    assert r["confusion_matrix"] == {"TP": 2, "FP": 0, "TN": 2, "FN": 0}
    assert r["accuracy"] == 1.0 and r["coverage"] == round(4 / 6, 4)
    assert r["roc_auc"] == 1.0


def test_empty_folder_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        evaluate_folder(tmp_path, {".jpg"}, lambda p: {})


def test_image_suite_runs_on_real_files(tmp_path):
    from image_detector.benchmarks import ImageBenchmarkSuite

    rng = np.random.default_rng(0)
    for sub in ("ai_generated", "real"):
        (tmp_path / sub).mkdir()
        for i in range(2):
            Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(tmp_path / sub / f"{i}.png")
    r = ImageBenchmarkSuite().evaluate_dataset(tmp_path)
    assert r["files"] == 4 and r["failed"] == 0 and r["scored"] == 4
    assert r["decided"] + r["abstained"] == 4


def test_audio_suite_runs_on_real_files(tmp_path):
    from audio_detector.benchmarks import AudioBenchmarkSuite
    from audio_detector.tests.audio_fixtures import tone, write_wav

    for sub in ("ai_generated", "real"):
        (tmp_path / sub).mkdir()
        write_wav(tmp_path / sub / "a.wav", tone(2.0, noise=0.05, seed=1))
    r = AudioBenchmarkSuite().evaluate_dataset(tmp_path)
    assert r["files"] == 2 and r["failed"] == 0 and r["scored"] == 2


def test_video_suite_runs_on_real_files(tmp_path):
    import cv2

    from video_detector.benchmarks import VideoBenchmarkSuite

    rng = np.random.default_rng(1)
    for sub in ("ai_generated", "real"):
        (tmp_path / sub).mkdir()
        w = cv2.VideoWriter(str(tmp_path / sub / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
        for _ in range(20):
            w.write(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
        w.release()
    r = VideoBenchmarkSuite().evaluate_dataset(tmp_path)
    assert r["files"] == 2 and r["failed"] == 0 and r["scored"] == 2
