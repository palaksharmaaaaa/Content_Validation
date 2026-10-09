"""A file name is not evidence: the same pixels score the same whatever the file is called."""
import shutil

import numpy as np
from PIL import Image

from image_detector.attribution import ImageModelAttributionEngine
from image_detector.detector import ImageAIDetector
from image_detector.features import detect_scanned_photo


def test_same_pixels_same_score_under_any_name(tmp_path):
    rng = np.random.default_rng(3)
    base = tmp_path / "photo.png"
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(base)
    det = ImageAIDetector()
    det.load()
    scores = set()
    for name in ("photo.png", "face-swap_reactor_deepfake.png", "gemini_export.png"):
        p = tmp_path / name
        if p != base:
            shutil.copy(base, p)
        r = det.predict(p)
        scores.add((r["ai_percentage"], r["label"]))
    assert len(scores) == 1


def test_attribution_ignores_the_file_name(tmp_path):
    Image.new("RGB", (64, 64)).save(tmp_path / "gemini_faceswap.png")
    Image.new("RGB", (64, 64)).save(tmp_path / "plain.png")
    eng = ImageModelAttributionEngine()
    a = eng.attribute_image(tmp_path / "gemini_faceswap.png", {}, {}, {})
    b = eng.attribute_image(tmp_path / "plain.png", {}, {}, {})
    assert a == b


def test_a_big_picture_without_a_scanner_frame_is_not_a_scan():
    big = np.full((4500, 5000, 3), 200, np.uint8)               # 22.5 MP, bright edges, no EXIF
    assert detect_scanned_photo("x.png", big, {"has_exif": False})["is_scanned"] is False
    framed = big.copy()
    framed[:60] = 10
    assert detect_scanned_photo("x.png", framed, {"has_exif": True})["is_scanned"] is True


def test_screenshot_is_decided_from_the_picture_not_its_name():
    from image_detector.features import detect_screenshot

    rng = np.random.default_rng(1)
    photo = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    assert detect_screenshot("Screenshot_2026.png", photo, {})["is_screenshot"] is False


def test_profile_and_detector_report_the_same_fourier_slope(tmp_path):
    from image_detector.pipeline import ImageForensicPipeline

    rng = np.random.default_rng(5)
    p = tmp_path / "x.png"
    Image.fromarray(rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)).save(p)
    report = ImageForensicPipeline().analyze(p)
    shown = report["quantified_inventory"]["pixel_physics_metrics"]["fourier_fft_alpha"]
    used = report["ai_detection"]["forensic_metrics"]["spectral_decay_alpha"]
    assert shown is not None and shown == used == report["nine_dimensions_dossier"]["dimension_5"]["spectral_decay_alpha"]
