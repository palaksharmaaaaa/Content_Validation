"""Image honesty: a file name is not evidence, the profile shows the detector's own numbers, and untrusted chunks are bounded."""
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


def test_profile_noise_and_smoothness_are_the_detectors_numbers(tmp_path):
    from image_detector.pipeline import ImageForensicPipeline

    rng = np.random.default_rng(6)
    p = tmp_path / "y.png"
    Image.fromarray(rng.normal(120, 30, (400, 500, 3)).clip(0, 255).astype(np.uint8)).save(p)
    r = ImageForensicPipeline().analyze(p)
    phys = r["quantified_inventory"]["pixel_physics_metrics"]
    fm = r["ai_detection"]["forensic_metrics"]
    assert phys["prnu_noise_mean"] == round(fm["noise_residual_mean"], 3) and phys["surface_smoothness"] == fm["surface_smoothness"]


def test_a_compressed_text_chunk_cannot_expand_without_bound():
    import tracemalloc
    import zlib

    from image_detector.dimension_checks._common import png_text_fields

    bomb = zlib.compress(b"A" * (200 * 1024 * 1024), 9)             # about 200 KB that would inflate to 200 MB
    tracemalloc.start()
    out = png_text_fields([("zTXt", b"Comment\x00\x00" + bomb)], max_len=1000)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert len(out["Comment"]) == 1000 and peak < 20 * 1024 * 1024


def test_the_image_narrative_does_not_claim_missing_grain_when_the_grain_is_there():
    from image_detector.explain import _synthetic_takeaway

    grainy = _synthetic_takeaway(80.0, "Fully AI Generated", noise=2.4, smooth=3.0)
    smooth = _synthetic_takeaway(80.0, "Fully AI Generated", noise=0.5, smooth=0.9)
    assert "not unusually low" in grainy and "completely missing" not in grainy and "Midjourney" not in grainy
    assert "is low here" in smooth and "not proof" in smooth


def test_an_empty_profile_yields_not_recorded_not_invented_readings():
    from image_detector.explain import build_nine_dimensions_dossier

    d = build_nine_dimensions_dossier({}, {"taxonomy_state": "UNDECIDED"}, {}, {}, {})
    assert d["dimension_2"]["geometry"] == "Not recorded" and d["dimension_2"]["bit_depth"] == "Not recorded"
    assert d["dimension_2"]["luminance_dynamic_range"] == "Not recorded"
    assert d["dimension_3"]["is_natural_shot_noise"] is None and d["dimension_3"]["diagnosis"] == "Not measured"
    assert d["dimension_4"]["is_diffusion_smoothed"] is None and d["dimension_4"]["diagnosis"] == "Not measured"
    assert d["dimension_1"]["software_signature"] == "Not recorded" and d["dimension_1"]["date_taken"] == "Not recorded"
    assert d["dimension_8"]["color_channels"] is None
