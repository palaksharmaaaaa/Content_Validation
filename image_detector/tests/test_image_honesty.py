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


def test_an_unreadable_input_is_not_labelled_an_authentic_photograph():
    from image_detector.detector import ImageAIDetector

    r = ImageAIDetector().predict(b"not an image")
    assert r["is_available"] is False and r["taxonomy_state"] == "UNDETERMINED" and r["taxonomy_label"] == "Undetermined"


def test_a_file_without_exif_records_nothing_it_did_not_say(tmp_path):
    from image_detector.profiler import ImageProfiler

    rng = np.random.default_rng(8)
    p = tmp_path / "plain.png"
    Image.fromarray(rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)).save(p)
    exif = ImageProfiler().profile_image(p)["exif_device_details"]
    for key in ("flash", "white_balance", "metering_mode", "exposure_bias", "orientation_tag", "color_space_tag"):
        assert exif[key] is None, key
    assert exif["gps_details"]["coordinates_str"] is None


def test_narratives_for_blank_and_unmeasured_pictures():
    from image_detector.explain import generate_newbie_explanation

    prof = {"spatial_geometry": {"width": 64, "height": 64, "aspect_ratio_str": "1:1"}}
    blank = generate_newbie_explanation("b.png", prof, {}, {}, {"final_status": "BLANK_OR_DEGRADED", "authenticity_probabilities": {}})
    assert "no verdict" in blank and "camera photograph" not in blank
    unmeasured = generate_newbie_explanation("u.png", prof, {}, {}, {"final_status": "LIKELY_SYNTHETIC", "authenticity_probabilities": {"p_ai": 80.0, "p_real": 10.0}})
    assert "not available" in unmeasured and "0.00" not in unmeasured


def test_software_needles_match_whole_words_only(tmp_path):
    from image_detector.attribution import ImageModelAttributionEngine

    Image.new("RGB", (70, 70)).save(tmp_path / "a.png")
    eng = ImageModelAttributionEngine()
    other = eng.attribute_image(tmp_path / "a.png", provenance_data={"metadata": {"software": "Maxaimag Studio 3"}})
    assert other["model_key"] == "unknown"
    named = eng.attribute_image(tmp_path / "a.png", provenance_data={"metadata": {"software": "xAI Grok Imagine"}})
    assert named["model_key"] == "grok_imagine"


def _photo(rng, h=768, w=1024, noise=3.0):
    import cv2

    base = cv2.GaussianBlur(rng.integers(40, 220, (h // 8, w // 8, 3), dtype=np.uint8), (5, 5), 0)
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC).astype(float)
    return np.clip(base + rng.normal(0, noise, (h, w, 3)), 0, 255).astype(np.uint8)


def test_a_photo_resaved_as_a_jpeg_is_not_called_ai_because_the_compression_removed_its_grain(tmp_path):
    import cv2

    from image_detector.detector import ImageAIDetector

    img = _photo(np.random.default_rng(1))
    det = ImageAIDetector()
    det.load()
    verdicts = {}
    for q in (100, 90, 75, 60):
        p = tmp_path / f"q{q}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, q])
        verdicts[q] = det.predict(p)
    for q in (90, 75, 60):
        assert verdicts[q]["taxonomy_state"] != "FULLY_AI_GENERATED" and verdicts[q]["ai_percentage"] < 55.0, (q, verdicts[q]["ai_percentage"])
    assert any("compressed too hard" in c for c in verdicts[75]["forensic_cues"])


def test_ordinary_1080p_photos_are_not_screenshots_on_their_size_alone():
    from image_detector.features import detect_screenshot

    rng = np.random.default_rng(3)
    photo = _photo(rng, 1080, 1920, noise=6.0)
    assert detect_screenshot("p.jpg", photo, {})["is_screenshot"] is False
    phone = np.full((2400, 1080, 3), 120, np.uint8)
    assert detect_screenshot("p.png", phone, {})["is_screenshot"] is True            # an unambiguous phone size still counts


def test_a_noisy_sky_and_grass_photo_is_not_digital_art():
    from image_detector.features import detect_digital_art_and_painting

    rng = np.random.default_rng(4)
    h, w = 600, 800
    scene = np.zeros((h, w, 3), np.uint8)
    scene[: h // 2] = (230, 140, 30)          # saturated sky (BGR)
    scene[h // 2:] = (40, 170, 40)            # saturated grass
    photo = np.clip(scene.astype(float) + rng.normal(0, 4, scene.shape), 0, 255).astype(np.uint8)
    assert detect_digital_art_and_painting("x.png", photo)["is_digital_art"] is False


def test_dossier_smoothness_cutoff_sits_between_real_and_generated():
    from image_detector.explain import build_nine_dimensions_dossier

    real = build_nine_dimensions_dossier({"raw_physical_signals": {"surface_smoothness_index": 1.95}}, {}, {}, {}, {})
    fake = build_nine_dimensions_dossier({"raw_physical_signals": {"surface_smoothness_index": 0.75}}, {}, {}, {}, {})
    assert real["dimension_4"]["is_diffusion_smoothed"] is False and fake["dimension_4"]["is_diffusion_smoothed"] is True
