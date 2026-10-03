"""Unauthenticated camera EXIF must not be able to cancel strong synthetic pixel evidence."""
import cv2
import numpy as np
import pytest
from PIL import Image

from image_detector.detector import ImageAIDetector


@pytest.fixture(scope="module")
def detector():
    d = ImageAIDetector()
    d.load()
    return d


def _exif():
    ex = Image.Exif()
    ex[271], ex[272] = "Canon", "Canon EOS R5"
    sub = ex.get_ifd(0x8769)
    sub[0x9003] = "2024:05:01 10:00:00"
    sub[0x829A], sub[0x829D], sub[0x8827], sub[0x920A] = (1, 250), (28, 10), 100, (50, 1)
    return ex


def _save(tmp_path, name, arr, with_exif):
    p = tmp_path / f"{name}_{int(with_exif)}.jpg"
    Image.fromarray(arr).save(p, "JPEG", quality=92, **({"exif": _exif()} if with_exif else {}))
    return str(p)


def _gradient():
    yy, xx = np.mgrid[0:512, 0:512]
    return np.stack([(xx / 2).astype(np.uint8), (yy / 2).astype(np.uint8), ((xx + yy) / 4).astype(np.uint8)], axis=-1)


def _grain():
    return np.clip(np.random.default_rng(7).normal(128, 9, (512, 512, 3)), 0, 255).astype(np.uint8)


def test_forged_exif_cannot_flip_synthetic_gradient_to_real(detector, tmp_path):
    plain = detector.predict(_save(tmp_path, "g", _gradient(), False))
    forged = detector.predict(_save(tmp_path, "g", _gradient(), True))
    assert plain["ai_percentage"] > 90
    assert forged["ai_percentage"] > 60, "forged camera EXIF must not erase strong synthetic evidence"
    assert forged["label"] != "LIKELY REAL"
    assert any("CONTRADICTED" in c for c in forged["forensic_cues"])
    assert forged["log_likelihood_ratios"]["exif_hardware"] == pytest.approx(-0.3)


def test_blurred_synthetic_like_image_with_exif_stays_suspicious(detector, tmp_path):
    arr = cv2.GaussianBlur(np.random.default_rng(7).integers(60, 190, (512, 512, 3), dtype=np.uint8), (0, 0), 6)
    assert detector.predict(_save(tmp_path, "b", arr, True))["ai_percentage"] > 60


def test_camera_exif_with_natural_noise_keeps_full_credit(detector, tmp_path):
    r = detector.predict(_save(tmp_path, "n", _grain(), True))
    assert r["label"] == "LIKELY REAL"
    assert r["log_likelihood_ratios"]["exif_hardware"] == pytest.approx(-1.4)
    assert not any("CONTRADICTED" in c for c in r["forensic_cues"])


def test_no_exif_behavior_unchanged(detector, tmp_path):
    # Golden values captured before the trust policy was introduced (EXIF-less images never reach the policy).
    assert detector.predict(_save(tmp_path, "g", _gradient(), False))["ai_percentage"] == pytest.approx(97.0, abs=0.1)
    assert detector.predict(_save(tmp_path, "n", _grain(), False))["ai_percentage"] == pytest.approx(7.2, abs=0.1)
