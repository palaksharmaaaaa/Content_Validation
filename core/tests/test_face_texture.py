import numpy as np

from core.perception.face_texture import risk_label, texture_cues


def test_flat_face_is_waxy_and_noise_free():
    c = texture_cues(np.full((80, 80), 120, np.uint8))
    assert c["is_waxy_texture"] and c["facial_noise_residual"] < 1.8 and c["score"] == 0.9


def test_noisy_face_scores_the_floor():
    rng = np.random.default_rng(0)
    c = texture_cues(rng.integers(0, 256, (80, 80), dtype=np.uint8))
    assert not c["is_waxy_texture"] and c["score"] == 0.15


def test_a_patch_too_small_to_measure_is_none_and_adds_no_risk():
    c = texture_cues(np.full((1, 1), 120, np.uint8))
    assert c["facial_noise_residual"] is None and c["score"] == min(0.95, (1.5 / 2.5) * 0.75 + 0.15)


def test_empty_crop_and_labels():
    assert texture_cues(np.zeros((0, 0), np.uint8)) is None
    assert [risk_label(x) for x in (0.1, 0.5, 0.9)] == ["LOW_RISK_NATURAL_TEXTURE", "SUSPICIOUS_ARTIFACTS", "HIGH_SYNTHETIC_RISK"]
