import pytest

from core.calibration_report import evaluate


def test_perfect_and_empty():
    r = evaluate([("ai_generated", 99.0)] * 20 + [("real", 1.0)] * 20)
    assert r["accuracy"] == 1.0 and r["f1"] == 1.0 and r["false_positive_rate"] == 0.0
    assert r["ece"] < 0.02 and not r["warnings"]
    assert evaluate([])["n"] == 0


def test_overconfident_model_has_large_ece_and_warns_small_n():
    r = evaluate([("ai", 95.0), ("real", 95.0)] * 4)
    assert r["accuracy"] == 0.5 and r["ece"] == pytest.approx(0.45, abs=0.01)
    assert any("unreliable" in w for w in r["warnings"])


def test_single_class_warns_and_band_occupancy():
    r = evaluate([("real", 3.0)] * 40)
    assert any("one class" in w.lower() for w in r["warnings"])
    assert r["band_occupancy"]["real"]
    assert r["precision"] is None
