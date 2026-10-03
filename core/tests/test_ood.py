"""Epistemic OOD gate (Mahalanobis). Never invents a threshold when uncalibrated."""
import numpy as np

from core.forensics.ood import OODGate


def _cloud(n=400, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)


def test_uncalibrated_gate_reports_not_calibrated():
    gate = OODGate()
    res = gate.score(np.zeros(8, dtype=np.float32))
    assert res["status"] == "NOT_CALIBRATED"
    assert res["distance"] is None


def test_missing_file_loads_uncalibrated(tmp_path):
    gate = OODGate.load(tmp_path / "nope.npz")
    assert not gate.calibrated


def test_fit_then_score_in_and_out_of_distribution():
    gate = OODGate()
    gate.fit(_cloud())
    assert gate.calibrated
    assert gate.score(np.zeros(8, dtype=np.float32))["status"] == "IN_DISTRIBUTION"
    far = np.full(8, 12.0, dtype=np.float32)
    res = gate.score(far)
    assert res["status"] == "OUT_OF_DISTRIBUTION"
    assert res["distance"] > res["threshold"]


def test_save_load_round_trip(tmp_path):
    gate = OODGate()
    gate.fit(_cloud())
    p = tmp_path / "ood.npz"
    gate.save(p)
    loaded = OODGate.load(p)
    assert loaded.calibrated
    v = np.full(8, 3.0, dtype=np.float32)
    assert loaded.score(v)["distance"] == gate.score(v)["distance"]


def test_fit_rejects_too_few_samples():
    gate = OODGate()
    gate.fit(np.ones((3, 8), dtype=np.float32))
    assert not gate.calibrated


def test_wrong_dimension_is_not_calibrated_result():
    gate = OODGate()
    gate.fit(_cloud())
    res = gate.score(np.zeros(5, dtype=np.float32))
    assert res["status"] == "NOT_CALIBRATED"
