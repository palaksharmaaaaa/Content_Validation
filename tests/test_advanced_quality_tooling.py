"""Tests for advanced AI/ML quality tooling integrations:
- Evidently AI data/feature drift reporting in core.retrain_engine.
- Cleanlab confident learning label noise filtering in ui.results.feedback.
"""
from __future__ import annotations

import pytest

from core.retrain_engine import generate_drift_report
from ui.results.feedback import evaluate_feedback_label_quality


def test_evidently_drift_report_generation(tmp_path):
    """Verifies Evidently AI data drift report execution and interactive HTML report emission."""
    pytest.importorskip("evidently", reason="optional dependency: pip install -r requirements.txt")
    ref_samples, curr_samples = [], []
    for i in range(12):
        ref = tmp_path / f"ref{i}.png"
        ref.write_bytes(b"a" * (1024 + 37 * i))
        cur = tmp_path / f"curr{i}.png"
        cur.write_bytes(b"c" * (1024 + 400 * i))
        ref_samples.append((f"sha_r{i}", ref, "real" if i % 2 else "ai_generated"))
        curr_samples.append((f"sha_c{i}", cur, "ai_generated"))

    report_html = tmp_path / "reports" / "drift_report.html"
    res = generate_drift_report(ref_samples, curr_samples, report_output_path=report_html)

    assert isinstance(res, dict)
    assert "drift_detected" in res
    assert "drift_share" in res
    assert "drifted_features" in res
    assert res["report_path"] is not None
    assert report_html.is_file()
    assert report_html.stat().st_size > 500  # Generated standalone HTML file


def test_cleanlab_feedback_label_noise_detection():
    """Cleanlab should identify contradictory feedback as suspect noise when model confidence is high."""
    # Case 1: Model is 96% AI, user insists it's Real -> suspect noise
    forensic_ai = {
        "ai_percentage": 96.0,
        "real_percentage": 4.0,
        "prediction": "AI",
        "confidence": 0.96,
    }
    eval_ai = evaluate_feedback_label_quality("Real / authentic", forensic_ai)
    assert eval_ai["is_suspect_noise"] is True
    assert eval_ai["quality_score"] < 0.20
    assert "sure the other way" in eval_ai["reason"]

    # Case 2: Model is 92% Real, user confirms Real -> high quality, not noise
    forensic_real = {
        "ai_percentage": 8.0,
        "real_percentage": 92.0,
        "prediction": "REAL",
        "confidence": 0.92,
    }
    eval_real = evaluate_feedback_label_quality("Real / authentic", forensic_real)
    assert eval_real["is_suspect_noise"] is False
    assert eval_real["quality_score"] > 0.80

    # Case 3: Ambiguous or undecided case (e.g. 52% AI vs 48% Real) -> not flagged as suspect noise
    forensic_close = {
        "ai_percentage": 52.0,
        "real_percentage": 48.0,
        "prediction": "AI",
        "confidence": 0.52,
    }
    eval_close = evaluate_feedback_label_quality("Real / authentic", forensic_close)
    assert eval_close["is_suspect_noise"] is False


def test_a_correction_against_a_very_confident_detector_waits_for_confirmation(tmp_path, monkeypatch):
    """One mistaken click must not move the thresholds; a deliberate, confirmed correction still goes through."""
    import ui.results.feedback as fu

    recorded = []

    class Fake:
        def record_feedback(self, path, label, metrics, notes=""):
            recorded.append((label, notes))
            return {"samples_processed": 1}

    monkeypatch.setattr(fu, "_LEARNERS", {m: (lambda: Fake()) for m in ("image", "video", "audio")})
    shown = []
    monkeypatch.setattr(fu.st, "warning", lambda msg, *a, **k: shown.append(("warning", msg)))
    monkeypatch.setattr(fu.st, "success", lambda msg, *a, **k: shown.append(("success", msg)))
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    detector_sure_ai = {"ai_percentage": 97.0}
    assert fu._submit_feedback(f, "image", detector_sure_ai, "Real / authentic", "") is False
    assert recorded == [] and shown[-1][0] == "warning"                                  # nothing learned, and the reviewer is told why
    assert fu._submit_feedback(f, "image", detector_sure_ai, "Real / authentic", "", confirmed=True) is True
    assert recorded[0][0] == "REAL" and "confirmed against a confident detector result" in recorded[0][1]
    assert fu._submit_feedback(f, "image", {"ai_percentage": 55.0}, "Real / authentic", "") is True      # an unsure detector never blocks
    assert fu._submit_feedback(f, "image", detector_sure_ai, "AI-generated", "") is True                  # agreeing with it never blocks


def test_drift_report_is_skipped_when_a_side_is_too_small(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"x" * 100)
    res = generate_drift_report([("s", f, "real")] * 3, [("t", f, "real")] * 3)
    assert res["report_path"] is None and res["drift_detected"] is False and "Too few samples" in res["message"]
