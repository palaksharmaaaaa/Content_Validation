"""Logic-only tests for ui.stages (no Streamlit runtime needed)."""
from ui import stages


def _finding(**kw):
    base = {
        "check_id": "x", "title": "Check X", "status": "PASS", "evidence_class": "SECURITY",
        "detail": "all good", "llr": None, "severity": "NONE", "stage": "file_integrity", "data": {},
    }
    base.update(kw)
    return base


def test_status_badges_cover_every_status():
    for status in ("PASS", "WARN", "FAIL", "INFO", "NOT_APPLICABLE", "NOT_CALIBRATED", "RECOGNIZED_OOS", "ERROR"):
        emoji, label = stages.status_badge(status)
        assert emoji and label
    assert stages.status_badge("SOMETHING_NEW") == ("❔", "SOMETHING_NEW")


def test_rows_basic_columns():
    rows = stages.findings_to_rows([_finding()], show_weight=False)
    assert rows == [{"Status": "✅ Pass", "Check": "Check X", "Class": "Security", "Detail": "all good"}]


def test_rows_weight_column_text():
    advisory = _finding(evidence_class="LEGAL_FLAG")
    weak_with = _finding(evidence_class="METADATA_WEAK", llr=-0.25)
    weak_without = _finding(evidence_class="METADATA_WEAK", llr=None)
    physical = _finding(evidence_class="PHYSICAL_SIGNAL", llr=0.25)
    rows = stages.findings_to_rows([advisory, weak_with, weak_without, physical], show_weight=True)
    assert rows[0]["Evidence weight"] == "Advisory only (never changes the score)"
    assert rows[1]["Evidence weight"] == "-0.25 log-odds (weak, capped)"
    assert rows[2]["Evidence weight"] == "Weak (no score effect)"
    assert rows[3]["Evidence weight"] == "+0.25 log-odds (physical, capped)"


def test_band_badges_for_all_five():
    seen = set()
    for band in ("HIGH_CONFIDENCE_SYNTHETIC", "LEANING_SYNTHETIC", "INCONCLUSIVE", "LEANING_AUTHENTIC", "HIGH_CONFIDENCE_AUTHENTIC"):
        emoji, label = stages.band_badge(band)
        assert emoji and label
        seen.add(emoji)
    assert len(seen) == 5


def test_collect_findings_across_stages_preserves_order():
    report = {"findings_by_stage": {
        "file_integrity": [_finding(check_id="a")],
        "formats": [_finding(check_id="b")],
        "legal": [_finding(check_id="c")],
    }}
    got = stages.collect_findings(report, ["formats", "file_integrity", "missing"])
    assert [f["check_id"] for f in got] == ["b", "a"]


def test_ood_status_text():
    assert "not fitted" in stages.ood_text({"status": "NOT_CALIBRATED"}).lower()
    assert "out-of-distribution" in stages.ood_text({"status": "OUT_OF_DISTRIBUTION", "distance": 9.1, "threshold": 4.2}).lower()
    assert "in-distribution" in stages.ood_text({"status": "IN_DISTRIBUTION", "distance": 1.0, "threshold": 4.2}).lower()
    assert stages.ood_text(None)  # never raises on missing data
