"""Tests for core.forensics schemas + registry (caps, isolation, phases)."""
from pathlib import Path

import pytest

from core.forensics.registry import CheckContext, CheckRegistry, build_report
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity


def _finding(check_id="c", cls=EvidenceClass.METADATA_WEAK, llr=None, cap=0.25, **kw):
    return Finding(
        check_id=check_id,
        dimension="O",
        stage="formats",
        title=check_id,
        status=kw.pop("status", FindingStatus.INFO),
        severity=Severity.LOW,
        evidence_class=cls,
        llr=llr,
        llr_cap=cap,
        **kw,
    )


def _ctx():
    return CheckContext(path=Path("x.png"), modality="image")


def test_llr_forced_none_for_ineligible_classes():
    for cls in (EvidenceClass.SECURITY, EvidenceClass.LEGAL_FLAG, EvidenceClass.CONTEXT, EvidenceClass.RELIABILITY):
        report = build_report([_finding(cls=cls, llr=0.2)])
        assert report.findings[0].llr is None
        assert report.score_terms == {}


def test_per_finding_cap_applied():
    report = build_report([_finding("a", llr=0.9)])
    assert report.score_terms["a"] == pytest.approx(0.25)
    report = build_report([_finding("b", llr=-0.9)])
    assert report.score_terms["b"] == pytest.approx(-0.25)


def test_explicit_cap_override_allows_040():
    report = build_report([_finding("png", llr=0.9, cap=0.40)])
    assert report.score_terms["png"] == pytest.approx(0.40)


def test_total_clamp_is_proportional():
    report = build_report([_finding("a", llr=0.25), _finding("b", llr=0.25), _finding("c", llr=0.25)])
    assert sum(report.score_terms.values()) == pytest.approx(0.40)
    assert report.score_terms["a"] == pytest.approx(report.score_terms["b"])


def test_opposite_signs_not_clamped_when_within_cap():
    report = build_report([_finding("a", llr=0.25), _finding("b", llr=-0.25)])
    assert report.score_terms == {"a": pytest.approx(0.25), "b": pytest.approx(-0.25)}


def test_none_and_zero_llr_produce_no_terms():
    report = build_report([_finding("a", llr=None), _finding("b", llr=0.0)])
    assert report.score_terms == {}


def test_exception_in_check_becomes_error_finding():
    reg = CheckRegistry()

    @reg.register("image", "boom")
    def boom(ctx):
        raise RuntimeError("kaput")

    findings = reg.run("image", _ctx(), phase="pre")
    assert len(findings) == 1
    assert findings[0].status == FindingStatus.ERROR
    assert findings[0].check_id == "boom"
    assert "kaput" in findings[0].detail


def test_list_return_and_phase_separation():
    reg = CheckRegistry()

    @reg.register("image", "multi", phase="pre")
    def multi(ctx):
        return [_finding("m1"), _finding("m2")]

    @reg.register("image", "later", phase="post")
    def later(ctx):
        return _finding("later")

    assert [f.check_id for f in reg.run("image", _ctx(), phase="pre")] == ["m1", "m2"]
    assert [f.check_id for f in reg.run("image", _ctx(), phase="post")] == ["later"]
    assert reg.run("audio", _ctx(), phase="pre") == []


def test_none_return_is_ignored():
    reg = CheckRegistry()

    @reg.register("image", "nothing")
    def nothing(ctx):
        return None

    assert reg.run("image", _ctx(), phase="pre") == []


def test_report_to_dict_groups_by_stage_and_exposes_reliability():
    f = _finding("lim", cls=EvidenceClass.RELIABILITY, data={"level": "REDUCED", "limiters": ["low resolution"]})
    f.stage = "reliability"
    d = build_report([f], gates={"hard_block": {"status": "CLEAR"}}).to_dict()
    assert "reliability" in d["findings_by_stage"]
    assert d["gates"]["hard_block"]["status"] == "CLEAR"
    assert d["score_terms"] == {}
    assert d["summary"]["total"] == 1
