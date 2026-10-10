"""Regressions for the core findings of the independent audit (Oct 2026)."""
import hashlib
import json
import math
import os

import numpy as np
import pytest

from core import atomic_io
from core.forensics import gates as G
from core.forensics.ood import OODGate
from core.forensics.registry import build_report
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from core.media_library import MediaLibrary, register_feedback
from core.retrain_engine import run_retrain
from core.shared_results import shift_probability_by_log_odds


def _sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize("encode", [
    lambda t: t.encode("utf-8-sig"),                       # BOM
    lambda t: t.encode("utf-16"),                          # PowerShell redirect
    lambda t: t.replace("\n", "\r\n").encode(),
])
def test_block_list_formats_are_all_read(tmp_path, encode):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"known bad")
    lst = tmp_path / "l.txt"
    lst.write_bytes(encode("# comment\nsha256:" + _sha(b"known bad") + "\n"))
    assert G.HardBlockGate(lst).check(bad).status == "HARD_BLOCK_ESCALATE"


def test_csv_prefix_and_label_forms(tmp_path):
    bad = tmp_path / "b.bin"
    bad.write_bytes(b"x1")
    lst = tmp_path / "l.csv"
    lst.write_text(f"label,hash\nfoo,{_sha(b'x1')}\n")
    assert G.HardBlockGate(lst).check(bad).triggered is True


def test_an_unusable_list_is_inactive_with_a_reason_never_clear(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"data")
    empty = tmp_path / "e.txt"
    empty.write_text("")
    comments = tmp_path / "c.txt"
    comments.write_text("# nothing here\n")
    for lst in (empty, comments, tmp_path, tmp_path / "missing.txt"):
        r = G.HardBlockGate(lst).check(f)
        assert r.status == "INACTIVE" and r.triggered is False and r.to_dict().get("note"), lst


def test_a_jpeg_with_dicm_at_byte_128_is_still_analysed(tmp_path):
    p = tmp_path / "t.jpg"
    body = bytearray(b"\xff\xd8\xff\xfe" + b"\x00" * 300)
    body[128:132] = b"DICM"
    p.write_bytes(bytes(body))
    assert G.recognize_scientific_format(p) is None


def test_an_unreadable_json_file_is_set_aside_not_overwritten(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_bytes(b'{"entries": [1, 2, ')                               # truncated
    assert atomic_io.atomic_read_json(p, default={}) == {}
    assert not p.exists() and list(tmp_path.glob("manifest.json.corrupt-*"))
    q = tmp_path / "bin.json"
    q.write_bytes(b"\xff\xfe\x00bad")
    assert atomic_io.atomic_read_json(q, default=None) is None
    assert list(tmp_path.glob("bin.json.corrupt-*"))


def test_a_nan_validation_accuracy_is_rolled_back(tmp_path):
    lib = MediaLibrary(tmp_path / "lib.json")
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"aaaa" * 50)
    b.write_bytes(b"bbbb" * 50)
    lib.add(a, "real")
    lib.add(b, "ai_generated")
    live = tmp_path / "live.pt"
    live.write_bytes(b"old")
    log = tmp_path / "LOG.md"

    def train(train_s, val_s, candidate):
        candidate.write_bytes(b"new")
        return 0.1, math.nan

    res = run_retrain(lib, log, live, train, min_new=1)
    assert res.promoted is False and live.read_bytes() == b"old" and not log.exists()


def test_only_ai_and_real_are_labels(tmp_path):
    lib = MediaLibrary(tmp_path / "lib.json")
    f = tmp_path / "x.bin"
    f.write_bytes(b"x" * 40)
    assert register_feedback(lib, f, "Not sure") is False
    assert register_feedback(lib, f, "AI") is True


def test_a_nan_llr_is_no_evidence():
    f = Finding(check_id="c", dimension="d", stage="s", title="t", status=FindingStatus.WARN, severity=Severity.LOW,
                evidence_class=EvidenceClass.METADATA_WEAK, detail="", llr=float("nan"))
    assert build_report([f]).score_terms == {}


def test_log_odds_shift_is_bounded_and_ignores_nan():
    assert shift_probability_by_log_odds(0.5, {"a": -400.0})[0] > 0.0                  # used to raise OverflowError
    assert shift_probability_by_log_odds(0.5, {"a": float("nan")})[0] == 0.5


def test_ood_non_finite_vectors():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 4))
    data[3, 1] = np.nan
    gate = OODGate().fit(data)
    assert gate.calibrated
    assert gate.score(np.array([np.nan, 0, 0, 0]))["status"] == "NOT_CALIBRATED"
