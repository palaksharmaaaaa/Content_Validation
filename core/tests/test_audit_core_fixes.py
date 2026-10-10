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


def test_nan_age_is_not_adult():
    from core.perception.age import classify_minor
    assert classify_minor(float("nan"), None, usable=True) == "ADULT"          # nothing else flags a minor: only the age was lost
    assert classify_minor(float("nan"), None, usable=False) == "UNDETERMINED"
    assert classify_minor(float("nan"), 0.9, usable=True) == "POSSIBLE_MINOR"


def test_ood_gate_needs_at_least_as_many_samples_as_features():
    import numpy as np
    from core.forensics.ood import OODGate
    rng = np.random.default_rng(0)
    assert not OODGate().fit(rng.normal(size=(40, 64))).calibrated            # 40 samples cannot describe 64 dimensions
    assert OODGate().fit(rng.normal(size=(80, 16))).calibrated


def test_only_a_generator_the_file_declares_can_be_named():
    from core.shared_results import finalize_attribution

    catalog = {"a": {"name": "Alpha", "provider": "P"}, "b": {"name": "Beta", "provider": "Q"}}
    scores = {"a": 0.9, "b": 1.6}                       # signal statistics gave Beta the higher score
    out = finalize_attribution(scores, catalog, declared=True, declared_keys={"a"}, cues=["declares Alpha"],
                               unknown_name="Unknown", no_cue_text="none", region_of=lambda k: "X")
    assert out["attributed_model"] == "Alpha"
    undeclared = finalize_attribution(scores, catalog, declared=False, declared_keys=set(), cues=[], unknown_name="Unknown",
                                      no_cue_text="none", region_of=lambda k: "X")
    assert undeclared["attributed_model"] == "Unknown" and undeclared["top_candidates"] == []


def test_age_screen_accepts_grey_and_refuses_to_pass_an_unreadable_picture():
    import numpy as np

    from core.perception.age import AgeEstimator, _as_bgr

    assert _as_bgr(np.zeros((8, 8), np.uint8)).shape == (8, 8, 3)
    assert _as_bgr(np.zeros((8, 8, 4), np.uint8)).shape == (8, 8, 3)
    assert _as_bgr(np.zeros((0, 0, 3), np.uint8)) is None and _as_bgr(None) is None and _as_bgr(np.zeros((8, 8, 2), np.uint8)) is None
    out = AgeEstimator().assess(None)
    assert out["status"] == "NO_IMAGE" and out["review_required"] is True


def test_boxes_are_reported_on_the_original_picture():
    from image_detector.content import _scaled_box

    assert _scaled_box((10, 20, 30, 40), 2.5) == (25, 50, 75, 100)


def test_benchmark_counts_an_error_result_as_failed_not_abstained(tmp_path):
    from core.benchmark import evaluate_folder

    for sub in ("ai_generated", "real"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "a.png").write_bytes(b"x")
    report = evaluate_folder(tmp_path, [".png"], lambda p: {"error": "cannot decode"})
    assert report["failed"] == 2 and report["abstained"] == 0 and report["scored"] == 0


def test_a_zip_larger_than_the_scan_window_is_still_found(tmp_path):
    import io
    import os
    import zipfile

    from PIL import Image

    from core.forensics.bytescan import WINDOW, read_windows, scan_windows

    png = io.BytesIO()
    Image.new("RGB", (8, 8)).save(png, "PNG")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        z.writestr("big.bin", os.urandom(WINDOW + 1024 * 1024))
    path = tmp_path / "poly.png"
    path.write_bytes(png.getvalue() + archive.getvalue())
    head, tail, size = read_windows(path)
    assert scan_windows([(head, False), (tail, False)]) == ["ZIP"]


def test_two_processes_updating_one_json_file_lose_no_update(tmp_path):
    """The per-path thread lock cannot stop a second process; the process lock must."""
    import json
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    target = tmp_path / "counter.json"
    target.write_text(json.dumps({"n": 0}))
    repo = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(repo)!r})
        from core.atomic_io import atomic_update_json
        for _ in range(25):
            atomic_update_json({str(target)!r}, lambda d: {{"n": d["n"] + 1}}, default={{"n": 0}})
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script]) for _ in range(4)]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0, 0]
    assert json.loads(target.read_text())["n"] == 100


def test_the_checkpoint_log_is_appended_atomically(tmp_path):
    from core.checkpoint_log import append_row, read_last_accuracy

    log = tmp_path / "CHECKPOINT_LOG.md"
    for v in range(1, 4):
        append_row(log, v, "2026-10-10", 0.1, 0.5 + v / 10, 3, 3 * v)
    assert read_last_accuracy(log) == 0.8 and log.read_text(encoding="utf-8").count("| 2026-10-10 |") == 3
    assert not list(tmp_path.glob(".*tmp_*"))                       # no temporary file left behind
