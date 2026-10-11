"""The calibration command-line tool, and the damping of feedback steps."""
import json

from core.calibration_io import damped


def test_calibration_cli_scores_the_library_and_writes_a_report(tmp_path, monkeypatch):
    from services import calibration_cli

    files = []
    for i, label in enumerate(("ai_generated", "real", "ai_generated", "real")):
        p = tmp_path / f"{i}.bin"
        p.write_bytes(b"x")
        files.append((f"sha{i}", p, label))

    class FakeLibrary:
        def samples(self, split=None):
            return files

    monkeypatch.setattr(calibration_cli, "library_for", lambda modality: FakeLibrary())
    monkeypatch.setattr(calibration_cli, "_scorer", lambda modality: (lambda p: 90.0 if int(p.stem) % 2 == 0 else 10.0))
    out = tmp_path / "report.json"
    report = calibration_cli.main(["--modality", "image", "--split", "all", "--out", str(out)])
    assert report["n"] == 4 and report["accuracy"] == 1.0 and report["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert json.loads(out.read_text())["n"] == 4


def test_calibration_cli_skips_an_unscoreable_file_and_says_so(tmp_path, monkeypatch, capsys):
    from services import calibration_cli

    good, bad = tmp_path / "good.bin", tmp_path / "bad.bin"
    good.write_bytes(b"x")
    bad.write_bytes(b"x")

    def scorer(p):
        if p.name == "bad.bin":
            raise ValueError("cannot decode")
        return 80.0

    pairs = calibration_cli.collect_pairs.__wrapped__ if hasattr(calibration_cli.collect_pairs, "__wrapped__") else calibration_cli.collect_pairs
    monkeypatch.setattr(calibration_cli, "library_for", lambda m: type("L", (), {"samples": lambda self, split=None: [("a", good, "ai_generated"), ("b", bad, "real")]})())
    assert pairs("image", "all", 0, scorer) == [("ai_generated", 80.0)]
    assert "cannot decode" in capsys.readouterr().err


def test_later_corrections_move_the_thresholds_less():
    assert damped(0.03, 0) == 0.03
    assert damped(0.03, 10) == 0.0075                      # a quarter after ten corrections
    assert damped(0.03, 500) < damped(0.03, 50) < damped(0.03, 0)
    assert damped(0.03, -5) == 0.03


def test_many_wrong_submissions_cannot_pin_a_threshold_quickly(tmp_path):
    from image_detector.learner import ImageSelfImprover

    learner = ImageSelfImprover(memory_file=tmp_path / "m.json", calibration_file=tmp_path / "c.json")
    for i in range(300):
        f = tmp_path / f"{i}.jpg"
        f.write_bytes(i.to_bytes(4, "little") * 16)
        cal = learner.record_feedback(str(f), "AI", {"noise_residual_mean": 1.0})
    # undamped, 13 corrections already reach the 0.60 clamp; damped, even 300 stay below it (the steps sum to at most 0.02 * 10)
    assert 0.35 < cal["feature_weights"]["noise_residual"] < 0.56


def test_a_reset_sets_the_learned_file_aside_and_the_defaults_return(tmp_path):
    from core.calibration_io import set_aside_calibration
    from image_detector.learner import ImageSelfImprover

    f = tmp_path / "x.jpg"
    f.write_bytes(b"abc")
    learner = ImageSelfImprover(memory_file=tmp_path / "m.json", calibration_file=tmp_path / "c.json")
    learned = learner.record_feedback(str(f), "AI", {"noise_residual_mean": 1.0})
    assert learned["feature_weights"]["noise_residual"] > 0.35
    aside = set_aside_calibration(learner.calibration_file)
    assert aside is not None and aside.is_file() and not learner.calibration_file.exists()      # kept, not deleted
    assert learner.load_calibration()["feature_weights"]["noise_residual"] == 0.35
    assert len(learner.load_memory()) == 1                                                       # the recorded correction stays
    assert set_aside_calibration(learner.calibration_file) is None                              # nothing left to set aside
