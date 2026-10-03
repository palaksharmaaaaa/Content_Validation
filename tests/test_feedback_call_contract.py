"""The UI must pass free-text notes by keyword: audio's 4th positional parameter is voice_generator_tag."""
import ast
import json
from pathlib import Path

from audio_detector import AudioSelfImprover
from audio_detector.tests.audio_fixtures import tone, write_wav


def test_ui_passes_notes_by_keyword():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "ui" / "feedback_ui.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "record_feedback"]
    assert calls, "expected record_feedback calls in ui/feedback_ui.py"
    for c in calls:
        assert len(c.args) <= 3, "notes must not be passed positionally"
        assert any(k.arg == "notes" for k in c.keywords)


def test_audio_notes_are_not_mistaken_for_generator_tag(tmp_path):
    wav = write_wav(tmp_path / "a.wav", tone(1.0))
    mem = tmp_path / "mem.json"
    imp = AudioSelfImprover(memory_file=mem, calibration_file=tmp_path / "cal.json")
    imp.record_feedback(str(wav), "REAL", {"x": 1.0}, notes="plain note")
    rec = json.loads(mem.read_text(encoding="utf-8"))[-1]
    assert rec["notes"] == "plain note"


def test_all_learners_accept_modality_neutral_metrics(tmp_path):
    from image_detector import ImageSelfImprover
    from video_detector import VideoSelfImprover

    wav = write_wav(tmp_path / "m.wav", tone(1.0))
    a = AudioSelfImprover(memory_file=tmp_path / "a.json", calibration_file=tmp_path / "ac.json")
    assert a.record_feedback(str(wav), "REAL", metrics={"x": 1.0}) is not None
    img = tmp_path / "m.jpg"
    from PIL import Image
    Image.new("RGB", (32, 32), (5, 5, 5)).save(img)
    i = ImageSelfImprover(memory_file=tmp_path / "i.json", calibration_file=tmp_path / "ic.json")
    assert i.record_feedback(str(img), "REAL", metrics={"x": 1.0}) is not None
    v = VideoSelfImprover(memory_file=tmp_path / "v.json", calibration_file=tmp_path / "vc.json")
    assert v.record_feedback(str(wav), "REAL", metrics={"x": 1.0}) is not None


def test_submit_feedback_routes_to_owning_learner(monkeypatch, tmp_path):
    import ui.feedback_ui as fu

    calls = []

    class Fake:
        def __init__(self, tag):
            self.tag = tag

        def record_feedback(self, path, label, metrics, notes=""):
            calls.append((self.tag, Path(path).name, label, notes))
            return {"samples_processed": 1}

    monkeypatch.setattr(fu, "ImageSelfImprover", lambda: Fake("image"))
    monkeypatch.setattr(fu, "VideoSelfImprover", lambda: Fake("video"))
    monkeypatch.setattr(fu, "AudioSelfImprover", lambda: Fake("audio"))
    monkeypatch.setattr(fu.st, "success", lambda *a, **k: None)
    monkeypatch.setattr(fu.st, "info", lambda *a, **k: None)
    ratings = {"real_score": 5, "synthetic_score": 8, "forged_score": 3, "undetected_score": 2}
    for media, modality, tag in (("a.JPG", "auto", "image"), ("b.mp4", "auto", "video"), ("c.wav", "auto"
                                 , "audio"), ("d.bin", "image", "image")):
        fu._submit_feedback(tmp_path / media, modality, {"x": 1}, ratings, "Real / Camera Authentic", "gen", "note")
    assert [c[0] for c in calls] == ["image", "video", "audio", "image"]
    assert all(c[2] == "AI" for c in calls)  # synthetic score >= 5.5 forces AI
    assert "note" in calls[0][3] and "Gen: gen" in calls[0][3]
