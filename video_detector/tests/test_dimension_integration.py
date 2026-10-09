"""Integration: extra_log_lrs in the video detector, gates + dimension report in the video pipeline."""
import hashlib
import io
import shutil
import subprocess
import zipfile

import pytest

from video_detector.detector import VideoAIDetector
from video_detector.pipeline import VideoForensicPipeline
from video_detector.tests.video_fixtures import moving_square_frames, write_clip

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def detector():
    d = VideoAIDetector()
    d.load()
    return d


@pytest.fixture(scope="module")
def pipeline(detector):
    return VideoForensicPipeline(detector=detector)


def _clip(tmp_path, name="c.mp4", n=48, step=3):
    return write_clip(tmp_path / name, moving_square_frames(n, step=step), fps=25)


def test_extra_log_lrs_none_and_empty_are_identity(detector, tmp_path):
    p = _clip(tmp_path)
    base = detector.analyze_video(str(p))
    for extra in (None, {}):
        again = detector.analyze_video(str(p), extra_log_lrs=extra)
        for k in ("ai_percentage", "real_percentage", "undecided_percentage", "label"):
            assert base[k] == again[k]


def test_extra_log_lrs_shift_probability(detector, tmp_path):
    p = _clip(tmp_path, "b.mp4")
    base = detector.analyze_video(str(p))
    up = detector.analyze_video(str(p), extra_log_lrs={"mp4_metadata": 0.40})
    down = detector.analyze_video(str(p), extra_log_lrs={"x": -0.25})
    assert up["ai_percentage"] > base["ai_percentage"]
    assert down["ai_percentage"] < base["ai_percentage"]
    assert any("mp4_metadata" in c for c in up["forensic_cues"])


def test_pipeline_result_has_dimension_report(pipeline, tmp_path):
    res = pipeline.analyze(_clip(tmp_path, "d.mp4"))
    assert res["content_valid"] is True
    report = res["dimension_report"]
    assert {"file_integrity", "container", "signal", "context", "legal", "lifecycle", "reliability"} <= set(report["findings_by_stage"])
    assert res["confidence_band"]["band"] in {
        "HIGH_CONFIDENCE_SYNTHETIC", "LEANING_SYNTHETIC", "INCONCLUSIVE", "LEANING_AUTHENTIC", "HIGH_CONFIDENCE_AUTHENTIC"}
    assert res["ood"]["status"] == "NOT_CALIBRATED"
    assert res["gate"]["triggered"] is False


@needs_ffmpeg
def test_real_ffmpeg_comment_tag_scores_toward_ai(pipeline, tmp_path):
    src = _clip(tmp_path, "src.mp4")
    plain = tmp_path / "plain.mp4"
    tagged = tmp_path / "tagged.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c", "copy", str(plain)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c", "copy", "-metadata", "comment=Created with Kling AI",
                    str(tagged)], check=True)
    a, b = pipeline.analyze(plain), pipeline.analyze(tagged)
    assert b["dimension_report"]["score_terms"]["mp4_metadata"] == pytest.approx(0.40)
    assert b["authenticity_probabilities"]["p_ai"] >= a["authenticity_probabilities"]["p_ai"]


def test_hard_block_short_circuits(pipeline, tmp_path, monkeypatch):
    p = _clip(tmp_path, "blocked.mp4")
    bl = tmp_path / "bl.txt"
    bl.write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    monkeypatch.setenv("OMNI_HARDBLOCK_SHA256_FILE", str(bl))
    res = pipeline.analyze(p)
    assert res["final_status"] == "HARD_BLOCK_ESCALATE"
    assert res["authenticity_probabilities"]["p_undecided"] == 100.0
    assert "dimension_report" not in res


def test_event_stream_recognized_not_scored(pipeline, tmp_path):
    p = tmp_path / "events.mp4"
    p.write_bytes(b"#!AER-DAT3.1\r\n" + b"\x00" * 100)
    res = pipeline.analyze(p)
    assert res["final_status"] == "RECOGNIZED_OUT_OF_SCOPE" and res["gate"]["recognition"]["type"] == "AEDAT"


def test_security_findings_do_not_change_probabilities(pipeline, tmp_path):
    clean = _clip(tmp_path, "clean.mp4")
    dirty = tmp_path / "dirty.mp4"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x" * 300)
    dirty.write_bytes(clean.read_bytes() + buf.getvalue())
    a, b = pipeline.analyze(clean), pipeline.analyze(dirty)
    assert b["dimension_report"]["summary"]["by_status"].get("FAIL", 0) >= 1
    assert a["authenticity_probabilities"] == b["authenticity_probabilities"]
