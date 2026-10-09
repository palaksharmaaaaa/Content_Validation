"""Integration: extra_log_lrs in the audio detector, gates + dimension report in the audio pipeline."""
import hashlib
import io
import zipfile

import numpy as np
import pytest

from audio_detector.detector import AudioAIDetector
from audio_detector.pipeline import AudioForensicPipeline
from audio_detector.tests.audio_fixtures import write_wav, write_wav_with_chunks


@pytest.fixture(scope="module")
def detector():
    d = AudioAIDetector()
    d.load()
    return d


@pytest.fixture(scope="module")
def pipeline(detector):
    return AudioForensicPipeline(detector=detector)


def _speechy(path, seconds=6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 16000)) / 16000
    x = 0.2 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.03 * rng.standard_normal(len(t))
    return write_wav(path, x.astype(np.float32))


def test_extra_log_lrs_none_and_empty_are_identity(detector, tmp_path):
    p = _speechy(tmp_path / "a.wav")
    base = detector.analyze_audio_file(str(p))
    for extra in (None, {}):
        again = detector.analyze_audio_file(str(p), extra_log_lrs=extra)
        for k in ("ai_percentage", "real_percentage", "undecided_percentage", "label"):
            assert base[k] == again[k]


def test_extra_log_lrs_shift_probability(detector, tmp_path):
    p = _speechy(tmp_path / "b.wav", seed=1)
    base = detector.analyze_audio_file(str(p))
    up = detector.analyze_audio_file(str(p), extra_log_lrs={"id3_tags": 0.40})
    down = detector.analyze_audio_file(str(p), extra_log_lrs={"enf_trace": -0.15})
    assert up["ai_percentage"] > base["ai_percentage"]
    assert down["ai_percentage"] < base["ai_percentage"]
    assert any("id3_tags" in c for c in up["forensic_cues"])


def test_pipeline_result_has_dimension_report(pipeline, tmp_path):
    res = pipeline.analyze(_speechy(tmp_path / "c.wav"))
    assert res["content_valid"] is True
    report = res["dimension_report"]
    assert {"file_integrity", "container", "signal", "context", "legal", "lifecycle", "reliability"} <= set(report["findings_by_stage"])
    assert res["confidence_band"]["band"] in {
        "HIGH_CONFIDENCE_SYNTHETIC", "LEANING_SYNTHETIC", "INCONCLUSIVE", "LEANING_AUTHENTIC", "HIGH_CONFIDENCE_AUTHENTIC"}
    assert res["ood"]["status"] == "NOT_CALIBRATED"
    assert res["gate"]["triggered"] is False


def test_generator_tag_moves_score_toward_ai(pipeline, tmp_path):
    info = b"INFO" + b"ISFT" + (len(b"ElevenLabs API\x00")).to_bytes(4, "little") + b"ElevenLabs API\x00"
    x = np.clip(0.2 * np.sin(2 * np.pi * 180 * np.arange(96000) / 16000), -1, 1).astype(np.float32)
    plain = write_wav(tmp_path / "plain.wav", x)
    tagged = write_wav_with_chunks(tmp_path / "tag.wav", x, extra={b"LIST": info + (b"\x00" if len(info) & 1 else b"")})
    a, b = pipeline.analyze(plain), pipeline.analyze(tagged)
    assert b["dimension_report"]["score_terms"]["riff_structure"] == pytest.approx(0.40)
    assert b["authenticity_probabilities"]["p_ai"] >= a["authenticity_probabilities"]["p_ai"]


def test_hard_block_short_circuits(pipeline, tmp_path, monkeypatch):
    p = _speechy(tmp_path / "blocked.wav")
    bl = tmp_path / "bl.txt"
    bl.write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    monkeypatch.setenv("OMNI_HARDBLOCK_SHA256_FILE", str(bl))
    res = pipeline.analyze(p)
    assert res["final_status"] == "HARD_BLOCK_ESCALATE"
    assert res["authenticity_probabilities"]["p_undecided"] == 100.0
    assert "dimension_report" not in res


def test_midi_is_recognized_not_scored(pipeline, tmp_path):
    p = tmp_path / "song.mid"
    p.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0")
    res = pipeline.analyze(p)
    assert res["final_status"] == "RECOGNIZED_OUT_OF_SCOPE"
    assert res["gate"]["recognition"]["type"] == "MIDI"


def test_security_findings_do_not_change_probabilities(pipeline, tmp_path):
    clean = _speechy(tmp_path / "clean.wav", seed=4)
    dirty = tmp_path / "dirty.wav"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x" * 300)
    dirty.write_bytes(clean.read_bytes() + buf.getvalue())
    a, b = pipeline.analyze(clean), pipeline.analyze(dirty)
    assert b["dimension_report"]["summary"]["by_status"].get("FAIL", 0) >= 1
    assert a["authenticity_probabilities"] == b["authenticity_probabilities"]
