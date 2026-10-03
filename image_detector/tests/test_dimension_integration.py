"""Integration: extra_log_lrs in the detector, gates + dimension report in the pipeline."""
import hashlib
import struct

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from image_detector.detector import ImageAIDetector
from image_detector.pipeline import ImageForensicPipeline


@pytest.fixture(scope="module")
def detector():
    d = ImageAIDetector()
    d.load()
    return d


@pytest.fixture(scope="module")
def pipeline(detector):
    return ImageForensicPipeline(detector=detector)


def _noise_png(path, text=None, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(50, 200, (256, 256, 3), dtype=np.uint8)
    info = PngImagePlugin.PngInfo()
    for k, v in (text or {}).items():
        info.add_text(k, v)
    Image.fromarray(arr).save(path, "PNG", pnginfo=info)
    return path


def test_extra_log_lrs_none_and_empty_are_identity(detector, tmp_path):
    p = _noise_png(tmp_path / "a.png")
    base = detector.predict(str(p))
    again_none = detector.predict(str(p), extra_log_lrs=None)
    again_empty = detector.predict(str(p), extra_log_lrs={})
    for key in ("ai_percentage", "real_percentage", "undecided_percentage", "label", "taxonomy_state"):
        assert base[key] == again_none[key] == again_empty[key]
    assert base["log_likelihood_ratios"] == again_none["log_likelihood_ratios"]


def test_extra_log_lrs_shifts_probability_and_is_visible(detector, tmp_path):
    p = _noise_png(tmp_path / "b.png")
    base = detector.predict(str(p))
    boosted = detector.predict(str(p), extra_log_lrs={"png_chunks": 0.40})
    assert boosted["log_likelihood_ratios"]["dim_png_chunks"] == pytest.approx(0.4)
    assert boosted["ai_percentage"] >= base["ai_percentage"]
    lowered = detector.predict(str(p), extra_log_lrs={"exif_consistency": -0.25})
    assert lowered["ai_percentage"] <= base["ai_percentage"]


def test_pipeline_result_has_dimension_report_and_band(pipeline, tmp_path):
    p = _noise_png(tmp_path / "c.png")
    res = pipeline.analyze(p)
    assert res["content_valid"] is True
    report = res["dimension_report"]
    assert {"file_integrity", "formats", "metadata", "legal", "lifecycle", "reliability", "context"} <= set(report["findings_by_stage"])
    assert res["confidence_band"]["band"] in {
        "HIGH_CONFIDENCE_SYNTHETIC", "LEANING_SYNTHETIC", "INCONCLUSIVE", "LEANING_AUTHENTIC", "HIGH_CONFIDENCE_AUTHENTIC"}
    assert res["ood"]["status"] == "NOT_CALIBRATED"
    assert res["gate"]["triggered"] is False
    assert "attribution_open_set" in res


def test_pipeline_generator_chunk_moves_score_toward_ai(pipeline, tmp_path):
    plain = pipeline.analyze(_noise_png(tmp_path / "plain.png"))
    tagged = pipeline.analyze(_noise_png(tmp_path / "tag.png", {"parameters": "x\nSteps: 20, Sampler: Euler, CFG scale: 7"}))
    assert tagged["dimension_report"]["score_terms"]["png_chunks"] == pytest.approx(0.40)
    assert tagged["authenticity_probabilities"]["p_ai"] >= plain["authenticity_probabilities"]["p_ai"]
    assert any("png_chunks" in k for k in tagged["ai_detection"]["log_likelihood_ratios"] if k.startswith("dim_")) or \
        "dim_png_chunks" in tagged["ai_detection"]["log_likelihood_ratios"]


def test_pipeline_hard_block_short_circuits(pipeline, tmp_path, monkeypatch):
    p = _noise_png(tmp_path / "blocked.png")
    bl = tmp_path / "bl.txt"
    bl.write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "\n")
    monkeypatch.setenv("OMNI_HARDBLOCK_SHA256_FILE", str(bl))
    res = pipeline.analyze(p)
    assert res["final_status"] == "HARD_BLOCK_ESCALATE"
    assert res["gate"]["hard_block"]["triggered"] is True
    assert res["authenticity_probabilities"]["p_undecided"] == 100.0
    assert "dimension_report" not in res


def test_pipeline_recognizes_fits_without_verdict(pipeline, tmp_path):
    p = tmp_path / "sky.fits"
    p.write_bytes(b"SIMPLE  =                    T" + b" " * 200)
    res = pipeline.analyze(p)
    assert res["final_status"] == "RECOGNIZED_OUT_OF_SCOPE"
    assert res["gate"]["recognition"]["type"] == "FITS"
    assert res["ai_detected"] is False


def test_security_findings_do_not_change_probabilities(pipeline, tmp_path):
    import zipfile
    import io

    clean = _noise_png(tmp_path / "clean.png")
    dirty = tmp_path / "dirty.png"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x" * 200)
    dirty.write_bytes(clean.read_bytes() + buf.getvalue())
    a, b = pipeline.analyze(clean), pipeline.analyze(dirty)
    assert b["dimension_report"]["summary"]["by_status"].get("FAIL", 0) >= 1
    assert a["authenticity_probabilities"] == b["authenticity_probabilities"]
