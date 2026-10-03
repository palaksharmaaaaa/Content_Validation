"""End-to-end UI flow: real pipelines on synthetic media, rendered through Streamlit's AppTest (no mocks)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from audio_detector import AudioContentAnalyzer, AudioModelAttributionEngine
from audio_detector.tests.audio_fixtures import tone, write_wav
from services.forensic_service import ForensicService
from ui.batch_ui import process_single_audio, process_single_image, process_single_video

CHECKPOINT = Path(__file__).resolve().parents[1] / "image_detector" / "models" / "ai_detector.pt"


@pytest.fixture(scope="module")
def service():
    return ForensicService.get_instance(checkpoint_path=CHECKPOINT)


def _render_image(item):
    from ui.feedback_ui import render_linear_image_pipeline_results

    render_linear_image_pipeline_results(item)


def _render_audio(item):
    from ui.feedback_ui import render_linear_audio_pipeline_results

    render_linear_audio_pipeline_results(item)


def _run(fn, item):
    at = AppTest.from_function(fn, args=(item,), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_image_flow_renders_every_stage(service, tmp_path):
    p = tmp_path / "flow.jpg"
    Image.fromarray(np.random.default_rng(5).integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(p, "JPEG", quality=90)
    item = process_single_image(str(p), "flow.jpg", service.image_detector, service.content_analyzer, service.attribution_engine)
    assert item["success"] and item["decision"]["calibration_status"] == "UNCALIBRATED_HEURISTIC"
    assert item["decision"]["decision_mode"] == "image_authoritative"
    at = _run(_render_image, item)
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Stage 1b" in text and "uncalibrated" in text.lower()


def test_audio_flow_renders_every_stage(service, tmp_path):
    p = write_wav(tmp_path / "flow.wav", tone(2.0, noise=0.01))
    item = process_single_audio(str(p), "flow.wav", service.audio_detector, AudioContentAnalyzer(), AudioModelAttributionEngine())
    assert item["success"]
    at = _run(_render_audio, item)
    text = " ".join(m.value for m in at.markdown)
    assert "Stage 1b" in text


def test_hardblock_gate_short_circuits_ui(service, tmp_path, monkeypatch):
    import hashlib

    p = tmp_path / "blocked.jpg"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(p, "JPEG")
    hashes = tmp_path / "hashes.txt"
    hashes.write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    monkeypatch.setenv("OMNI_HARDBLOCK_SHA256_FILE", str(hashes))
    item = process_single_image(str(p), "blocked.jpg", service.image_detector, service.content_analyzer, service.attribution_engine)
    assert item.get("gate_blocked") is True
    at = _run(_render_image, item)
    assert not at.exception


def _render_video(item):
    from ui.feedback_ui import render_linear_video_pipeline_results

    render_linear_video_pipeline_results(item)


def test_video_flow_renders_and_uses_video_verdict(service, tmp_path):
    import cv2

    p = tmp_path / "flow.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    if not w.isOpened():
        pytest.skip("OpenCV mp4v writer unavailable")
    rng = np.random.default_rng(1)
    for _ in range(30):
        w.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    w.release()
    item = process_single_video(
        str(p), "flow.mp4", service.image_detector, service.content_analyzer, service.audio_detector,
        service.attribution_engine, cache_dir=tmp_path, video_detector=service.video_detector,
    )
    assert item["success"]
    d = item["decision"]
    assert d["decision_mode"] == "fused"
    vid_ai = item["video_result"]["ai_percentage"]
    # The verdict must follow the video detector (not a keyframe): same side of 50% unless the video is near-balanced.
    if abs(vid_ai - 50.0) > 10.0:
        assert (d["authenticity_probabilities"]["p_ai"] > 50.0) == (vid_ai > 50.0)
    _run(_render_video, item)


def test_package_pipeline_and_ui_agree_on_image_verdict(service, tmp_path):
    p = tmp_path / "parity.jpg"
    Image.fromarray(np.random.default_rng(9).integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(p, "JPEG", quality=90)
    ui = process_single_image(str(p), "parity.jpg", service.image_detector, service.content_analyzer, service.attribution_engine)
    pipe = service.image_pipeline.analyze(p)
    assert pipe["decision"]["final_status"] == ui["decision"]["final_status"] == pipe["final_status"]
    assert pipe["authenticity_probabilities"] == ui["decision"]["authenticity_probabilities"]


def test_package_pipeline_and_ui_agree_on_audio_verdict(service, tmp_path):
    p = write_wav(tmp_path / "parity.wav", tone(2.0, noise=0.02, seed=4))
    ui = process_single_audio(str(p), "parity.wav", service.audio_detector, AudioContentAnalyzer(), AudioModelAttributionEngine())
    pipe = service.audio_pipeline.analyze(p)
    assert pipe["final_status"] == ui["decision"]["final_status"]
    assert pipe["authenticity_probabilities"] == ui["decision"]["authenticity_probabilities"]


def test_package_pipeline_and_ui_agree_on_video_verdict(service, tmp_path):
    import cv2

    p = tmp_path / "parity.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96))
    if not w.isOpened():
        pytest.skip("OpenCV mp4v writer unavailable")
    rng = np.random.default_rng(2)
    for i in range(30):
        f = np.full((96, 128, 3), 100, np.uint8)
        cv2.rectangle(f, (i * 2, 10), (i * 2 + 20, 50), (20, 200, 90), -1)
        w.write((f + rng.integers(0, 8, f.shape, dtype=np.uint8)).astype(np.uint8))
    w.release()
    ui = process_single_video(
        str(p), "parity.mp4", service.image_detector, service.content_analyzer, service.audio_detector,
        service.attribution_engine, cache_dir=tmp_path, video_detector=service.video_detector,
    )
    pipe = service.video_pipeline.analyze(p)
    assert pipe["final_status"] == ui["decision"]["final_status"]
    assert pipe["authenticity_probabilities"] == ui["decision"]["authenticity_probabilities"]
