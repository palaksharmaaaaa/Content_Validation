"""Full app, driven through Streamlit's AppTest: upload -> analysis -> result page, no mocks."""
import io
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from audio_detector.tests.audio_fixtures import tone, write_wav


def _png(seed: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _app() -> AppTest:
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=300).run()


def test_image_upload_renders_full_flow():
    at = _app()
    at.file_uploader[0].upload("t.png", _png(1)).run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "In plain English" in text
    assert len(at.metric) > 20


def test_audio_upload_renders_full_flow(tmp_path):
    wav = write_wav(tmp_path / "a.wav", tone(1.5, noise=0.01))
    at = _app()
    at.file_uploader[2].upload("a.wav", wav.read_bytes()).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "In plain English" in " ".join(m.value for m in at.markdown)


def test_same_name_same_size_different_bytes_is_reanalysed():
    """The cache signature is a content hash, not name+size (old audit finding M-3)."""
    a, b = _png(1), _png(2)
    assert len(a) != len(b) or a != b
    at = _app()
    at.file_uploader[0].upload("same.png", a).run()
    first = at.session_state["img_batch_sig"]
    at.file_uploader[0].upload("same.png", b).run()
    assert at.session_state["img_batch_sig"] != first


def test_video_upload_renders_full_flow(tmp_path):
    import cv2
    import pytest

    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96))
    if not w.isOpened():
        pytest.skip("OpenCV mp4v writer unavailable")
    rng = np.random.default_rng(3)
    for i in range(30):
        f = np.full((96, 128, 3), 100, np.uint8)
        cv2.rectangle(f, (i * 2, 10), (i * 2 + 20, 50), (20, 200, 90), -1)
        w.write((f + rng.integers(0, 8, f.shape, dtype=np.uint8)).astype(np.uint8))
    w.release()
    at = _app()
    at.file_uploader[1].upload("v.mp4", p.read_bytes()).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "In plain English" in " ".join(m.value for m in at.markdown)
