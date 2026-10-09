"""Golden for the three headless pipelines (image jpg/png, audio wav, video mp4): the complete report dicts."""
import json
import pytest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from audio_detector.tests.audio_fixtures import tone, write_wav
from services.forensic_service import ForensicService
from tests.golden_support import assert_golden, scrub_encoding, skip_unless_state_matches


def norm(o):
    return json.loads(json.dumps(o, sort_keys=True, default=lambda x: f"<{type(x).__name__}>"))


def strip(d):
    d = dict(d)
    for k in ("heatmap_rgb", "spectrogram_image", "keyframes"):
        d.pop(k, None)
    return d



def test_headless_pipelines_match_golden(tmp_path):
    skip_unless_state_matches(
        "pipelines_golden.meta.json",
        ["image_detector/models/ai_detector.pt", "image_detector/data/image_calibration.json",
         "audio_detector/data/audio_calibration.json", "video_detector/data/video_calibration.json"],
    )
    tmp = tmp_path

    rng = np.random.default_rng(12)
    svc = ForensicService.get_instance()
    res = {}
    ex = Image.Exif(); ex[271] = "Canon"; ex[272] = "Canon EOS R5"
    arr = rng.integers(0, 255, (256, 320, 3), dtype=np.uint8)
    Image.fromarray(arr).save(tmp / "a.jpg", "JPEG", quality=90, exif=ex)
    Image.fromarray(np.full((512, 512, 3), 120, np.uint8)).save(tmp / "b.png")
    for name in ("a.jpg", "b.png"):
        r = strip(svc.image_pipeline.analyze(tmp / name))
        r.pop("path", None)
        res["img_" + name] = norm(r)
    m = svc.image_pipeline.analyze(tmp / "none.png"); m["reason"] = "missing"
    res["img_missing"] = norm(m)
    w = write_wav(tmp / "a.wav", tone(2.0, noise=0.02, seed=3))
    r = strip(svc.audio_pipeline.analyze(w)); r.pop("path", None)
    res["aud"] = norm(r)
    vw = cv2.VideoWriter(str(tmp / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96))
    for i in range(30):
        f = np.full((96, 128, 3), 100, np.uint8); cv2.rectangle(f, (i * 2, 10), (i * 2 + 20, 50), (20, 200, 90), -1)
        vw.write((f + rng.integers(0, 8, f.shape, dtype=np.uint8)).astype(np.uint8))
    vw.release()
    r = strip(svc.video_pipeline.analyze(tmp / "v.mp4")); r.pop("path", None)
    res["vid"] = norm(r)
    assert_golden("pipelines_golden", scrub_encoding(res))
