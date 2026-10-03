"""Golden for the Streamlit renderers: the element tree (types and text) of every stage for image/audio/video items."""
import hashlib
import json
import pytest
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from audio_detector import AudioContentAnalyzer, AudioModelAttributionEngine
from audio_detector.tests.audio_fixtures import tone, write_wav
from services.forensic_service import ForensicService
from tests.golden_support import skip_unless_state_matches
from ui.batch_ui import process_single_audio, process_single_image, process_single_video


def r_image(item):
    from ui.feedback_ui import render_linear_image_pipeline_results
    render_linear_image_pipeline_results(item)


def r_audio(item):
    from ui.feedback_ui import render_linear_audio_pipeline_results
    render_linear_audio_pipeline_results(item)


def r_video(item):
    from ui.feedback_ui import render_linear_video_pipeline_results
    render_linear_video_pipeline_results(item)


def r_decision(item):
    from ui.feedback_ui import render_analysis_right_panel
    render_analysis_right_panel(item["decision"], item.get("content_res", {}), modality=item["modality"])


def r_dossier(item):
    from ui.feedback_ui import render_forensic_dossier
    render_forensic_dossier(item["decision"])


def dump(node, out, depth=0):
    kids = getattr(node, "children", None)
    t = getattr(node, "type", type(node).__name__)
    val = None
    for attr in ("value", "label", "body", "text"):
        v = getattr(node, attr, None)
        if isinstance(v, (str, int, float, bool)) and v != "":
            val = v
            break
    if t not in ("root", "main", "sidebar", "block") or val is not None:
        out.append(f"{'  ' * depth}{t}:{val}")
    if kids:
        items = kids.values() if isinstance(kids, dict) else kids
        for k in items:
            dump(k, out, depth + 1)


def render(fn, item, tmp):
    at = AppTest.from_function(fn, args=(item,), default_timeout=240).run()
    lines = []
    dump(at.main, lines)
    if at.exception:
        lines.append("EXCEPTION:" + " | ".join(str(e.value) for e in at.exception))
    text = "\n".join(lines)
    raw = str(tmp)
    for variant in (raw, raw.replace("\\", "\\\\"), raw.replace("\\", "/")):
        text = text.replace(variant, "<TMP>")
    text = re.sub(r"tmp[0-9a-z_]{8}", "<TMP>", text)
    return text



def test_ui_renders_match_golden(tmp_path):
    skip_unless_state_matches(
        "ui_render_golden.meta.json",
        ["image_detector/models/ai_detector.pt", "image_detector/data/image_calibration.json",
         "audio_detector/data/audio_calibration.json", "video_detector/data/video_calibration.json"],
    )
    expected = json.loads((Path(__file__).resolve().parents[1] / "tests" / "data" / "ui_render_golden.json").read_text(encoding="utf-8"))
    tmp = tmp_path

    rng = np.random.default_rng(12)
    svc = ForensicService.get_instance()
    ex = Image.Exif(); ex[271] = "Canon"; ex[272] = "Canon EOS R5"
    Image.fromarray(rng.integers(0, 255, (256, 320, 3), dtype=np.uint8)).save(tmp / "a.jpg", "JPEG", quality=90, exif=ex)
    Image.fromarray(np.full((512, 512, 3), 120, np.uint8)).save(tmp / "b.png")
    items = {}
    for name in ("a.jpg", "b.png"):
        items["img_" + name] = ("image", process_single_image(str(tmp / name), name, svc.image_detector, svc.content_analyzer, svc.attribution_engine))
    w = write_wav(tmp / "a.wav", tone(2.0, noise=0.02, seed=3))
    items["aud"] = ("audio", process_single_audio(str(w), "a.wav", svc.audio_detector, AudioContentAnalyzer(), AudioModelAttributionEngine()))
    vw = cv2.VideoWriter(str(tmp / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96))
    for i in range(30):
        f = np.full((96, 128, 3), 100, np.uint8); cv2.rectangle(f, (i * 2, 10), (i * 2 + 20, 50), (20, 200, 90), -1)
        vw.write((f + rng.integers(0, 8, f.shape, dtype=np.uint8)).astype(np.uint8))
    vw.release()
    items["vid"] = ("video", process_single_video(str(tmp / "v.mp4"), "v.mp4", svc.image_detector, svc.content_analyzer, svc.audio_detector,
                                                  svc.attribution_engine, cache_dir=tmp, video_detector=svc.video_detector))
    res = {}
    fns = {"image": r_image, "audio": r_audio, "video": r_video}
    for k, (mod, item) in items.items():
        res["linear_" + k] = render(fns[mod], item, tmp)
        d = dict(item); d["modality"] = mod
        res["decision_" + k] = render(r_decision, d, tmp)
        res["dossier_" + k] = render(r_dossier, d, tmp)
    assert not [k for k, v in res.items() if "EXCEPTION" in v]
    assert res == expected
