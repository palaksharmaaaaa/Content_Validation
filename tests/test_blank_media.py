"""A file with nothing in it gets "no usable content", not an AI or real verdict."""
import cv2
import numpy as np
from PIL import Image

from audio_detector.tests.audio_fixtures import tone, write_wav
from services.forensic_service import ForensicService


def _svc():
    return ForensicService.get_instance()


def test_solid_colour_images_are_blank_not_ai_generated_or_real(tmp_path):
    for name, value in (("black", 0), ("gray", 128), ("white", 255)):
        p = tmp_path / f"{name}.png"
        Image.fromarray(np.full((300, 300, 3), value, np.uint8)).save(p)
        r = _svc().image_pipeline.analyze(p)
        assert r["final_status"] == "BLANK_OR_DEGRADED", name
        assert r["authenticity_probabilities"]["p_ai"] == 0.0 and r["authenticity_probabilities"]["p_undecided"] == 100.0


def test_a_real_looking_image_is_not_blank(tmp_path):
    p = tmp_path / "noise.png"
    Image.fromarray(np.clip(np.random.default_rng(0).normal(128, 40, (300, 300, 3)), 0, 255).astype(np.uint8)).save(p)
    assert _svc().image_pipeline.analyze(p)["final_status"] != "BLANK_OR_DEGRADED"


def test_digital_silence_is_blank_but_a_quiet_recording_is_not(tmp_path):
    silent = write_wav(tmp_path / "silent.wav", np.zeros(16000 * 2))
    assert _svc().audio_pipeline.analyze(silent)["final_status"] == "BLANK_OR_DEGRADED"
    quiet = write_wav(tmp_path / "quiet.wav", tone(2.0, noise=0.002, seed=1) * 0.02)
    assert _svc().audio_pipeline.analyze(quiet)["final_status"] != "BLANK_OR_DEGRADED"


def test_a_video_of_blank_frames_is_blank(tmp_path):
    p = tmp_path / "black.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(40):
        vw.write(np.zeros((120, 160, 3), np.uint8))
    vw.release()
    assert _svc().video_pipeline.analyze(p)["final_status"] == "BLANK_OR_DEGRADED"


def test_the_verdict_card_says_no_usable_content():
    from ui.results.summary import verdict_style

    level, headline, meaning = verdict_style({"final_status": "BLANK_OR_DEGRADED"})
    assert level == "info" and headline == "No usable content" and "no verdict" in meaning
