"""Regression tests for the verified findings of the old brutal audit (each test names the finding it pins)."""
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

from audio_detector.features import compute_spectral_features
from audio_detector.learner import AudioSelfImprover
from audio_detector.models.backbone import AudioClassifierNet, build_audio_classifier
from audio_detector.profiler import AudioProfiler
from audio_detector.tests.audio_fixtures import tone, write_wav
from audio_detector.validator import resample_antialiased
from core import atomic_io
from core.decision import normalize_percentages


def _bandlimited_noise(sr: int, cutoff_hz: float, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal(int(sr * seconds))
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(len(x), 1.0 / sr) > cutoff_hz] = 0
    y = np.fft.irfft(spec, n=len(x))
    return (0.2 * y / np.max(np.abs(y))).astype(np.float32)


# ---- C-5: the checkpoint key the trainer writes must load
def test_audio_backbone_loads_trainer_checkpoint(tmp_path):
    net = AudioClassifierNet()
    with torch.no_grad():
        net.net[0].weight.fill_(0.5)
    ckpt = tmp_path / "audio.pt"
    torch.save({"model_state_dict": net.state_dict(), "in_features": 5, "class_to_idx": {"ai": 0, "real": 1}}, ckpt)
    loaded = build_audio_classifier(checkpoint_path=ckpt, device=torch.device("cpu"))
    assert torch.allclose(loaded.net[0].weight, torch.full_like(loaded.net[0].weight, 0.5))


# ---- C-4: the image category page must not crash without a precomputed dossier
def test_image_category_page_renders_without_nine_dimensions():
    from streamlit.testing.v1 import AppTest

    def page():
        from ui.feedback_ui import render_image_type_and_category

        render_image_type_and_category({"taxonomy_state": "FULLY_AI_GENERATED"}, {}, {}, nine_dims=None)

    at = AppTest.from_function(page, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


# ---- H-2: an anti-aliasing roll-off at Nyquist is not a vocoder brick wall
def test_nyquist_edge_cutoff_is_not_a_vocoder():
    edge = compute_spectral_features(_bandlimited_noise(16000, 7900), 16000)
    assert not edge["has_vocoder_cutoff"]
    mid = compute_spectral_features(_bandlimited_noise(44100, 7500), 44100)
    assert mid["has_vocoder_cutoff"]


# ---- H-3: the undecided margin must respond to how close ai/real are
def test_undecided_margin_scales_with_gap():
    close = normalize_percentages(51.0, 49.0)
    far = normalize_percentages(95.0, 5.0)
    assert close[2] > far[2] + 5.0
    assert sum(close) == pytest.approx(100.0, abs=0.2)


# ---- H-5/H-6: model loading runs once even under contention
def test_detector_load_is_serialised(monkeypatch):
    from image_detector.detector import ImageAIDetector

    det = ImageAIDetector()
    det._is_loaded = False
    active, overlaps = [0], [0]

    def slow_load():
        active[0] += 1
        overlaps[0] = max(overlaps[0], active[0])
        time.sleep(0.05)
        det._is_loaded = True
        active[0] -= 1
        return True

    monkeypatch.setattr(det, "_load_locked", slow_load)
    threads = [threading.Thread(target=det.load) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert overlaps[0] == 1


# ---- H-7: concurrent feedback must not lose entries
def test_concurrent_feedback_loses_nothing(tmp_path):
    imp = AudioSelfImprover(memory_file=tmp_path / "mem.json", calibration_file=tmp_path / "cal.json")
    wavs = [write_wav(tmp_path / f"{i}.wav", tone(0.5, seed=i)) for i in range(8)]
    threads = [threading.Thread(target=imp.record_feedback, args=(str(w), "REAL", {"x": 1.0})) for w in wavs]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(imp.load_memory()) == 8
    assert imp.load_calibration()["samples_processed"] == 8


# ---- H-8: the lock registry must not grow forever
def test_path_lock_registry_is_bounded(tmp_path):
    p = tmp_path / "x.json"
    lock = atomic_io._get_path_lock(p)
    assert atomic_io._get_path_lock(p) is lock  # shared while someone holds it
    key = str(p.resolve()).lower()
    ref = weakref.ref(lock)
    del lock
    assert ref() is None and key not in atomic_io._FILE_LOCKS


# ---- H-9: downsampling must not alias
def test_resampling_does_not_alias():
    sr_in, sr_out = 44100, 16000
    t = np.arange(sr_in) / sr_in
    x = np.sin(2 * np.pi * 12000 * t).astype(np.float32)  # above the new 8 kHz Nyquist: must vanish, not fold to 4 kHz
    y = resample_antialiased(x, sr_in, sr_out)
    naive = np.interp(np.linspace(0, len(x) - 1, len(y)), np.arange(len(x)), x)
    assert np.sqrt(np.mean(y ** 2)) < 0.1 * np.sqrt(np.mean(naive ** 2))


# ---- H-10: moov at the end of a non-fast-start MP4 is still seen
def test_video_atoms_found_in_file_tail(tmp_path):
    from video_detector.provenance import VideoProvenanceValidator

    p = tmp_path / "tail.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 300_000 + b"\x00\x00\x00\x08moov")
    assert "moov" in VideoProvenanceValidator().analyze_provenance(p)["container_atoms"]


# ---- M-8: OpenCV frame count of -1 must not give a negative duration
def test_negative_frame_count_gives_zero_duration(monkeypatch, tmp_path):
    import cv2

    from video_detector.extractor import VideoFrameExtractor

    class FakeCap:
        def isOpened(self):
            return True

        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 25.0, cv2.CAP_PROP_FRAME_COUNT: -1.0}.get(prop, 0.0)

        def release(self):
            pass

    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: FakeCap())
    meta = VideoFrameExtractor.get_video_metadata(f)
    assert meta["duration_seconds"] == 0.0 and meta["total_frames"] == 0


# ---- M-9: URL downloads can be placed in the session scratch dir
def test_url_download_honours_dest_dir(monkeypatch, tmp_path):
    from ui import validators

    seen = {}

    class FakeFetcher:
        def __init__(self, **_):
            pass

        def fetch(self, url, dest_dir=None, expected_type="image"):
            seen["dest_dir"] = dest_dir
            return {"success": True, "file_path": "x", "filename": "x", "size_mb": 0, "content_type": ""}

    monkeypatch.setattr(validators, "SecureUrlFetcher", FakeFetcher)
    validators.fetch_media_from_url("https://example.com/a.jpg", dest_dir=tmp_path)
    assert seen["dest_dir"] == tmp_path


# ---- L-4: dynamic range is a level spread, not the peak/RMS crest factor
def test_profiler_separates_dynamic_range_from_crest_factor(tmp_path):
    quiet_loud = np.concatenate([tone(1.0, amp=0.02), tone(1.0, amp=0.6)])
    steady = tone(2.0, amp=0.3)
    prof = AudioProfiler()
    a = prof.profile_audio(write_wav(tmp_path / "a.wav", quiet_loud))
    b = prof.profile_audio(write_wav(tmp_path / "b.wav", steady))
    assert a["dynamic_range_db"] > b["dynamic_range_db"] + 10
    assert "crest_factor_db" in a


# ---- found while verifying M-11: "udio" matched inside the word "audio" and attributed everything to Udio
def test_short_vendor_names_match_whole_words_only(tmp_path):
    from audio_detector.attribution import AudioModelAttributionEngine

    f = write_wav(tmp_path / "a.wav", tone(1.0))
    engine = AudioModelAttributionEngine()
    plain = engine.attribute_audio(f, provenance_data={"metadata": {"encoder": "Audacity audio editor", "n": "studio audio"}})
    assert plain["model_key"] != "udio"
    real = engine.attribute_audio(f, provenance_data={"metadata": {"generator": "Made with Udio"}})
    assert real["model_key"] == "udio"
