"""What the result pages and explanations say about a file must be what the file is: real values, never zero or invented defaults."""
import wave

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from audio_detector.profiler import AudioProfiler, read_stream_info
from video_detector.profiler import VideoProfiler


def _video(tmp_path, w=640, h=360, fps=25, seconds=5):
    p = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(1)
    for i in range(fps * seconds):
        f = np.full((h, w, 3), 90, np.uint8)
        cv2.rectangle(f, (i * 3 % (w - 60), 50), (i * 3 % (w - 60) + 60, 150), (30, 200, 90), -1)
        vw.write((f + rng.integers(0, 10, f.shape, dtype=np.uint8)).astype(np.uint8))
    vw.release()
    return p


def _stereo_wav(tmp_path, rate=44100, bits=16, seconds=2.0):
    p = tmp_path / "s.wav"
    t = np.arange(int(rate * seconds)) / rate
    mono = (0.3 * np.sin(2 * np.pi * 440 * t) * (2 ** (bits - 1) - 1)).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(bits // 8)
        w.setframerate(rate)
        w.writeframes(np.column_stack([mono, mono]).tobytes())
    return p


def test_video_profile_carries_the_nested_blocks_every_consumer_reads(tmp_path):
    p = VideoProfiler().profile_video(_video(tmp_path))
    assert p["geometry"] == {"width": 640, "height": 360, "fps": 25.0, "duration_seconds": 5.0, "total_frames": 125, "aspect_ratio": 1.78}
    assert p["codec_and_container"]["container"] == "MP4" and p["codec_and_container"]["bitrate_kbps"] > 0
    assert p["cryptographic_hashes"]["sha256"] == p["sha256"]


def test_video_explanation_and_dossier_state_the_real_numbers(tmp_path):
    from services.forensic_service import ForensicService

    r = ForensicService.get_instance().video_pipeline.analyze(_video(tmp_path))
    text = r["newbie_explanation"]
    assert "5.0-second" in text and "640 × 360" in text and "25.0 frames per second" in text
    d2 = r["nine_dimensions_dossier"]["dimension_2"]
    assert d2["dimensions"] == "640 x 360 px" and d2["frame_rate"] == "25.00 FPS" and "125 frames" in d2["duration"]


def test_wav_stream_facts_come_from_the_header_not_from_the_16k_decode(tmp_path):
    p = _stereo_wav(tmp_path, rate=44100, bits=16)
    info = read_stream_info(p)
    assert info["sample_rate"] == 44100 and info["channels"] == 2 and info["bit_depth"] == 16
    prof = AudioProfiler().profile_audio(p)
    assert prof["sample_rate"] == 16000 and prof["native_sample_rate"] == 44100            # decode rate vs the file's own
    assert prof["channels"] == 2 and prof["bit_depth"] == 16 and prof["geometry"]["channels"] == 2


def test_audio_explanation_and_dossier_state_the_real_numbers(tmp_path):
    from services.forensic_service import ForensicService

    r = ForensicService.get_instance().audio_pipeline.analyze(_stereo_wav(tmp_path, rate=44100, bits=16))
    assert "2.0-second" in r["newbie_explanation"] and "44,100 Hz" in r["newbie_explanation"]
    d2 = r["nine_dimensions_dossier"]["dimension_2"]
    assert d2["sample_rate"] == "44,100 Hz" and d2["channels"] == "Stereo (2 channels)" and d2["bit_depth"] == 16


def test_unknown_stream_facts_are_none_not_guesses(tmp_path):
    junk = tmp_path / "x.mp3"
    junk.write_bytes(b"not audio")
    info = read_stream_info(junk)
    assert info["channels"] is None and info["bit_depth"] is None


def _panel(kind, item, profile):
    from ui.results.profile import render_audio_signal_specs, render_video_stream_specs

    render_video_stream_specs(profile) if kind == "video" else render_audio_signal_specs(item, profile)


def _shown(at):
    return {m.label: str(m.value) for m in at.metric}


def test_video_panel_shows_real_values_and_not_recorded_instead_of_zero(tmp_path):
    prof = VideoProfiler().profile_video(_video(tmp_path))
    shown = _shown(AppTest.from_function(_panel, args=("video", {}, prof), default_timeout=60).run())
    assert shown["Dimensions"].startswith("640") and shown["Frame Rate"] == "25.0 fps" and shown["Duration"] == "5.00 s"
    assert shown["Total Frames"] == "125" and shown["Container / Codec"].startswith("MP4")
    empty = _shown(AppTest.from_function(_panel, args=("video", {}, {}), default_timeout=60).run())
    assert empty["Frame Rate"] == "Not recorded" and empty["Total Frames"] == "Not recorded" and "AVC" not in empty["Container / Codec"]


def test_audio_panel_shows_the_real_header_values(tmp_path):
    prof = AudioProfiler().profile_audio(_stereo_wav(tmp_path))
    shown = _shown(AppTest.from_function(_panel, args=("audio", {}, prof), default_timeout=60).run())
    assert shown["Sampling Rate"] == "44,100 Hz" and shown["Channels"] == "Stereo (2 channels)" and shown["Bit Depth"] == "16-bit"
    empty = _shown(AppTest.from_function(_panel, args=("audio", {}, {}), default_timeout=60).run())
    assert empty["Sampling Rate"] == "Not recorded" and empty["Bit Depth"] == "Not recorded"


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None or __import__("shutil").which("ffprobe") is None, reason="ffmpeg/ffprobe not installed")
def test_compressed_audio_stream_facts_come_from_ffprobe(tmp_path):
    import subprocess

    mp3 = tmp_path / "t.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=48000", "-ac", "2", str(mp3)], check=True)
    info = read_stream_info(mp3)
    assert info["sample_rate"] == 48000 and info["channels"] == 2 and info["codec"] == "mp3"
    assert info["bit_depth"] is None                                   # lossy: there is no bit depth to report, and none is invented


def test_the_audio_file_is_decoded_once_per_analysis(tmp_path, monkeypatch):
    from audio_detector.tests.audio_fixtures import tone, write_wav
    from audio_detector.validator import AudioValidator
    from services.forensic_service import ForensicService

    p = write_wav(tmp_path / "once.wav", tone(2.0, noise=0.05, seed=2))
    calls = []
    original = AudioValidator.extract_pcm_samples

    def counting(self, path):
        calls.append(str(path))
        return original(self, path)

    monkeypatch.setattr(AudioValidator, "extract_pcm_samples", counting)
    ForensicService.get_instance().audio_pipeline.analyze(p)
    assert len(calls) == 1
