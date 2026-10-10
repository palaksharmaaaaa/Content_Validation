import shutil
import wave

import numpy as np
import pytest

from core.ffmpeg import ffmpeg_input, ffprobe_input, pcm_to_float


def test_every_call_is_hardened():
    cmd = ffmpeg_input("x.mp3", max_seconds=5)
    assert cmd[:2] == ["ffmpeg", "-nostdin"] and ["-protocol_whitelist", "file,pipe"] == cmd[cmd.index("-protocol_whitelist"):][:2]
    assert cmd[-2:] == ["-i", "x.mp3"] and "-t" in cmd
    assert "-t" not in ffmpeg_input("x.mp3")
    assert "-protocol_whitelist" in ffprobe_input("x.mp3")


@pytest.mark.parametrize("width,dtype,peak", [(1, None, None), (2, "<i2", 32767), (4, "<i4", 2**31 - 1)])
def test_pcm_widths_decode_to_unit_range(width, dtype, peak):
    if width == 1:
        raw = bytes([0, 128, 255])
        out = pcm_to_float(raw, 1)
        assert out[0] == -1.0 and out[1] == 0.0 and out[2] == pytest.approx(127 / 128)
    else:
        raw = np.array([-peak - 1, 0, peak], dtype=dtype).tobytes()
        out = pcm_to_float(raw, width)
        assert out[0] == -1.0 and out[1] == 0.0 and out[2] == pytest.approx(1.0, abs=1e-4)


def test_24_bit_pcm_is_decoded_not_misread_as_16_bit():
    vals = np.array([-8388608, 0, 8388607], dtype=np.int32)
    raw = b"".join(int(v & 0xFFFFFF).to_bytes(3, "little") for v in vals)
    out = pcm_to_float(raw, 3)
    assert out[0] == -1.0 and out[1] == 0.0 and out[2] == pytest.approx(1.0, abs=1e-6)


def test_an_unknown_width_is_an_error():
    with pytest.raises(ValueError):
        pcm_to_float(b"\x00" * 10, 5)


def test_a_24_bit_wav_is_decoded_correctly_without_ffmpeg(tmp_path, monkeypatch):
    from audio_detector.validator import AudioValidator

    sr = 16000
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 440 * t) * 8388607).astype(np.int32)
    p = tmp_path / "a24.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(b"".join(int(v & 0xFFFFFF).to_bytes(3, "little") for v in x))
    monkeypatch.setattr(shutil, "which", lambda *_: None)
    import audio_detector.validator as V

    monkeypatch.setattr(V.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))   # no ffmpeg: native path
    s, fr, dur = AudioValidator().extract_pcm_samples(p)
    assert fr == sr and dur == pytest.approx(1.0) and float(np.max(np.abs(s))) == pytest.approx(0.5, abs=0.01)


def test_a_playlist_named_like_media_is_refused_not_followed(tmp_path):
    """An upload that is really an ffconcat list must not make ffmpeg read the file it names."""
    import shutil
    import subprocess

    import pytest

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    from core.ffmpeg import ffmpeg_input

    real = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=0.2", str(real)], check=True)
    sneaky = tmp_path / "c.mp3"
    sneaky.write_text("ffconcat version 1.0\nfile 'a.wav'\n")
    done = subprocess.run(ffmpeg_input(sneaky) + ["-f", "null", "-"], capture_output=True, text=True)
    assert done.returncode != 0 and "whitelist" in done.stderr.lower()
    assert subprocess.run(ffmpeg_input(real) + ["-f", "null", "-"], capture_output=True).returncode == 0


def test_a_playlist_named_like_a_video_is_never_given_to_opencv(tmp_path):
    from core.ffmpeg import looks_like_video_container, open_video

    playlist = tmp_path / "evil.mp4"
    playlist.write_text("#EXTM3U\n#EXTINF:1,x\nhttp://127.0.0.1:9/never.ts\n")
    assert looks_like_video_container(playlist) is False and open_video(playlist).isOpened() is False
    for head in (b"\x00\x00\x00\x18ftypmp42", b"\x00\x00\x00\x08wide", b"\x1a\x45\xdf\xa3\x01", b"RIFF\x00\x00\x00\x00AVI LIST"):
        real = tmp_path / "real.bin"
        real.write_bytes(head + b"\x00" * 32)
        assert looks_like_video_container(real) is True, head
    assert looks_like_video_container(tmp_path / "missing.mp4") is False
