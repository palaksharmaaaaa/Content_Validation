"""Signal-level audio checks: ENF trace, fake hi-res / bit padding, telephony band-limit, loudness."""
import numpy as np
import pytest

from audio_detector.dimension_checks import signal
from audio_detector.tests.audio_fixtures import tone, write_wav
from core.forensics.registry import CheckContext
from core.forensics.schemas import EvidenceClass, FindingStatus


def _ctx(path, x=None, sr=16000, extra=None):
    ex = dict(extra or {})
    if x is not None:
        ex["samples"] = (x, sr, len(x) / sr)
    return CheckContext(path=path, modality="audio", extra=ex)


def _hum(seconds, sr=16000, base=60.0, wander=0.05, period=20.0, amp=0.01, noise=0.02, seed=1, phase0=0.0):
    t = np.arange(int(seconds * sr)) / sr
    f = base + wander * np.sin(2 * np.pi * t / period + phase0)
    phase = 2 * np.pi * np.cumsum(f) / sr
    return (amp * np.sin(phase) + noise * np.random.default_rng(seed).standard_normal(len(t))).astype(np.float32)


# ---------------- enf_trace ----------------
def test_enf_continuous_trace_is_slightly_toward_real(tmp_path):
    f = signal.check_enf_trace(_ctx(tmp_path / "x.wav", _hum(40)))
    assert f.status == FindingStatus.PASS
    assert f.data["nominal_hz"] == 60
    assert f.llr == pytest.approx(-0.15)
    assert f.evidence_class == EvidenceClass.PHYSICAL_SIGNAL
    assert abs(f.data["median_hz"] - 60.0) < 0.1


def test_enf_50hz_grid_detected(tmp_path):
    f = signal.check_enf_trace(_ctx(tmp_path / "x.wav", _hum(40, base=50.0)))
    assert f.data["nominal_hz"] == 50 and f.status == FindingStatus.PASS


def test_enf_splice_discontinuity(tmp_path):
    a = _hum(25, base=59.90, wander=0.03, seed=1)
    b = _hum(25, base=60.25, wander=0.03, seed=2)
    f = signal.check_enf_trace(_ctx(tmp_path / "x.wav", np.concatenate([a, b])))
    assert f.status == FindingStatus.WARN
    assert f.llr == pytest.approx(0.15)
    assert f.data["jumps"] >= 1


def test_enf_absent_is_info_without_llr(tmp_path):
    x = (0.02 * np.random.default_rng(3).standard_normal(16000 * 30)).astype(np.float32)
    f = signal.check_enf_trace(_ctx(tmp_path / "x.wav", x))
    assert f.status == FindingStatus.INFO
    assert f.llr is None
    assert "not evidence" in f.detail.lower()


def test_enf_constant_tone_is_not_a_mains_trace(tmp_path):
    t = np.arange(16000 * 25) / 16000
    x = (0.3 * np.sin(2 * np.pi * 60.0 * t) + 0.01 * np.random.default_rng(4).standard_normal(len(t))).astype(np.float32)
    f = signal.check_enf_trace(_ctx(tmp_path / "x.wav", x))
    assert f.status == FindingStatus.INFO
    assert f.llr is None
    assert f.data.get("stable_tone") is True


def test_enf_too_short(tmp_path):
    assert signal.check_enf_trace(_ctx(tmp_path / "x.wav", _hum(5))).status == FindingStatus.NOT_APPLICABLE


# ---------------- fake_hires ----------------
def _band_limited_noise(seconds, sr, cutoff, seed=0):
    n = int(seconds * sr)
    X = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    X[f > cutoff] = 0
    x = np.fft.irfft(X, n)
    x = 0.3 * x / np.max(np.abs(x))
    return x.astype(np.float32)


def test_fake_hires_96k_with_20k_cutoff(tmp_path):
    x = _band_limited_noise(4, 96000, 20000)
    p = write_wav(tmp_path / "hi.wav", x, sr=96000, bits=24)
    f = signal.check_fake_hires(_ctx(p))
    assert f.status == FindingStatus.WARN
    assert f.data["verdict"] == "UPSAMPLED_FROM_LOWER_RATE"
    assert 18000 <= f.data["cutoff_hz"] <= 21500
    assert f.llr is None


def test_true_wideband_noise_at_96k_passes(tmp_path):
    x = (0.1 * np.random.default_rng(5).standard_normal(96000 * 3)).astype(np.float32)
    p = write_wav(tmp_path / "full.wav", x, sr=96000, bits=24)
    f = signal.check_fake_hires(_ctx(p))
    assert f.status == FindingStatus.PASS


def test_lossy_origin_16k_lowpass_in_cd_wav(tmp_path):
    x = _band_limited_noise(4, 44100, 16000, seed=2)
    p = write_wav(tmp_path / "cd.wav", x, sr=44100, bits=16)
    f = signal.check_fake_hires(_ctx(p))
    assert f.data["verdict"] == "POSSIBLE_LOSSY_ORIGIN"
    assert f.status == FindingStatus.INFO


def test_16bit_padded_into_24bit(tmp_path):
    import wave

    x = (0.1 * np.random.default_rng(6).standard_normal(48000 * 2)).astype(np.float32)
    v24 = (np.round(x * 32767).astype(np.int32)) << 8  # 16-bit values placed in the top 16 bits of 24
    raw = np.stack([(v24 & 0xFF), ((v24 >> 8) & 0xFF), ((v24 >> 16) & 0xFF)], axis=1).astype(np.uint8).tobytes()
    p = tmp_path / "pad.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)
        wf.setframerate(48000)
        wf.writeframes(raw)
    f = signal.check_fake_hires(_ctx(p))
    assert f.data["padded_bits"] is True
    assert f.status == FindingStatus.WARN


def test_fake_hires_na_for_mp3_extension(tmp_path):
    p = tmp_path / "a.mp3"
    p.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 200)
    assert signal.check_fake_hires(_ctx(p)).status == FindingStatus.NOT_APPLICABLE


# ---------------- telephony_channel ----------------
def _bandpass_noise(lo, hi, seconds=5, sr=16000, seed=0):
    n = int(seconds * sr)
    X = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    X[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(X, n)
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def test_narrowband_telephony_detected(tmp_path):
    f = signal.check_telephony_channel(_ctx(tmp_path / "t.wav", _bandpass_noise(300, 3400)))
    assert f.data["narrowband_telephony"] is True


def test_wideband_not_telephony(tmp_path):
    x = (0.1 * np.random.default_rng(7).standard_normal(16000 * 5)).astype(np.float32)
    f = signal.check_telephony_channel(_ctx(tmp_path / "t.wav", x))
    assert f.data["narrowband_telephony"] is False


def test_lowpass_only_is_band_limited_not_telephony(tmp_path):
    f = signal.check_telephony_channel(_ctx(tmp_path / "t.wav", _bandpass_noise(20, 3400)))
    assert f.data["band_limited_4k"] is True and f.data["narrowband_telephony"] is False


# ---------------- loudness_dynamics ----------------
def test_loudness_basic_metrics(tmp_path):
    f = signal.check_loudness_dynamics(_ctx(tmp_path / "l.wav", tone(2.0, amp=0.5)))
    assert f.data["peak_dbfs"] == pytest.approx(-6.0, abs=0.3)
    assert f.data["crest_db"] == pytest.approx(3.0, abs=0.3)
    assert f.data["over_compressed"] is True  # sine crest 3 dB < 6 dB


def test_clipping_flagged(tmp_path):
    x = np.clip(tone(2.0, amp=3.0), -1, 1)
    f = signal.check_loudness_dynamics(_ctx(tmp_path / "c.wav", x))
    assert f.status == FindingStatus.WARN and f.data["clipping_ratio"] > 0.01


def test_silent_audio_not_applicable(tmp_path):
    assert signal.check_loudness_dynamics(_ctx(tmp_path / "s.wav", np.zeros(16000, np.float32))).status == FindingStatus.NOT_APPLICABLE
