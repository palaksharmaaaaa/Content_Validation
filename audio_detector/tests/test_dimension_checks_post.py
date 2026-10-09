"""Audio context / legal / lifecycle / reliability checks and the AudioDimensionAnalysis helper."""
import json

import numpy as np
import pytest

from audio_detector.dimension_checks import (
    AudioDimensionAnalysis, check_audio_gates, context, legal, lifecycle, normalize_provenance, reliability,
)
from audio_detector.tests.audio_fixtures import id3v23, text_frame_v23, tone, write_wav, write_wav_with_chunks
from core.forensics.registry import CheckContext
from core.forensics.schemas import EvidenceClass, FindingStatus


def _noise(seconds, sr=16000, seed=0):
    return (0.2 * np.random.default_rng(seed).standard_normal(int(seconds * sr))).astype(np.float32)


def _ctx(path, x=None, sr=16000, content=None, ai=None, prov=None, extra=None):
    ex = dict(extra or {})
    if x is not None:
        ex["samples"] = (x, sr, len(x) / sr)
    return CheckContext(path=path, modality="audio", content=content or {}, ai_result=ai or {}, provenance=prov or {}, extra=ex)


# ---------------- fingerprint / context ----------------
def test_fingerprint_is_deterministic_and_discriminative():
    a, b = _noise(10, seed=1), _noise(10, seed=2)
    fa1, fa2, fb = context.fingerprint(a), context.fingerprint(a.copy()), context.fingerprint(b)
    assert np.array_equal(fa1, fa2)
    assert context.best_ber(fa1, fb) > 0.40


def test_fingerprint_matches_clip_of_reference():
    ref = _noise(30, seed=3)
    clip = ref[16000 * 8 : 16000 * 18]
    ber = context.best_ber(context.fingerprint(clip), context.fingerprint(ref))
    assert ber < 0.15


def test_context_no_index(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_AUDIO_FP_INDEX", str(tmp_path / "none.jsonl"))
    f = context.check_audio_fingerprint(_ctx(tmp_path / "a.wav", _noise(6)))
    assert f.status == FindingStatus.INFO and f.evidence_class == EvidenceClass.CONTEXT
    assert f.data["index_loaded"] is False


def test_context_index_hit(tmp_path, monkeypatch):
    ref = _noise(20, seed=9)
    idx = tmp_path / "idx.jsonl"
    idx.write_text(json.dumps({"fp": context.fingerprint_hex(ref), "label": "2019 speech", "source": "archive"}) + "\n", encoding="utf-8")
    monkeypatch.setenv("OMNI_AUDIO_FP_INDEX", str(idx))
    f = context.check_audio_fingerprint(_ctx(tmp_path / "a.wav", ref[16000 * 3 : 16000 * 12]))
    assert f.status == FindingStatus.WARN
    assert f.data["matches"][0]["label"] == "2019 speech"


# ---------------- legal ----------------
def _legal(tmp_path, **kw):
    return {f.check_id: f for f in legal.check_rights_and_privacy(_ctx(tmp_path / "a.wav", **kw))}


def test_rights_notice_from_riff_icop(tmp_path):
    info = b"INFO" + b"ICOP" + (len(b"(c) 2024 ACME\x00")).to_bytes(4, "little") + b"(c) 2024 ACME\x00"
    p = write_wav_with_chunks(tmp_path / "a.wav", tone(0.2), extra={b"LIST": info + (b"\x00" if len(info) & 1 else b"")})
    out = {f.check_id: f for f in legal.check_rights_and_privacy(_ctx(p))}
    assert out["rights_notice"].data["copyright"] == "(c) 2024 ACME"
    assert all(f.evidence_class == EvidenceClass.LEGAL_FLAG for f in out.values())


def test_voice_biometric_notice_for_speech(tmp_path):
    write_wav(tmp_path / "a.wav", tone(0.2))
    out = _legal(tmp_path, content={"dominant_modality": "Vocal Speech", "speech_ratio": 0.8})
    assert out["voice_biometric_notice"].status == FindingStatus.INFO
    out2 = _legal(tmp_path, content={"dominant_modality": "Ambient Environmental Sound", "speech_ratio": 0.1})
    assert out2["voice_biometric_notice"].status == FindingStatus.NOT_APPLICABLE


def test_ai_disclosure_label(tmp_path):
    write_wav(tmp_path / "a.wav", tone(0.2))
    assert _legal(tmp_path, ai={"ai_percentage": 85.0})["ai_disclosure_label"].status == FindingStatus.WARN
    assert _legal(tmp_path, ai={"ai_percentage": 85.0}, prov={"c2pa_present": True})["ai_disclosure_label"].status == FindingStatus.PASS
    assert _legal(tmp_path, ai={"ai_percentage": 10.0})["ai_disclosure_label"].status == FindingStatus.NOT_APPLICABLE


def test_synthetic_speech_advisory(tmp_path):
    write_wav(tmp_path / "a.wav", tone(0.2))
    out = _legal(tmp_path, content={"dominant_modality": "Vocal Speech", "speech_ratio": 0.8}, ai={"ai_percentage": 90.0})
    assert out["synthetic_voice_advisory"].status == FindingStatus.WARN
    assert _legal(tmp_path, ai={"ai_percentage": 90.0})["synthetic_voice_advisory"].status == FindingStatus.NOT_APPLICABLE


# ---------------- lifecycle / reliability ----------------
def test_cascade_likelihood_high_for_narrowband_low_bitrate_mp3(tmp_path):
    p = tmp_path / "a.mp3"
    p.write_bytes(id3v23({"TIT2": text_frame_v23("x")}) + b"\xff\xfb\x40\x00" + b"\x00" * 6000)  # 64 kbps header, ~6 KB
    x = np.fft.irfft(np.where((np.fft.rfftfreq(16000 * 5, 1 / 16000) > 300) & (np.fft.rfftfreq(16000 * 5, 1 / 16000) < 3400),
                              np.fft.rfft(_noise(5, seed=4)), 0), 16000 * 5).astype(np.float32)
    res = lifecycle.assess_transcoding(_ctx(p, x))
    assert res["likelihood"] >= 0.55
    f = lifecycle.check_transcoding_cascade(_ctx(p, x))
    assert f.evidence_class == EvidenceClass.RELIABILITY and f.llr is None


def test_cascade_low_for_clean_wav(tmp_path):
    p = write_wav(tmp_path / "a.wav", _noise(3), sr=16000)
    assert lifecycle.assess_transcoding(_ctx(p, _noise(3)))["likelihood"] < 0.35


def test_reliability_limiters_for_short_clipped(tmp_path):
    x = np.clip(tone(1.0, amp=3.0), -1, 1)
    p = write_wav(tmp_path / "a.wav", x)
    f = reliability.check_confidence_limiters(_ctx(p, x))
    assert f.status == FindingStatus.WARN and f.data["level"] in ("REDUCED", "LOW")
    joined = " ".join(f.data["limiters"]).lower()
    assert "short" in joined and "clipp" in joined


def test_reliability_normal_for_clean_long(tmp_path):
    x = _noise(8)
    p = write_wav(tmp_path / "a.wav", x)
    f = reliability.check_confidence_limiters(_ctx(p, x))
    assert f.status == FindingStatus.PASS and f.data["level"] == "NORMAL"


# ---------------- helper ----------------
def test_normalize_provenance():
    assert normalize_provenance(None) == {"c2pa_present": False}
    assert normalize_provenance({"c2pa_present": True}) == {"c2pa_present": True}
    assert normalize_provenance({"c2pa": {"c2pa_present": True}}) == {"c2pa_present": True}


def test_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_HARDBLOCK_SHA256_FILE", str(tmp_path / "none.txt"))
    assert check_audio_gates(write_wav(tmp_path / "a.wav", tone(0.2)))["triggered"] is False
    midi = tmp_path / "a.mid"
    midi.write_bytes(b"MThd\x00\x00\x00\x06")
    g = check_audio_gates(midi)
    assert g["triggered"] and g["final_status"] == "RECOGNIZED_OUT_OF_SCOPE" and g["recognition"]["type"] == "MIDI"


def test_run_pre_terms_clamped_and_post_report(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_AUDIO_FP_INDEX", str(tmp_path / "none.jsonl"))
    tag = id3v23({"TXXX": b"\x00note\x00Made with Suno"})
    p = tmp_path / "g.mp3"
    p.write_bytes(tag + b"\xff\xfb\x90\x00" + b"\x00" * 800)
    x = _noise(5)
    a = AudioDimensionAnalysis(p, profile={}, provenance=None, samples=(x, 16000, 5.0))
    terms = a.run_pre()
    assert terms["id3_tags"] == pytest.approx(0.40)
    assert abs(sum(terms.values())) <= 0.40 + 1e-9
    report = a.run_post(content={}, ai_result={"ai_percentage": 99.7, "acoustic_features": {
        "has_vocoder_cutoff": True, "cutoff_freq_hz": 7000.0, "spectral_flatness": 0.001, "digital_silence_ratio": 0.2, "high_freq_ratio": 0.0}},
        attribution={"model_key": "unknown"})
    assert report["confidence_band"]["band"] == "HIGH_CONFIDENCE_SYNTHETIC"
    assert report["ood"]["status"] == "NOT_CALIBRATED"
    assert report["attribution_open_set"]["unknown_source"] is True
    assert {"file_integrity", "container", "signal", "legal", "lifecycle", "reliability", "context"} <= set(report["findings_by_stage"])


def test_unknown_source_false_for_authentic(tmp_path):
    p = write_wav(tmp_path / "a.wav", _noise(3))
    a = AudioDimensionAnalysis(p, samples=(_noise(3), 16000, 3.0))
    a.run_pre()
    r = a.run_post(content={}, ai_result={"ai_percentage": 5.0}, attribution={"model_key": "unknown"})
    assert r["attribution_open_set"]["unknown_source"] is False
    assert r["confidence_band"]["band"] == "LEANING_AUTHENTIC"


def test_a_streamed_wav_with_an_unset_riff_size_is_not_called_truncated(tmp_path):
    import struct
    import wave

    from audio_detector.dimension_checks.integrity import check_trailing_data
    from core.forensics.registry import CheckContext

    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(b"\x01\x00" * 800)
    raw = bytearray(p.read_bytes())
    raw[4:8] = struct.pack("<I", 0xFFFFFFFF)
    p.write_bytes(bytes(raw))
    f = check_trailing_data(CheckContext(path=p, modality="audio"))
    assert f.status.name == "NOT_APPLICABLE"


def test_a_streamed_wav_data_chunk_is_not_reported_as_truncated(tmp_path):
    import struct
    import wave

    from audio_detector.dimension_checks._common import walk_riff

    p = tmp_path / "s2.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(b"\x01\x00" * 800)
    raw = bytearray(p.read_bytes())
    i = raw.index(b"data")
    raw[i + 4:i + 8] = struct.pack("<I", 0xFFFFFFFF)
    p.write_bytes(bytes(raw))
    assert walk_riff(p)["issues"] == []
