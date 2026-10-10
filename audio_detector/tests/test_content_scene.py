"""The audio scene analysis describes the whole recording with the fields its consumers read, and claims nothing it cannot measure."""
import numpy as np

from audio_detector.content import AudioContentAnalyzer

SR = 16000


def _t(seconds):
    return np.arange(int(seconds * SR)) / SR


def test_it_reads_the_whole_recording_not_its_first_64_milliseconds():
    """Silence-like noise for the first 100 ms, then 5 s of a voice-band tone: the verdict must follow the 5 s."""
    rng = np.random.default_rng(0)
    head = rng.normal(0, 0.3, int(0.1 * SR))                       # broadband burst at the start
    voice = 0.4 * np.sin(2 * np.pi * 1000 * _t(5.0))
    result = AudioContentAnalyzer().analyze_audio_scene(np.concatenate([head, voice]), SR)
    assert "speech" in result["dominant_modality"].lower() and result["speech_ratio"] > 0.65


def test_high_frequency_content_is_music_like_and_noise_is_ambient():
    high = 0.4 * np.sin(2 * np.pi * 6000 * _t(3.0))
    assert "Music-like" in AudioContentAnalyzer().analyze_audio_scene(high, SR)["dominant_modality"]
    rng = np.random.default_rng(1)
    flat = rng.normal(0, 0.02, 3 * SR)
    assert "Ambient" in AudioContentAnalyzer().analyze_audio_scene(flat, SR)["dominant_modality"]


def test_it_reports_exactly_the_documented_fields_and_no_room_or_speaker_claims():
    result = AudioContentAnalyzer().analyze_audio_scene(0.3 * np.sin(2 * np.pi * 800 * _t(2.0)), SR)
    assert set(result) == {"signal_level", "dominant_modality", "delivery_style", "speech_ratio", "rms_energy", "crest_factor"}
    text = " ".join(str(v) for v in result.values()).lower()
    assert "studio" not in text and "speaker" not in text and "domestic" not in text


def test_a_recording_too_short_to_measure_says_unknown_not_a_number():
    result = AudioContentAnalyzer().analyze_audio_scene(np.zeros(100), SR)
    assert result["dominant_modality"] == "Unknown" and result["speech_ratio"] is None and result["rms_energy"] is None


def test_the_ui_shows_the_real_fields():
    from streamlit.testing.v1 import AppTest

    def page(content):
        from ui.results.content import render_scene_and_content_intelligence
        render_scene_and_content_intelligence(content, modality="audio")

    content = AudioContentAnalyzer().analyze_audio_scene(0.4 * np.sin(2 * np.pi * 1000 * _t(3.0)), SR)
    at = AppTest.from_function(page, args=(content,), default_timeout=60).run()
    shown = " ".join(str(m.value) for m in at.metric)
    assert "Speech-like" in shown and "Strong signal" in shown
    assert not any("Studio" in str(m.value) or "Conversational" in str(m.value) for m in at.metric)


def test_a_cutoff_near_the_recordings_own_nyquist_is_not_a_vocoder():
    from audio_detector.scoring import pool_acoustic_evidence

    feats = {"has_vocoder_cutoff": False, "cutoff_freq_hz": 7600.0, "spectral_flatness": 0.05, "digital_silence_ratio": 0.0, "high_freq_ratio": 0.05}
    plain = pool_acoustic_evidence(feats, {}, None, 0.0, {})[0]
    flagged = pool_acoustic_evidence({**feats, "has_vocoder_cutoff": True}, {}, None, 0.0, {})[0]
    assert plain < flagged


def test_audio_is_named_only_when_its_metadata_says_so(tmp_path):
    from audio_detector.attribution import AudioModelAttributionEngine
    from audio_detector.tests.audio_fixtures import tone, write_wav

    p = write_wav(tmp_path / "a.wav", tone(2.0, noise=0.05, seed=1))
    eng = AudioModelAttributionEngine()
    acoustic = {"spectral_features": {"has_vocoder_cutoff": True, "cutoff_freq_hz": 7600.0, "spectral_flatness": 0.001, "digital_silence_ratio": 0.3}}
    out = eng.attribute_audio(p, acoustic_data=acoustic, provenance_data={"metadata": {}})
    assert out["model_key"] == "unknown" and out["top_candidates"] == [] and out["cues"]
    named = eng.attribute_audio(p, acoustic_data=acoustic, provenance_data={"metadata": {"comment": "made with ElevenLabs"}})
    assert named["model_key"] == "elevenlabs"


def test_the_audio_narrative_is_grounded_in_what_was_measured():
    from audio_detector.explain import generate_audio_newbie_explanation

    ar = {"spectral_flatness": 0.0012, "digital_silence_ratio": 0.31, "cutoff_freq_hz": 7600.0, "has_vocoder_cutoff": True,
          "forensic_cues": ["Brick-wall frequency cutoff at 7600 Hz"]}
    prof = {"duration_seconds": 4.0, "native_sample_rate": 16000}
    ai = generate_audio_newbie_explanation("a.wav", prof, {}, ar, {"final_status": "LIKELY_SYNTHETIC", "authenticity_probabilities": {"p_ai": 80.0, "p_real": 10.0}})
    assert "0.0012" in ai and "31.0%" in ai and "7,600 Hz" in ai and "ElevenLabs" not in ai and "not proof" in ai
    blank = generate_audio_newbie_explanation("a.wav", prof, {}, ar, {"final_status": "BLANK_OR_DEGRADED", "authenticity_probabilities": {}})
    assert "no verdict" in blank


def test_audio_dossier_does_not_invent_synthesis_facts():
    from audio_detector.explain import build_audio_nine_dimensions_dossier

    ar = {"label": "LIKELY REAL", "ai_percentage": 20.0, "acoustic_features": {"cutoff_freq_hz": 7000.0, "measured": True}}
    d = build_audio_nine_dimensions_dossier({"native_sample_rate": 44100, "duration_seconds": 3.0, "channels": 2}, ar, {}, {}, {})
    assert d["dimension_7"]["is_synthetic_voice"] is False and d["dimension_7"]["synthesis_medium"] == "Not determined"
    assert d["dimension_8"]["analysed_band_limit"].startswith("8,000") and d["dimension_8"]["energy_cutoff_hz"] == "7,000 Hz"
    assert d["dimension_9"]["watermark_detected"] is None
    und = build_audio_nine_dimensions_dossier({}, {"label": "UNDECIDED"}, {}, {}, {})
    assert und["dimension_7"]["is_synthetic_voice"] is None


def test_audio_vendor_names_must_be_whole_words(tmp_path):
    from audio_detector.attribution import AudioModelAttributionEngine
    from audio_detector.tests.audio_fixtures import tone, write_wav

    p = write_wav(tmp_path / "w.wav", tone(1.0, seed=3))
    eng = AudioModelAttributionEngine()
    for text in ("this take resembles my earlier one", "supersonic", "an audio interview"):
        assert eng.attribute_audio(p, provenance_data={"metadata": {"comment": text}})["model_key"] == "unknown", text
    assert eng.attribute_audio(p, provenance_data={"metadata": {"comment": "voice by Resemble AI"}})["model_key"] == "resemble_ai"


def test_container_generator_tags_need_whole_names():
    from audio_detector.dimension_checks.container import _generator_in

    for text in ("resemble airplane noise", "Stable audio levels", "Coquitlam", "an audio interview", "supersonic"):
        assert _generator_in({"t": text}) is None, text
    assert _generator_in({"t": "Made with ElevenLabs"}) == "elevenlabs"
    assert _generator_in({"t": "Stable Audio Open"}) == "stable-audio"


def test_ai_duration_never_exceeds_the_recording_and_overlaps_are_merged():
    from audio_detector.detector import _covered_seconds

    segs = [{"start_seconds": 0.0, "end_seconds": 3.0, "label": "LIKELY AI-GENERATED"},
            {"start_seconds": 1.5, "end_seconds": 4.5, "label": "LIKELY AI-GENERATED"},
            {"start_seconds": 3.0, "end_seconds": 6.0, "label": "LIKELY REAL"}]
    assert _covered_seconds(segs, "LIKELY AI-GENERATED") == 4.5


def test_a_tag_that_names_a_generator_reaches_attribution(tmp_path):
    from audio_detector.attribution import AudioModelAttributionEngine
    from audio_detector.provenance import AudioProvenanceValidator
    from audio_detector.tests.audio_fixtures import tone, write_wav

    p = write_wav(tmp_path / "t.wav", tone(1.0, seed=2))
    raw = bytearray(p.read_bytes())
    info = b"INFO" + b"ISFT" + (14).to_bytes(4, "little") + b"Made by Suno\x00\x00"
    raw += b"LIST" + len(info).to_bytes(4, "little") + info
    raw[4:8] = (len(raw) - 8).to_bytes(4, "little")
    p.write_bytes(bytes(raw))
    prov = AudioProvenanceValidator().analyze_provenance(p)
    out = AudioModelAttributionEngine().attribute_audio(p, provenance_data=prov)
    assert out["model_key"] == "suno_ai"


def test_exact_zeros_in_8_bit_audio_are_not_a_synthetic_cue(tmp_path):
    """Quiet passages in 8-bit PCM round to exact zero whatever recorded them, so they must not read as 'digital silence'."""
    import numpy as np

    from audio_detector.detector import AudioAIDetector
    from audio_detector.tests.audio_fixtures import tone, write_wav

    quiet = np.concatenate([tone(1.0, amp=0.3), np.full(16000 * 2, 0.0004, np.float32), tone(1.0, amp=0.3, freq=300)])
    eight = write_wav(tmp_path / "e8.wav", quiet, bits=8)
    sixteen = write_wav(tmp_path / "e16.wav", quiet, bits=16)
    det = AudioAIDetector()
    r8 = det.analyze_audio_file(eight)
    r16 = det.analyze_audio_file(sixteen)
    assert r8["digital_silence_ratio"] is None
    assert r16["digital_silence_ratio"] is not None
    assert not any("Digital zero" in c for c in r8["forensic_cues"])


def test_opposite_phase_stereo_is_not_mistaken_for_silence(tmp_path):
    """Channels in opposite phase cancel when averaged to mono; the file must still be analysed from one channel."""
    import wave

    import numpy as np

    from audio_detector.tests.audio_fixtures import tone
    from audio_detector.validator import AudioValidator

    x = (tone(1.0, amp=0.4) * 32767).astype("<i2")
    stereo = np.stack([x, -x], axis=1).reshape(-1)
    p = tmp_path / "antiphase.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(stereo.tobytes())
    result = AudioValidator().validate(p)
    assert result.valid and result.is_silent is False


def test_a_hostile_sample_rate_does_not_hang_the_hires_check(tmp_path):
    import struct
    import time

    import numpy as np

    from audio_detector.dimension_checks.signal import check_fake_hires
    from core.forensics.registry import CheckContext

    data = (np.random.default_rng(0).normal(0, 3000, 40000)).astype("<i2").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 2_000_000_000, 4_000_000_000 % 2**32, 2, 16) + b"data" + struct.pack("<I", len(data))
    p = tmp_path / "hostile.wav"
    p.write_bytes(header + data)
    t0 = time.perf_counter()
    check_fake_hires(CheckContext(path=p, modality="audio"))
    assert time.perf_counter() - t0 < 5.0
