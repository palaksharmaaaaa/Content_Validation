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
