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
