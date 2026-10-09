"""The nine-dimension page shows what the dossier holds for every modality, and never turns a missing measurement into a tick."""
from streamlit.testing.v1 import AppTest

from audio_detector.explain import build_audio_nine_dimensions_dossier
from video_detector.explain import build_video_nine_dimensions_dossier


def _page(dossier):
    from ui.results.dimensions import render_nine_dimensions_breakdown
    render_nine_dimensions_breakdown(dossier, expanded=True)


def _render(dossier):
    return AppTest.from_function(_page, args=(dossier,), default_timeout=60).run()


def _text(at):
    return "\n".join(str(getattr(e, "value", "")) for e in list(at.markdown) + list(at.warning) + list(at.success) + list(at.info) + list(at.caption))


def test_suspicious_audio_findings_are_warnings_and_all_fields_are_shown():
    acoustics = {"measured": True, "has_vocoder_cutoff": True, "cutoff_freq_hz": 8000.0, "spectral_flatness": 0.001, "digital_silence_ratio": 0.4}
    dossier = build_audio_nine_dimensions_dossier({"sample_rate": 16000, "duration": 3.0}, {"acoustic_features": acoustics})
    at = _render(dossier)
    assert not at.exception
    text = _text(at)
    assert "Detected cutoff hz" in text and "8000 Hz" in text and "0.001" in text and "40.0%" in text
    assert len(at.warning) >= 3 and not any("Not measured" in w.value for w in at.warning)


def test_unremarkable_audio_is_a_tick_and_unmeasured_audio_is_neutral():
    ok = {"measured": True, "has_vocoder_cutoff": False, "cutoff_freq_hz": 7000.0, "spectral_flatness": 0.2, "digital_silence_ratio": 0.0}
    at = _render(build_audio_nine_dimensions_dossier({}, {"acoustic_features": ok}))
    assert len(at.success) >= 3
    short = _render(build_audio_nine_dimensions_dossier({}, {"acoustic_features": {"measured": False}}))
    assert not short.exception and not any("Not measured" in s.value for s in short.success)
    assert any("Not measured" in i.value for i in short.info)


def test_video_dossier_without_measurements_never_shows_a_tick_for_them():
    at = _render(build_video_nine_dimensions_dossier({"geometry": {}}, {}))
    assert not at.exception
    assert not any("Not measured" in s.value for s in at.success)
    assert sum("Not measured" in i.value for i in at.info) == 3            # noise, motion, flicker


def test_video_flicker_and_warping_are_warnings():
    res = {"mean_frame_noise": 0.9, "temporal_consistency": {"motion_variance": 150.0, "temporal_warping_risk": "HIGH_WARPING_DETECTED"},
           "diffusion_flicker": {"flicker_score": 0.8, "has_diffusion_flicker": True, "frames_compared": 20}}
    at = _render(build_video_nine_dimensions_dossier({"geometry": {}}, res))
    assert len(at.warning) >= 3
    text = _text(at)
    assert "Flicker score" in text and "0.8" in text and "Motion variance" in text
