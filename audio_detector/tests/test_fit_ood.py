import numpy as np

from audio_detector.dimension_checks import AudioDimensionAnalysis, acoustic_vector
from audio_detector.dimension_checks.fit_ood import fit_ood_gate
from audio_detector.tests.audio_fixtures import tone, write_wav
from core.forensics.ood import OODGate


def test_acoustic_vector_shape_and_none():
    v = acoustic_vector({"has_vocoder_cutoff": True, "cutoff_freq_hz": 7000.0, "spectral_flatness": 0.01,
                         "digital_silence_ratio": 0.1, "high_freq_ratio": 0.02})
    assert v.shape == (5,) and v[0] == 1.0
    assert acoustic_vector(None) is None and acoustic_vector({}) is None


def test_fit_and_use(tmp_path):
    rng = np.random.default_rng(1)
    vecs = {f"f{i}.wav": rng.normal(0, 1, 5) for i in range(60)}
    paths = [tmp_path / k for k in vecs]
    for p in paths:
        p.write_bytes(b"x")
    out = tmp_path / "ood.npz"
    res = fit_ood_gate(paths, out, vectorizer=lambda p: vecs[p.name])
    assert res["fitted"] and out.is_file()
    gate = OODGate.load(out)

    wav = write_wav(tmp_path / "probe.wav", tone(0.3))
    a = AudioDimensionAnalysis(wav, ood_gate=gate)
    a.run_pre()
    far = {"ai_percentage": 50.0, "acoustic_features": {"has_vocoder_cutoff": True, "cutoff_freq_hz": 200000.0,
                                                          "spectral_flatness": 0.5, "digital_silence_ratio": 9.0, "high_freq_ratio": 9.0}}
    assert a.run_post({}, far)["ood"]["status"] == "OUT_OF_DISTRIBUTION"
    near = {"ai_percentage": 50.0, "acoustic_features": {"has_vocoder_cutoff": False, "cutoff_freq_hz": 0.0,
                                                           "spectral_flatness": 0.0, "digital_silence_ratio": 0.0, "high_freq_ratio": 0.0}}
    assert a.run_post({}, near)["ood"]["status"] in ("IN_DISTRIBUTION", "OUT_OF_DISTRIBUTION")  # computed, not NOT_CALIBRATED


def test_too_few_not_fitted(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"x")
    assert fit_ood_gate([p], tmp_path / "o.npz", vectorizer=lambda q: np.zeros(5))["fitted"] is False
