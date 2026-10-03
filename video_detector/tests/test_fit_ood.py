import numpy as np

from core.forensics.ood import OODGate
from video_detector.dimension_checks import VideoDimensionAnalysis
from video_detector.dimension_checks.fit_ood import fit_ood_gate
from video_detector.tests.video_fixtures import moving_square_frames, write_clip


def test_fit_and_use(tmp_path):
    rng = np.random.default_rng(2)
    vecs = {f"f{i}.mp4": rng.normal(0, 1, 5) for i in range(60)}
    paths = [tmp_path / k for k in vecs]
    for p in paths:
        p.write_bytes(b"x")
    out = tmp_path / "ood.npz"
    res = fit_ood_gate(paths, out, vectorizer=lambda p: vecs[p.name])
    assert res["fitted"] and out.is_file()
    gate = OODGate.load(out)

    clip = write_clip(tmp_path / "probe.mp4", moving_square_frames(30))
    a = VideoDimensionAnalysis(clip, ood_gate=gate)
    a.run_pre()
    far = {"ai_percentage": 50.0, "temporal_consistency": {"mean_motion_delta": 500.0, "motion_variance": 1e9},
           "diffusion_flicker": {"flicker_score": 50.0, "flicker_ratio": 50.0, "mean_lum_jump": 500.0}}
    assert a.run_post({}, far)["ood"]["status"] == "OUT_OF_DISTRIBUTION"
    near = {"ai_percentage": 50.0, "temporal_consistency": {"mean_motion_delta": 0.0, "motion_variance": 0.0},
            "diffusion_flicker": {"flicker_score": 0.0, "flicker_ratio": 0.0, "mean_lum_jump": 0.0}}
    assert a.run_post({}, near)["ood"]["status"] in ("IN_DISTRIBUTION", "OUT_OF_DISTRIBUTION")


def test_too_few_not_fitted(tmp_path):
    p = tmp_path / "a.mp4"
    p.write_bytes(b"x")
    assert fit_ood_gate([p], tmp_path / "o.npz", vectorizer=lambda q: np.zeros(5))["fitted"] is False
