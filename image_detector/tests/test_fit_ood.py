import numpy as np

from core.forensics.ood import OODGate
from image_detector.dimension_checks import ImageDimensionAnalysis
from image_detector.dimension_checks.fit_ood import fit_ood_gate


def test_fit_from_paths_and_use_in_analysis(tmp_path):
    rng = np.random.default_rng(3)
    vecs = {f"f{i}.png": rng.normal(0, 1, 8) for i in range(60)}
    paths = [tmp_path / k for k in vecs]
    for p in paths:
        p.write_bytes(b"x")
    out = tmp_path / "ood.npz"
    res = fit_ood_gate(paths, out, embedder=lambda p: vecs[p.name])
    assert res["fitted"] is True and out.is_file()

    gate = OODGate.load(out)
    from PIL import Image

    img = tmp_path / "probe.jpg"
    Image.new("RGB", (64, 64)).save(img)
    far = ImageDimensionAnalysis(img, ood_gate=gate, ood_embedder=lambda p: np.full(8, 15.0))
    far.run_pre()
    assert far.run_post({}, {"ai_percentage": 50.0})["ood"]["status"] == "OUT_OF_DISTRIBUTION"
    near = ImageDimensionAnalysis(img, ood_gate=gate, ood_embedder=lambda p: np.zeros(8))
    near.run_pre()
    assert near.run_post({}, {"ai_percentage": 50.0})["ood"]["status"] == "IN_DISTRIBUTION"


def test_too_few_samples_not_fitted(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(b"x")
    res = fit_ood_gate([p], tmp_path / "o.npz", embedder=lambda q: np.zeros(4))
    assert res["fitted"] is False
    assert not (tmp_path / "o.npz").exists()


def test_embedder_errors_are_skipped(tmp_path):
    paths = [tmp_path / f"{i}.png" for i in range(30)]

    def flaky(p):
        if p.name.startswith("1"):
            raise RuntimeError("bad")
        return np.random.default_rng(0).normal(size=4)

    res = fit_ood_gate(paths, tmp_path / "o.npz", embedder=flaky)
    assert res["samples"] < 30
