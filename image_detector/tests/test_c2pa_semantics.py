"""C2PA marker presence is reported but never scored or labelled as cryptographic verification."""
import numpy as np
import pytest
from PIL import Image

from image_detector.detector import ImageAIDetector
from image_detector.provenance import ImageProvenanceValidator


def _img(tmp_path, name, extra=b""):
    p = tmp_path / name
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(p, "JPEG", quality=90)
    if extra:
        p.write_bytes(p.read_bytes() + extra)
    return p


def test_signature_marker_is_unverified(tmp_path):
    p = _img(tmp_path, "m.jpg", b"jumb c2pa c2pa.claim c2pa.signature")
    c2pa = ImageProvenanceValidator().analyze_provenance(p)["c2pa"]
    assert c2pa["c2pa_present"] and c2pa["is_signed"]
    assert c2pa["status"] == "SIGNATURE_MARKER_UNVERIFIED"
    assert "not cryptographic" in c2pa["details"].lower()


def test_provenance_view_never_claims_verification(tmp_path):
    view = ImageProvenanceValidator().analyze_provenance(_img(tmp_path, "v.jpg", b"c2pa.claim"))
    assert view["provenance_verdict"] == "C2PA_SIGNATURE_MARKER_UNVERIFIED"
    assert view["provenance_ai_confidence"] == 0.50


def test_presence_gives_no_score_credit(tmp_path):
    det = ImageAIDetector()
    det.load()
    p = _img(tmp_path, "a.jpg")
    base = det.predict(str(p))
    with_c2pa = det.predict(str(p), provenance={"c2pa_present": True})
    assert "c2pa_verified" not in with_c2pa["log_likelihood_ratios"]
    assert with_c2pa["ai_percentage"] == pytest.approx(base["ai_percentage"], abs=0.05)
