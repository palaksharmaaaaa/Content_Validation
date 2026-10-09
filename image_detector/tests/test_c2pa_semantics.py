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


def test_random_image_bytes_are_not_mistaken_for_content_credentials(tmp_path):
    """A bare 4-byte tag appears by chance in compressed data; only a distinctive marker, or the JUMBF box with a tag, counts."""
    rng = np.random.default_rng(7)
    hits = 0
    for i in range(300):
        blob = bytearray(rng.integers(0, 256, 600_000, dtype=np.uint8).tobytes())
        if i % 3 == 0:
            blob[1000:1004] = b"c2pa"                                  # a lone short tag planted in noise
        p = tmp_path / f"r{i}.jpg"
        p.write_bytes(bytes(blob))
        hits += ImageProvenanceValidator().scan_c2pa(p)["c2pa_present"]
    assert hits == 0


def test_a_real_looking_manifest_is_still_detected(tmp_path):
    p = tmp_path / "m.jpg"
    p.write_bytes(b"\xff\xd8" + b"\x00" * 100 + b"jumb" + b"c2pa" + b"\x00" * 100)
    assert ImageProvenanceValidator().scan_c2pa(p)["c2pa_present"] is True
