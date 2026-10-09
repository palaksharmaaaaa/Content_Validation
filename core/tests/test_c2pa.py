import numpy as np

from core.c2pa import scan_file


def test_markers_at_the_end_of_a_big_file_are_found(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"\x00" * 3_000_000 + b"jumb" + b"c2pa" + b"\x00" * 100)
    r = scan_file(p)
    assert r["c2pa_present"] and r["status"] == "C2PA_CREDENTIALS_FOUND"


def test_a_lone_short_tag_is_not_a_manifest(tmp_path):
    p = tmp_path / "b.bin"
    rng = np.random.default_rng(1)
    p.write_bytes(rng.integers(0, 256, 1000, dtype=np.uint8).tobytes() + b"c2ma" + b"\x01" * 50)
    assert scan_file(p)["c2pa_present"] is False


def test_distinctive_marker_counts_alone_and_missing_file_is_reported(tmp_path):
    p = tmp_path / "c.bin"
    p.write_bytes(b"xx urn:c2pa:1234 xx")
    assert scan_file(p)["c2pa_present"] is True
    assert scan_file(tmp_path / "nope")["status"] == "FILE_NOT_FOUND"
