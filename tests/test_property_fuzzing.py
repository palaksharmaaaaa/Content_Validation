"""Property-based fuzzing tests using Hypothesis for binary parsers and forensic scans.

Validates that binary parsers in image_detector, audio_detector, and core.forensics
are completely crash-resilient and uphold strict security invariants when presented
with arbitrary, corrupted, truncated, or hostile byte sequences.
"""
import pytest

pytest.importorskip("hypothesis", reason="optional dependency: pip install -r requirements.txt")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from audio_detector.dimension_checks._common import walk_riff
from audio_detector.dimension_checks.container import check_riff_structure
from core.forensics.bytescan import extract_xmp, scan_signatures
from core.forensics.registry import CheckContext
from image_detector.dimension_checks.integrity import check_polyglot_signatures
from image_detector.profiler import ImageProfiler
from image_detector.provenance import ImageProvenanceValidator


@settings(max_examples=50, deadline=None)
@given(blob=st.binary(min_size=0, max_size=100_000))
def test_c2pa_scanner_fuzz_never_crashes(blob: bytes, tmp_path_factory):
    """ImageProvenanceValidator.scan_c2pa must never raise an unhandled exception on arbitrary byte sequences."""
    tmp_dir = tmp_path_factory.mktemp("c2pa_fuzz")
    p = tmp_dir / "fuzz.bin"
    p.write_bytes(blob)

    res = ImageProvenanceValidator().scan_c2pa(p)
    assert isinstance(res, dict)
    assert "c2pa_present" in res and isinstance(res["c2pa_present"], bool)
    assert res["status"] in ("C2PA_CREDENTIALS_FOUND", "NO_C2PA_MANIFEST")


@settings(max_examples=30, deadline=None)
@given(
    tag=st.sampled_from([b"c2pa", b"jumb", b"urn:c2pa", b"application/c2pa", b"c2pa.claim"]),
    prefix=st.binary(max_size=20_000),
    suffix=st.binary(max_size=20_000),
)
def test_c2pa_scanner_planted_markers_respect_distinctive_threshold(tag: bytes, prefix: bytes, suffix: bytes, tmp_path_factory):
    """A bare 4-byte 'c2pa' tag requires 'jumb', whereas long distinctive markers (>= 8 chars) qualify independently."""
    tmp_dir = tmp_path_factory.mktemp("c2pa_planted")
    p = tmp_dir / "planted.bin"
    p.write_bytes(prefix + tag + suffix)

    res = ImageProvenanceValidator().scan_c2pa(p)
    assert isinstance(res, dict)
    if len(tag) >= 8:
        # A distinctive marker (e.g. urn:c2pa, application/c2pa) on its own counts
        assert res["c2pa_present"] is True


@settings(max_examples=50, deadline=None)
@given(blob=st.binary(min_size=0, max_size=50_000))
def test_polyglot_bytescan_fuzz_never_crashes(blob: bytes, tmp_path_factory):
    """core.forensics.bytescan routines must safely process arbitrary payloads without unhandled crashes."""
    tmp_dir = tmp_path_factory.mktemp("polyglot_fuzz")
    p = tmp_dir / "polyglot.bin"
    p.write_bytes(blob)

    # Direct bytescan scanners
    sigs_trail = scan_signatures(blob, trailing_only=True)
    sigs_lead = scan_signatures(blob, trailing_only=False)
    assert isinstance(sigs_trail, list)
    assert isinstance(sigs_lead, list)

    xmp = extract_xmp(blob)
    assert isinstance(xmp, str)

    # High-level forensic polyglot detector
    ctx = CheckContext(path=p, modality="image")
    res = check_polyglot_signatures(ctx)
    assert res is not None
    assert hasattr(res, "status")


@settings(max_examples=50, deadline=None)
@given(blob=st.binary(min_size=0, max_size=50_000))
def test_riff_container_parser_fuzz_never_crashes(blob: bytes, tmp_path_factory):
    """audio_detector container RIFF chunk parser must safely handle arbitrary corrupted chunks."""
    tmp_dir = tmp_path_factory.mktemp("riff_fuzz")
    p = tmp_dir / "corrupt.wav"
    p.write_bytes(blob)

    # Direct RIFF tree walker
    riff_data = walk_riff(p)
    assert isinstance(riff_data, dict)
    assert "chunks" in riff_data
    assert "issues" in riff_data

    # Forensic dimension check
    ctx = CheckContext(path=p, modality="audio")
    res = check_riff_structure(ctx)
    assert res is not None
    assert hasattr(res, "status")


@settings(max_examples=30, deadline=None)
@given(blob=st.binary(min_size=0, max_size=40_000))
def test_image_profiler_and_exif_fuzz_never_crashes(blob: bytes, tmp_path_factory):
    """ImageProfiler must safely parse or reject arbitrary corrupt image byte sequences without crashing."""
    tmp_dir = tmp_path_factory.mktemp("profiler_fuzz")
    p = tmp_dir / "test_corrupt.jpg"
    p.write_bytes(blob)

    try:
        prof = ImageProfiler().profile_image(p)
    except Exception as exc:
        # Expected PIL/decoding errors: the bytes cannot be opened as an image
        assert "cannot identify image" in str(exc).lower() or "unidentifiedimageerror" in type(exc).__name__.lower()
        return
    assert isinstance(prof, dict)
    if prof.get("valid"):
        assert "exif_device_details" in prof
    else:
        assert "error" in prof
