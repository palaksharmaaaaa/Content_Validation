"""Metadata & container dimension checks: EXIF coherence, timestamps, thumbnail, ICC."""
import io
import struct

import pytest
from PIL import Image, ImageDraw

from core.forensics.registry import CheckContext
from core.forensics.schemas import EvidenceClass, FindingStatus
from image_detector.dimension_checks import _common as C
from image_detector.dimension_checks import metadata


def _ctx(path, camera=False):
    return CheckContext(path=path, modality="image", provenance={"has_camera_hardware": camera})


def _exif(make="Canon", model="Canon EOS R5", original="2024:05:01 10:00:00", digitized=None, modified=None,
          full=True, gps=False):
    ex = Image.Exif()
    if make is not None:
        ex[271] = make
    if model is not None:
        ex[272] = model
    if modified:
        ex[306] = modified
    sub = ex.get_ifd(0x8769)
    if original:
        sub[0x9003] = original
    if digitized:
        sub[0x9004] = digitized
    if full:
        sub[0x829A] = (1, 250)
        sub[0x829D] = (28, 10)
        sub[0x8827] = 100
        sub[0x920A] = (50, 1)
    if gps:
        ex.get_ifd(0x8825)[1] = "N"
    return ex


def _jpeg(tmp_path, exif=None, name="a.jpg", size=(128, 96), draw=None):
    p = tmp_path / name
    im = Image.new("RGB", size, (200, 100, 50))
    if draw:
        ImageDraw.Draw(im).rectangle(draw, fill=(0, 0, 0))
    kwargs = {"exif": exif} if exif is not None else {}
    im.save(p, "JPEG", quality=90, **kwargs)
    return p


def _inject_thumbnail(path, thumb_jpeg: bytes):
    """Insert an Exif APP1 segment containing IFD0(empty) + IFD1 pointing at thumb_jpeg."""
    ifd0 = struct.pack("<H", 0) + struct.pack("<I", 14)
    thumb_off = 14 + 2 + 2 * 12 + 4
    ifd1 = (struct.pack("<H", 2)
            + struct.pack("<HHII", 0x0201, 4, 1, thumb_off)
            + struct.pack("<HHII", 0x0202, 4, 1, len(thumb_jpeg))
            + struct.pack("<I", 0))
    tiff = b"II*\x00" + struct.pack("<I", 8) + ifd0 + ifd1 + thumb_jpeg
    seg = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(seg) + 2) + seg
    data = path.read_bytes()
    path.write_bytes(data[:2] + app1 + data[2:])


def _thumb_bytes(draw=None, size=(32, 24)):
    im = Image.new("RGB", size, (200, 100, 50))
    if draw:
        ImageDraw.Draw(im).rectangle(draw, fill=(0, 0, 0))
    b = io.BytesIO()
    im.save(b, "JPEG")
    return b.getvalue()


# ---------------- exif_consistency ----------------
def test_coherent_camera_exif_is_never_rewarded(tmp_path):
    """Coherent EXIF is what a forger supplies: it must carry no score credit in either configuration."""
    p = _jpeg(tmp_path, _exif())
    for camera in (True, False):
        f = metadata.check_exif_consistency(_ctx(p, camera=camera))
        assert f.status == FindingStatus.PASS
        assert f.llr is None
        assert f.evidence_class == EvidenceClass.METADATA_WEAK


def test_brand_conflict_is_incoherent(tmp_path):
    p = _jpeg(tmp_path, _exif(make="Apple", model="SM-S918B"))
    f = metadata.check_exif_consistency(_ctx(p, camera=True))
    assert f.status == FindingStatus.WARN
    assert f.llr == pytest.approx(0.25)


def test_absent_exif_is_info_without_llr(tmp_path):
    f = metadata.check_exif_consistency(_ctx(_jpeg(tmp_path), camera=False))
    assert f.status == FindingStatus.INFO
    assert f.llr is None


def test_partial_exif_no_llr(tmp_path):
    p = _jpeg(tmp_path, _exif(full=False))
    f = metadata.check_exif_consistency(_ctx(p, camera=True))
    assert f.llr is None


# ---------------- timestamp_sanity ----------------
def test_digitized_before_original_is_paradox(tmp_path):
    p = _jpeg(tmp_path, _exif(original="2024:05:01 10:00:00", digitized="2024:04:01 09:00:00"))
    f = metadata.check_timestamp_sanity(_ctx(p))
    assert f.status == FindingStatus.WARN
    assert f.llr == pytest.approx(0.15)
    assert any("Digitized" in r for r in f.data["paradoxes"])


def test_future_timestamp_is_paradox(tmp_path):
    p = _jpeg(tmp_path, _exif(original="2999:01:01 00:00:00"))
    assert metadata.check_timestamp_sanity(_ctx(p)).status == FindingStatus.WARN


def test_ordered_timestamps_pass(tmp_path):
    p = _jpeg(tmp_path, _exif(original="2024:05:01 10:00:00", digitized="2024:05:01 10:00:00", modified="2024:05:02 08:00:00"))
    f = metadata.check_timestamp_sanity(_ctx(p))
    assert f.status == FindingStatus.PASS
    assert f.llr is None


def test_no_timestamps_not_applicable(tmp_path):
    assert metadata.check_timestamp_sanity(_ctx(_jpeg(tmp_path))).status == FindingStatus.NOT_APPLICABLE


# ---------------- thumbnail_match ----------------
def test_matching_thumbnail_passes(tmp_path):
    p = _jpeg(tmp_path, draw=(10, 10, 60, 50))
    _inject_thumbnail(p, _thumb_bytes(draw=(2, 3, 15, 12)))
    f = metadata.check_thumbnail_match(_ctx(p))
    assert f.status == FindingStatus.PASS
    assert f.llr is None


def test_mismatched_thumbnail_warns_with_llr(tmp_path):
    p = _jpeg(tmp_path, draw=(10, 10, 60, 50))
    # Thumbnail shows a clearly different layout (block on the opposite side).
    _inject_thumbnail(p, _thumb_bytes(draw=(18, 12, 31, 23)))
    f = metadata.check_thumbnail_match(_ctx(p))
    assert f.status == FindingStatus.WARN
    assert f.llr == pytest.approx(0.25)
    assert f.evidence_class == EvidenceClass.PHYSICAL_SIGNAL


def test_no_thumbnail_not_applicable(tmp_path):
    assert metadata.check_thumbnail_match(_ctx(_jpeg(tmp_path))).status == FindingStatus.NOT_APPLICABLE


def test_extract_exif_thumbnail_round_trip(tmp_path):
    p = _jpeg(tmp_path)
    t = _thumb_bytes()
    _inject_thumbnail(p, t)
    assert C.extract_exif_thumbnail(p) == t


# ---------------- icc_profile ----------------
def test_icc_absent_is_info(tmp_path):
    f = metadata.check_icc_profile(_ctx(_jpeg(tmp_path)))
    assert f.status == FindingStatus.INFO
    assert f.data["present"] is False
    assert f.llr is None


def test_icc_present_is_info_with_descriptor(tmp_path):
    from PIL import ImageCms

    prof = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    p = tmp_path / "icc.jpg"
    Image.new("RGB", (32, 32), (9, 9, 9)).save(p, "JPEG", icc_profile=prof)
    f = metadata.check_icc_profile(_ctx(p))
    assert f.data["present"] is True
    assert f.llr is None
