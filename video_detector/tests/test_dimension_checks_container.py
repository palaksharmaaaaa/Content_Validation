"""Video container / metadata checks: boxes, sample tables, metadata, telemetry, SEI, EBML."""
import shutil
import subprocess

import pytest

from core.forensics.registry import CheckContext, build_report
from core.forensics.schemas import EvidenceClass, FindingStatus
from video_detector.dimension_checks import _common as C
from video_detector.dimension_checks import container
from video_detector.tests.video_fixtures import box, build_mp4, text_atom, trak

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _ctx(p):
    return CheckContext(path=p, modality="video")


def _write(tmp_path, data, name="a.mp4"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ---------------- isobmff_boxes ----------------
def test_boxes_layout_faststart_vs_trailing_moov(tmp_path):
    fast = container.check_isobmff_boxes(_ctx(_write(tmp_path, build_mp4([trak()], faststart=True), "f.mp4")))
    slow = container.check_isobmff_boxes(_ctx(_write(tmp_path, build_mp4([trak()], faststart=False), "s.mp4")))
    assert fast.data["layout"] == "FASTSTART" and slow.data["layout"] == "MOOV_AT_END"
    assert fast.status == FindingStatus.INFO and fast.evidence_class == EvidenceClass.METADATA_WEAK
    assert fast.llr is None
    assert fast.data["major_brand"] == "isom"


def test_boxes_fragmented_and_c2pa_uuid(tmp_path):
    extra = box(b"moof", b"\x00" * 8) + box(b"uuid", C.C2PA_UUID + b"manifest")
    f = container.check_isobmff_boxes(_ctx(_write(tmp_path, build_mp4([trak()], extra_top=extra))))
    assert f.data["fragmented"] is True
    assert f.data["c2pa_box"] is True
    assert "not cryptographically validated" in f.detail


def test_boxes_missing_moov_warns(tmp_path):
    data = build_mp4([trak()])
    # drop the moov: keep ftyp + mdat only
    ftyp_len = int.from_bytes(data[:4], "big")
    mdat_len = int.from_bytes(data[ftyp_len : ftyp_len + 4], "big")
    only = data[: ftyp_len + mdat_len]
    f = container.check_isobmff_boxes(_ctx(_write(tmp_path, only)))
    assert f.status == FindingStatus.WARN and "moov" in f.detail


def test_boxes_not_applicable_for_mkv(tmp_path):
    p = _write(tmp_path, b"\x1a\x45\xdf\xa3" + b"\x00" * 100, "a.mkv")
    assert container.check_isobmff_boxes(_ctx(p)).status == FindingStatus.NOT_APPLICABLE


# ---------------- isobmff_tables ----------------
def test_tables_consistent_cfr(tmp_path):
    p = _write(tmp_path, build_mp4([trak(stts_entries=((25, 1000),), chunk_offsets=(48,))]))
    f = container.check_isobmff_tables(_ctx(p))
    assert f.status == FindingStatus.PASS
    assert f.data["tracks"][0]["fps"] == pytest.approx(25.0)
    assert f.data["tracks"][0]["cfr"] is True


def test_tables_sample_count_mismatch(tmp_path):
    p = _write(tmp_path, build_mp4([trak(stts_entries=((25, 1000),), sample_count=20)]))
    f = container.check_isobmff_tables(_ctx(p))
    assert f.status == FindingStatus.WARN
    assert any("stsz" in i for i in f.data["tracks"][0]["issues"])


def test_tables_vfr_is_info(tmp_path):
    p = _write(tmp_path, build_mp4([trak(stts_entries=((10, 1000), (10, 2000)), stss_idx=(1,))]))
    f = container.check_isobmff_tables(_ctx(p))
    assert f.status == FindingStatus.INFO
    assert f.data["tracks"][0]["cfr"] is False


def test_tables_sync_sample_out_of_range(tmp_path):
    p = _write(tmp_path, build_mp4([trak(stss_idx=(1, 999))]))
    assert container.check_isobmff_tables(_ctx(p)).status == FindingStatus.WARN


def test_tables_chunk_offset_beyond_file(tmp_path):
    p = _write(tmp_path, build_mp4([trak(chunk_offsets=(10 ** 9,))]))
    assert container.check_isobmff_tables(_ctx(p)).status == FindingStatus.WARN


def test_tables_stsc_total_mismatch(tmp_path):
    p = _write(tmp_path, build_mp4([trak(sample_per_chunk=10)]))
    assert container.check_isobmff_tables(_ctx(p)).status == FindingStatus.WARN


def test_tables_zero_duration_samples(tmp_path):
    p = _write(tmp_path, build_mp4([trak(stts_entries=((10, 1000), (5, 0)))]))
    assert container.check_isobmff_tables(_ctx(p)).status == FindingStatus.WARN


# ---------------- mp4_metadata ----------------
def test_metadata_plain_is_info(tmp_path):
    p = _write(tmp_path, build_mp4([trak()], mvhd_args={"creation": 3_800_000_000, "modification": 3_800_000_100}))
    f = container.check_mp4_metadata(_ctx(p))
    assert f.status == FindingStatus.INFO and f.llr is None
    assert f.data["creation"].startswith("2024")


def test_metadata_unset_times_info(tmp_path):
    f = container.check_mp4_metadata(_ctx(_write(tmp_path, build_mp4([trak()]))))
    assert f.status == FindingStatus.INFO and f.data["creation"] is None


def test_metadata_time_paradoxes(tmp_path):
    p = _write(tmp_path, build_mp4([trak()], mvhd_args={"creation": 3_800_000_500, "modification": 3_800_000_000}))
    f = container.check_mp4_metadata(_ctx(p))
    assert f.status == FindingStatus.WARN and any("modification" in x for x in f.data["paradoxes"])
    fut = _write(tmp_path, build_mp4([trak()], mvhd_args={"creation": 4_290_000_000, "modification": 4_290_000_000}), "fut.mp4")
    assert container.check_mp4_metadata(_ctx(fut)).status == FindingStatus.WARN


def test_metadata_capture_and_editor_tools(tmp_path):
    extra = box(b"udta", text_atom(b"\xa9too", "obs-studio 30.0"))
    p = _write(tmp_path, build_mp4([trak(handler_name="VideoHandler")], moov_extra=extra))
    f = container.check_mp4_metadata(_ctx(p))
    assert "obs" in f.data["capture_tools"]
    assert f.llr is None


def test_metadata_explicit_generator_string(tmp_path):
    extra = box(b"udta", text_atom(b"\xa9cmt", "Created with Kling AI video model"))
    p = _write(tmp_path, build_mp4([trak()], moov_extra=extra))
    f = container.check_mp4_metadata(_ctx(p))
    assert f.status == FindingStatus.WARN and f.data["generator"] == "kling"
    assert f.llr == pytest.approx(0.40)
    assert build_report([f]).score_terms["mp4_metadata"] == pytest.approx(0.40)


def test_metadata_title_naming_a_generator_not_flagged(tmp_path):
    extra = box(b"udta", text_atom(b"\xa9nam", "Sora and the sea"))
    f = container.check_mp4_metadata(_ctx(_write(tmp_path, build_mp4([trak()], moov_extra=extra))))
    assert f.llr is None and f.data["generator"] is None


# ---------------- telemetry_tracks ----------------
def test_telemetry_gopro_gpmf(tmp_path):
    p = _write(tmp_path, build_mp4([trak(), trak(handler=b"meta", fourcc=b"gpmd", handler_name="GoPro MET")]))
    f = container.check_telemetry_tracks(_ctx(p))
    assert f.status == FindingStatus.INFO
    assert any("GoPro" in t for t in f.data["telemetry"])
    assert f.llr is None


def test_telemetry_none(tmp_path):
    f = container.check_telemetry_tracks(_ctx(_write(tmp_path, build_mp4([trak()]))))
    assert f.data["telemetry"] == []


def test_telemetry_location_atom(tmp_path):
    extra = box(b"udta", text_atom(b"\xa9xyz", "+37.7749-122.4194/"))
    f = container.check_telemetry_tracks(_ctx(_write(tmp_path, build_mp4([trak()], moov_extra=extra))))
    assert f.data["location"] == "+37.7749-122.4194/"


# ---------------- encoder_sei ----------------
def test_encoder_sei_strings(tmp_path):
    payload = b"\x00\x00\x01\x06" + b"x264 - core 164 r3095 baee400 - H.264/MPEG-4 AVC codec - options: cabac=1 crf=23.0" + b"\x00" * 200
    p = _write(tmp_path, build_mp4([trak()], mdat_payload=payload))
    f = container.check_encoder_sei(_ctx(p))
    assert "x264" in f.data["encoders"] and "crf=23.0" in f.data["encoders"]["x264"]
    assert f.llr is None


def test_encoder_sei_absent_is_info(tmp_path):
    f = container.check_encoder_sei(_ctx(_write(tmp_path, build_mp4([trak()]))))
    assert f.status == FindingStatus.INFO and f.data["encoders"] == {}


@needs_ffmpeg
def test_real_ffmpeg_x264_clip(tmp_path):
    p = tmp_path / "x.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    assert "x264" in container.check_encoder_sei(_ctx(p)).data["encoders"]
    meta = container.check_mp4_metadata(_ctx(p))
    assert meta.data["tools"]  # Lavf / libx264 handler or encoder tag
    assert container.check_isobmff_tables(_ctx(p)).status in (FindingStatus.PASS, FindingStatus.INFO)


# ---------------- ebml_info ----------------
def test_ebml_info(tmp_path):
    head = b"\x1a\x45\xdf\xa3" + b"\x42\x82\x84webm" + b"\x00" * 20 + b"\x4d\x80\x8dLavf58.76.100" + b"\x57\x41\x8dLavf58.76.100"
    f = container.check_ebml_info(_ctx(_write(tmp_path, head, "a.webm")))
    assert f.data["doc_type"] == "webm" and f.data["muxing_app"] == "Lavf58.76.100"
    assert f.llr is None


def test_ebml_generator_string(tmp_path):
    app = b"Runway Gen-3 export"
    head = b"\x1a\x45\xdf\xa3" + b"\x42\x82\x84webm" + b"\x57\x41" + bytes([0x80 | len(app)]) + app
    f = container.check_ebml_info(_ctx(_write(tmp_path, head, "a.webm")))
    assert f.data["generator"] == "runway" and f.llr == pytest.approx(0.40)


def test_ebml_not_applicable_for_mp4(tmp_path):
    assert container.check_ebml_info(_ctx(_write(tmp_path, build_mp4([trak()])))).status == FindingStatus.NOT_APPLICABLE
