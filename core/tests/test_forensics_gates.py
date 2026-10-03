"""Hard-block hash gate + scientific-format recognition gate."""
import hashlib
import struct

from core.forensics.gates import HardBlockGate, recognize_scientific_format, recognize_symbolic_music


def _write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_no_blocklist_is_inactive(tmp_path):
    f = _write(tmp_path, "a.bin", b"hello")
    res = HardBlockGate(blocklist_path=tmp_path / "missing.txt").check(f)
    assert res.status == "INACTIVE"
    assert not res.triggered
    assert res.sha256 == hashlib.sha256(b"hello").hexdigest()


def test_match_triggers_escalation(tmp_path):
    data = b"bad-bytes"
    f = _write(tmp_path, "a.bin", data)
    digest = hashlib.sha256(data).hexdigest()
    bl = _write(tmp_path, "bl.txt", f"# comment\n\n{digest.upper()}  extra note\n".encode())
    res = HardBlockGate(blocklist_path=bl).check(f)
    assert res.status == "HARD_BLOCK_ESCALATE"
    assert res.triggered


def test_non_match_is_clear(tmp_path):
    f = _write(tmp_path, "a.bin", b"fine")
    bl = _write(tmp_path, "bl.txt", ("0" * 64 + "\n").encode())
    res = HardBlockGate(blocklist_path=bl).check(f)
    assert res.status == "CLEAR"
    assert not res.triggered


def test_env_var_blocklist(tmp_path, monkeypatch):
    data = b"envbad"
    f = _write(tmp_path, "a.bin", data)
    bl = _write(tmp_path, "bl.txt", hashlib.sha256(data).hexdigest().encode())
    monkeypatch.setenv("OMNI_HARDBLOCK_SHA256_FILE", str(bl))
    assert HardBlockGate().check(f).triggered


def test_missing_file_is_inactive_not_crash(tmp_path):
    res = HardBlockGate(blocklist_path=tmp_path / "x.txt").check(tmp_path / "nofile.bin")
    assert res.status == "INACTIVE"
    assert res.sha256 == ""


def test_recognizes_fits(tmp_path):
    f = _write(tmp_path, "s.fits", b"SIMPLE  =                    T" + b" " * 100)
    assert recognize_scientific_format(f)["type"] == "FITS"


def test_recognizes_dicom(tmp_path):
    f = _write(tmp_path, "s.dcm", b"\x00" * 128 + b"DICM" + b"\x00" * 50)
    assert recognize_scientific_format(f)["type"] == "DICOM"


def test_recognizes_openexr_hdf5_netcdf_aedat(tmp_path):
    assert recognize_scientific_format(_write(tmp_path, "a.exr", b"\x76\x2f\x31\x01" + b"\x00" * 20))["type"] == "OpenEXR"
    assert recognize_scientific_format(_write(tmp_path, "a.h5", b"\x89HDF\r\n\x1a\n" + b"\x00" * 20))["type"] == "HDF5"
    assert recognize_scientific_format(_write(tmp_path, "a.nc", b"CDF\x01" + b"\x00" * 20))["type"] == "NetCDF"
    assert recognize_scientific_format(_write(tmp_path, "a.aedat", b"#!AER-DAT3.1\r\n" + b"\x00" * 20))["type"] == "AEDAT"


def test_recognizes_geotiff(tmp_path):
    # Minimal little-endian TIFF with one IFD entry: tag 34735 (GeoKeyDirectoryTag).
    entry = struct.pack("<HHII", 34735, 3, 4, 0)
    data = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 1) + entry + struct.pack("<I", 0)
    f = _write(tmp_path, "g.tif", data)
    assert recognize_scientific_format(f)["type"] == "GeoTIFF"


def test_plain_tiff_png_not_recognized(tmp_path):
    entry = struct.pack("<HHII", 256, 3, 1, 10)
    tiff = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 1) + entry + struct.pack("<I", 0)
    assert recognize_scientific_format(_write(tmp_path, "p.tif", tiff)) is None
    assert recognize_scientific_format(_write(tmp_path, "p.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 30)) is None
    assert recognize_scientific_format(tmp_path / "missing") is None


def test_recognizes_midi_only(tmp_path):
    assert recognize_symbolic_music(_write(tmp_path, "a.mid", b"MThd\x00\x00\x00\x06"))["type"] == "MIDI"
    assert recognize_symbolic_music(_write(tmp_path, "a.wav", b"RIFF....WAVE")) is None
    assert recognize_symbolic_music(tmp_path / "missing") is None
