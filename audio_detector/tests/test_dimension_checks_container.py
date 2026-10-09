"""Audio container / metadata checks: ID3, RIFF+bext, FLAC MD5, MP3 encoder tag, Ogg structure."""
import shutil
import subprocess

import pytest

from audio_detector.dimension_checks import container
from audio_detector.tests.audio_fixtures import (
    id3v23, ogg_page, text_frame_v23, tone, txxx_v23, write_wav, write_wav_with_chunks,
)
from core.forensics.registry import CheckContext, build_report
from core.forensics.schemas import EvidenceClass, FindingStatus

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _ctx(p):
    return CheckContext(path=p, modality="audio")


MP3_FRAME = b"\xff\xfb\x90\x00"  # MPEG1 L3 128kbps 44.1kHz stereo


def _mp3(tmp_path, tag=b"", xing=b""):
    p = tmp_path / "a.mp3"
    p.write_bytes(tag + MP3_FRAME + xing + b"\x00" * 400)
    return p


# ---------------- id3_tags ----------------
def test_id3_inventory(tmp_path):
    tag = id3v23({"TIT2": text_frame_v23("Hello"), "TPE1": text_frame_v23("Artist")}, padding=64)
    f = container.check_id3_tags(_ctx(_mp3(tmp_path, tag)))
    assert f.status == FindingStatus.INFO
    assert f.data["version"].startswith("2.3")
    assert {fr["id"] for fr in f.data["frames"]} == {"TIT2", "TPE1"}
    assert f.llr is None
    assert f.evidence_class == EvidenceClass.METADATA_WEAK


def test_id3_explicit_generator_string(tmp_path):
    tag = id3v23({"TIT2": text_frame_v23("Song"), "TXXX": txxx_v23("comment", "Created with Suno v4")})
    f = container.check_id3_tags(_ctx(_mp3(tmp_path, tag)))
    assert f.status == FindingStatus.WARN
    assert f.data["generator"] == "suno"
    assert f.llr == pytest.approx(0.40)
    assert build_report([f]).score_terms["id3_tags"] == pytest.approx(0.40)


def test_artist_or_title_naming_a_generator_is_not_flagged(tmp_path):
    tag = id3v23({"TIT2": text_frame_v23("Suno"), "TPE1": text_frame_v23("Udio Band")})
    f = container.check_id3_tags(_ctx(_mp3(tmp_path, tag)))
    assert f.llr is None and f.data["generator"] is None


def test_id3_declared_size_beyond_file(tmp_path):
    tag = bytearray(id3v23({"TIT2": text_frame_v23("x")}))
    tag[6:10] = bytes([0, 0x7F, 0x7F, 0x7F])  # absurd synchsafe size
    p = tmp_path / "bad.mp3"
    p.write_bytes(bytes(tag) + MP3_FRAME + b"\x00" * 100)
    assert container.check_id3_tags(_ctx(p)).status == FindingStatus.WARN


def test_id3_absent_not_applicable_for_wav(tmp_path):
    assert container.check_id3_tags(_ctx(write_wav(tmp_path / "a.wav", tone(0.2)))).status == FindingStatus.NOT_APPLICABLE


# ---------------- riff_structure ----------------
def test_riff_inventory_plain(tmp_path):
    f = container.check_riff_structure(_ctx(write_wav(tmp_path / "a.wav", tone(0.2))))
    assert f.status == FindingStatus.INFO
    assert "fmt " in f.data["chunks"] and "data" in f.data["chunks"]
    assert f.data["bext"] is None


def test_riff_bext_and_info_software(tmp_path):
    bext = bytearray(602)
    bext[0:11] = b"Field recor"
    bext[256:263] = b"Zoom H6"
    ch = b"A=PCM,F=48000,W=24,M=mono,T=ZoomH6\r\n"
    info = b"INFO" + b"ISFT" + (len(b"Sound Forge\x00")).to_bytes(4, "little") + b"Sound Forge\x00"
    p = write_wav_with_chunks(tmp_path / "b.wav", tone(0.2), extra={b"bext": bytes(bext) + ch, b"LIST": info + (b"\x00" if len(info) & 1 else b"")})
    f = container.check_riff_structure(_ctx(p))
    assert f.data["bext"]["originator"] == "Zoom H6"
    assert "A=PCM" in f.data["bext"]["coding_history"]
    assert f.data["info"]["ISFT"] == "Sound Forge"
    assert f.llr is None


def test_riff_explicit_generator_in_info(tmp_path):
    info = b"INFO" + b"ISFT" + (len(b"ElevenLabs API\x00")).to_bytes(4, "little") + b"ElevenLabs API\x00"
    p = write_wav_with_chunks(tmp_path / "g.wav", tone(0.2), extra={b"LIST": info + (b"\x00" if len(info) & 1 else b"")})
    f = container.check_riff_structure(_ctx(p))
    assert f.status == FindingStatus.WARN
    assert f.data["generator"] == "elevenlabs"
    assert f.llr == pytest.approx(0.40)


def test_riff_truncated_data_warns(tmp_path):
    p = write_wav(tmp_path / "t.wav", tone(0.5))
    p.write_bytes(p.read_bytes()[:-500])
    assert container.check_riff_structure(_ctx(p)).status == FindingStatus.WARN


def test_riff_not_applicable_for_non_wav(tmp_path):
    assert container.check_riff_structure(_ctx(_mp3(tmp_path))).status == FindingStatus.NOT_APPLICABLE


# ---------------- flac_md5 ----------------
@needs_ffmpeg
def test_flac_md5_matches_then_tamper_detected(tmp_path):
    wav = write_wav(tmp_path / "a.wav", tone(1.0, sr=16000, noise=0.01))
    flac = tmp_path / "a.flac"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(flac)], check=True)
    assert container.check_flac_md5(_ctx(flac)).status == FindingStatus.PASS

    data = bytearray(flac.read_bytes())
    data[8 + 18] ^= 0xFF  # flip a byte of the stored MD5
    bad = tmp_path / "bad.flac"
    bad.write_bytes(bytes(data))
    f = container.check_flac_md5(_ctx(bad))
    assert f.status == FindingStatus.WARN


def test_flac_md5_unset_is_info(tmp_path):
    # Hand-built minimal FLAC header with an all-zero MD5 (decoder would not be run).
    info = bytearray(34)
    packed = (16000 << 44) | (0 << 41) | (15 << 36) | 16000
    info[10:18] = packed.to_bytes(8, "big")
    p = tmp_path / "z.flac"
    p.write_bytes(b"fLaC" + bytes([0x80, 0, 0, 34]) + bytes(info) + b"\x00" * 50)
    assert container.check_flac_md5(_ctx(p)).status == FindingStatus.INFO


def test_flac_not_applicable_for_wav(tmp_path):
    assert container.check_flac_md5(_ctx(write_wav(tmp_path / "a.wav", tone(0.2)))).status == FindingStatus.NOT_APPLICABLE


# ---------------- mp3_encoder_tag ----------------
def test_mp3_lame_tag_parsed(tmp_path):
    xing = b"\x00" * 32 + b"Info" + b"\x00" * 4 + b"\x00" * 96 + b"LAME3.100" + b"\x00" + bytes([195]) + b"\x00" * 20
    f = container.check_mp3_encoder_tag(_ctx(_mp3(tmp_path, xing=xing)))
    assert f.data["encoder"] == "LAME3.100"
    assert f.data["lowpass_hz"] == 19500
    assert f.data["sample_rate"] == 44100 and f.data["bitrate_kbps"] == 128
    assert f.llr is None


def test_mp3_without_tag_is_info(tmp_path):
    f = container.check_mp3_encoder_tag(_ctx(_mp3(tmp_path)))
    assert f.status == FindingStatus.INFO and f.data["encoder"] is None


def test_mp3_check_not_applicable_for_wav(tmp_path):
    assert container.check_mp3_encoder_tag(_ctx(write_wav(tmp_path / "a.wav", tone(0.2)))).status == FindingStatus.NOT_APPLICABLE


# ---------------- ogg_structure ----------------
def _ogg(tmp_path, pages):
    p = tmp_path / "a.ogg"
    p.write_bytes(b"".join(pages))
    return p


def test_ogg_valid_stream_passes(tmp_path):
    p = _ogg(tmp_path, [ogg_page(b"OpusHead" + b"\x00" * 11, seq=0, header_type=2),
                        ogg_page(b"a" * 30, seq=1, granule=960), ogg_page(b"b" * 30, seq=2, granule=1920, header_type=4)])
    f = container.check_ogg_structure(_ctx(p))
    assert f.status == FindingStatus.PASS and f.data["pages"] == 3


def test_ogg_bad_crc_warns(tmp_path):
    p = _ogg(tmp_path, [ogg_page(b"x" * 10, seq=0, header_type=2), ogg_page(b"y" * 10, seq=1, corrupt_crc=True)])
    f = container.check_ogg_structure(_ctx(p))
    assert f.status == FindingStatus.WARN and f.data["crc_failures"] == 1


def test_ogg_sequence_gap_warns(tmp_path):
    p = _ogg(tmp_path, [ogg_page(b"x" * 10, seq=0, header_type=2), ogg_page(b"y" * 10, seq=5)])
    f = container.check_ogg_structure(_ctx(p))
    assert f.status == FindingStatus.WARN and f.data["sequence_gaps"] == 1


def test_ogg_granule_regression_warns(tmp_path):
    p = _ogg(tmp_path, [ogg_page(b"x" * 10, seq=0, granule=5000, header_type=2), ogg_page(b"y" * 10, seq=1, granule=100)])
    assert container.check_ogg_structure(_ctx(p)).status == FindingStatus.WARN


def test_ogg_chained_streams_are_info(tmp_path):
    p = _ogg(tmp_path, [ogg_page(b"x" * 10, serial=1, seq=0, header_type=2), ogg_page(b"y" * 10, serial=1, seq=1, header_type=4),
                        ogg_page(b"z" * 10, serial=2, seq=0, header_type=2), ogg_page(b"w" * 10, serial=2, seq=1, header_type=4)])
    f = container.check_ogg_structure(_ctx(p))
    assert f.status == FindingStatus.INFO and f.data["serials"] == 2


def test_ogg_not_applicable_for_wav(tmp_path):
    assert container.check_ogg_structure(_ctx(write_wav(tmp_path / "a.wav", tone(0.2)))).status == FindingStatus.NOT_APPLICABLE


# ---------------- regression: structure beyond the 4 MB scan window ----------------
def _big_wav_with_trailing_list(tmp_path, list_payload: bytes, seconds=60):
    """>4 MB WAV whose LIST chunk sits AFTER the data chunk (as many editors write it)."""
    import struct as _s

    x = tone(seconds, sr=48000, noise=0.01)
    pcm = (x * 32767).astype("<i2").tobytes()
    fmt = _s.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16)
    body = b"WAVE" + b"fmt " + _s.pack("<I", len(fmt)) + fmt + b"data" + _s.pack("<I", len(pcm)) + pcm
    body += b"LIST" + _s.pack("<I", len(list_payload)) + list_payload + (b"\x00" if len(list_payload) & 1 else b"")
    p = tmp_path / "big.wav"
    p.write_bytes(b"RIFF" + _s.pack("<I", len(body)) + body)
    return p


def test_large_wav_is_not_flagged_truncated(tmp_path):
    info = b"INFO" + b"ISFT" + (len(b"Audacity\x00")).to_bytes(4, "little") + b"Audacity\x00"
    p = _big_wav_with_trailing_list(tmp_path, info)
    assert p.stat().st_size > 5 * 1024 * 1024
    f = container.check_riff_structure(_ctx(p))
    assert f.status == FindingStatus.INFO, f.detail
    assert f.data["issues"] == []
    assert f.data["info"]["ISFT"] == "Audacity"  # chunk located after the 4 MB window is still read


def test_large_wav_generator_tag_after_data_chunk_is_found(tmp_path):
    info = b"INFO" + b"ISFT" + (len(b"ElevenLabs API\x00")).to_bytes(4, "little") + b"ElevenLabs API\x00"
    f = container.check_riff_structure(_ctx(_big_wav_with_trailing_list(tmp_path, info)))
    assert f.status == FindingStatus.WARN and f.data["generator"] == "elevenlabs"


def test_truncated_large_wav_still_detected(tmp_path):
    p = _big_wav_with_trailing_list(tmp_path, b"INFO")
    p.write_bytes(p.read_bytes()[: 5 * 1024 * 1024])
    assert container.check_riff_structure(_ctx(p)).status == FindingStatus.WARN


def test_multiple_list_chunks_merged(tmp_path):
    a = b"INFO" + b"ISFT" + (len(b"ToolA\x00")).to_bytes(4, "little") + b"ToolA\x00"
    b = b"INFO" + b"ICOP" + (len(b"(c) X\x00")).to_bytes(4, "little") + b"(c) X\x00"
    p = write_wav_with_chunks(tmp_path / "m.wav", tone(0.2), extra=[(b"LIST", a), (b"LIST", b)])
    f = container.check_riff_structure(_ctx(p))
    assert f.data["info"]["ISFT"] == "ToolA"
    assert f.data["info"]["ICOP"] == "(c) X"          # the second LIST chunk is merged, not dropped
