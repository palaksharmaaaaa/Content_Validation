"""
audio_detector.dimension_checks._common: container parsers and native-rate loading used by the
audio dimension checks. Pure stdlib + numpy; every parser is bounded and returns None/partial
results instead of raising on malformed input.
"""
from __future__ import annotations

import logging

import struct
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.ffmpeg import ffmpeg_input, pcm_to_float
from core.forensics.bytescan import WINDOW, read_windows  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

EXT_TO_FORMAT = {
    ".wav": "wav", ".mp3": "mp3", ".aac": "aac", ".flac": "flac", ".ogg": "ogg", ".oga": "ogg",
    ".opus": "ogg", ".m4a": "m4a", ".aif": "aiff", ".aiff": "aiff",
}
MAX_OGG_BYTES = 16 * 1024 * 1024
FULL_READ_LIMIT = 64 * 1024 * 1024


def sniff_audio_format(head: bytes) -> str:
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[4:8] == b"ftyp":
        return "m4a"
    if head[:4] == b"MThd":
        return "midi"
    if head[:3] == b"ID3":
        return "mp3"
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "aac" if (head[1] & 0xF6) == 0xF0 else "mp3"
    return "unknown"


# ---- RIFF / WAV ------------------------------------------------------------------------------
def walk_riff(path: Path, max_chunks: int = 4096) -> Dict[str, Any]:
    """
    RIFF/WAVE chunk walk that SEEKS through the file (header-sized reads only), so structure after a large
    ``data`` chunk (LIST INFO, bext, cue lists) is seen regardless of file size.
    """
    out: Dict[str, Any] = {"ok": False, "form": None, "declared_size": None, "chunks": [], "issues": []}
    size = Path(path).stat().st_size
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            return out
        out["ok"] = True
        out["form"] = "WAVE"
        out["declared_size"] = struct.unpack("<I", head[4:8])[0]
        pos = 12
        while pos + 8 <= size and len(out["chunks"]) < max_chunks:
            f.seek(pos)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid = hdr[:4].decode("latin-1")
            (csize,) = struct.unpack("<I", hdr[4:8])
            out["chunks"].append({"id": cid, "offset": pos, "size": csize})
            if cid == "data" and csize == 0xFFFFFFFF:
                break                              # a streamed WAV leaves the data size unset: the data simply runs to the end of the file
            end = pos + 8 + csize
            if end > size:
                if cid == "data":
                    out["issues"].append("data chunk declares more bytes than the file contains (truncated)")
                else:
                    out["issues"].append(f"chunk '{cid}' extends past end of file")
                break
            pos = end + (csize & 1)
    return out


def riff_chunk_body(path: Path, chunk: Dict[str, Any], limit: int = 65536) -> bytes:
    with open(path, "rb") as f:
        f.seek(chunk["offset"] + 8)
        return f.read(min(chunk["size"], limit))


def parse_wav_fmt(body: bytes) -> Optional[Dict[str, int]]:
    if len(body) < 16:
        return None
    tag, ch, sr, _br, _ba, bits = struct.unpack("<HHIIHH", body[:16])
    if tag == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE: real tag in sub-format GUID
        tag = struct.unpack("<H", body[24:26])[0]
    return {"format_tag": tag, "channels": ch, "sample_rate": sr, "bits": bits}


def parse_bext(body: bytes) -> Dict[str, Any]:
    """EBU Tech 3285 Broadcast Audio Extension fields (ASCII, null padded)."""
    def txt(a: int, b: int) -> str:
        return body[a:b].split(b"\x00")[0].decode("ascii", errors="replace").strip()

    out = {
        "description": txt(0, 256), "originator": txt(256, 288), "originator_reference": txt(288, 320),
        "origination_date": txt(320, 330), "origination_time": txt(330, 338), "coding_history": "",
    }
    if len(body) > 602:
        out["coding_history"] = body[602:].split(b"\x00")[0].decode("ascii", errors="replace").strip()
    return out


def parse_list_info(body: bytes) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if body[:4] != b"INFO":
        return out
    i = 4
    while i + 8 <= len(body):
        key = body[i : i + 4].decode("latin-1")
        (size,) = struct.unpack("<I", body[i + 4 : i + 8])
        out[key] = body[i + 8 : i + 8 + size].split(b"\x00")[0].decode("utf-8", errors="replace").strip()
        i += 8 + size + (size & 1)
    return out


# ---- ID3v2 -----------------------------------------------------------------------------------
def _synchsafe(b: bytes) -> int:
    return ((b[0] & 0x7F) << 21) | ((b[1] & 0x7F) << 14) | ((b[2] & 0x7F) << 7) | (b[3] & 0x7F)


def _decode_text(enc: int, raw: bytes) -> str:
    try:
        if enc == 0:
            return raw.decode("latin-1")
        if enc == 1:
            return raw.decode("utf-16")
        if enc == 2:
            return raw.decode("utf-16-be")
        return raw.decode("utf-8")
    except Exception as exc:
        logger.debug("_decode_text: ignored %s: %s", type(exc).__name__, exc)
        return raw.decode("latin-1", errors="replace")


def parse_id3v2(data: bytes) -> Optional[Dict[str, Any]]:
    if len(data) < 10 or data[:3] != b"ID3":
        return None
    major, rev, flags = data[3], data[4], data[5]
    if major not in (2, 3, 4) or any(b & 0x80 for b in data[6:10]):
        return None
    tag_size = _synchsafe(data[6:10])
    body = data[10 : 10 + tag_size]
    id_len, hdr_len = (3, 6) if major == 2 else (4, 10)
    frames: List[Dict[str, Any]] = []
    text: Dict[str, str] = {}
    i, padding, has_apic = 0, 0, False
    while i + hdr_len <= len(body):
        fid = body[i : i + id_len]
        if fid[0] == 0:
            padding = len(body) - i
            break
        try:
            fid_s = fid.decode("ascii")
        except UnicodeDecodeError:
            break
        if major == 2:
            size = int.from_bytes(body[i + 3 : i + 6], "big")
        elif major == 3:
            size = struct.unpack(">I", body[i + 4 : i + 8])[0]
        else:
            size = _synchsafe(body[i + 4 : i + 8])
        payload = body[i + hdr_len : i + hdr_len + size]
        frames.append({"id": fid_s, "size": size})
        if fid_s in ("APIC", "PIC"):
            has_apic = True
        elif fid_s.startswith("T") and fid_s not in ("TXXX", "TXX") and payload:
            text[fid_s] = _decode_text(payload[0], payload[1:]).strip("\x00 ")
        elif fid_s in ("TXXX", "TXX") and payload:
            enc = payload[0]
            term = b"\x00\x00" if enc in (1, 2) else b"\x00"
            desc, _, val = payload[1:].partition(term)
            text[f"TXXX:{_decode_text(enc, desc)}"] = _decode_text(enc, val).strip("\x00 ")
        elif fid_s in ("COMM", "COM") and len(payload) > 4:
            enc = payload[0]
            term = b"\x00\x00" if enc in (1, 2) else b"\x00"
            _desc, _, val = payload[4:].partition(term)
            text[f"COMM:{_decode_text(enc, _desc)}"] = _decode_text(enc, val).strip("\x00 ")
        i += hdr_len + size
    footer = 10 if (major == 4 and flags & 0x10) else 0
    return {
        "version": f"2.{major}.{rev}", "flags": flags, "tag_size": tag_size, "frames": frames, "text": text,
        "padding_bytes": padding, "has_apic": has_apic, "total_len": 10 + tag_size + footer,
    }


# ---- FLAC / MP3 -------------------------------------------------------------------------------
def flac_streaminfo(data: bytes) -> Optional[Dict[str, Any]]:
    if data[:4] != b"fLaC" or len(data) < 4 + 4 + 34:
        return None
    if (data[4] & 0x7F) != 0:
        return None
    b = data[8 : 8 + 34]
    packed = int.from_bytes(b[10:18], "big")
    sample_rate = (packed >> 44) & 0xFFFFF
    channels = ((packed >> 41) & 0x7) + 1
    bps = ((packed >> 36) & 0x1F) + 1
    total = packed & 0xFFFFFFFFF
    return {"sample_rate": sample_rate, "channels": channels, "bits": bps, "total_samples": total, "md5": b[18:34].hex()}


_MP3_SR = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}
_MP3_BR_L3_V1 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
_MP3_BR_L3_V2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]


def mp3_first_frame(data: bytes, start: int = 0) -> Optional[Dict[str, Any]]:
    n = len(data)
    i = start
    while i + 4 <= min(n, start + 65536):
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            b1, b2, b3 = data[i + 1], data[i + 2], data[i + 3]
            ver = (b1 >> 3) & 0x3
            layer = (b1 >> 1) & 0x3
            sr_idx = (b2 >> 2) & 0x3
            br_idx = (b2 >> 4) & 0xF
            if ver != 1 and layer == 1 and sr_idx != 3 and br_idx not in (0, 15):
                table = _MP3_BR_L3_V1 if ver == 3 else _MP3_BR_L3_V2
                return {
                    "offset": i, "mpeg_version": {3: "1", 2: "2", 0: "2.5"}[ver], "layer": 3,
                    "sample_rate": _MP3_SR[ver][sr_idx], "bitrate_kbps": table[br_idx],
                    "channels": 1 if (b3 >> 6) == 3 else 2,
                }
        i += 1
    return None


# ---- Ogg --------------------------------------------------------------------------------------
def _ogg_crc_table() -> List[int]:
    table = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = ((r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if r & 0x80000000 else (r << 1) & 0xFFFFFFFF
        table.append(r)
    return table


_OGG_TABLE = _ogg_crc_table()


def ogg_crc(page: bytes) -> int:
    crc = 0
    for byte in page:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _OGG_TABLE[((crc >> 24) & 0xFF) ^ byte]
    return crc


def parse_ogg_pages(data: bytes, max_pages: int = 50000) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    i, n = 0, len(data)
    while i + 27 <= n and len(pages) < max_pages:
        if data[i : i + 4] != b"OggS":
            break
        nseg = data[i + 26]
        if i + 27 + nseg > n:
            break
        seg_total = sum(data[i + 27 : i + 27 + nseg])
        end = i + 27 + nseg + seg_total
        if end > n:
            pages.append({"offset": i, "truncated": True})
            break
        granule = struct.unpack("<q", data[i + 6 : i + 14])[0]
        serial, seq, crc = struct.unpack("<III", data[i + 14 : i + 26])
        raw = bytearray(data[i:end])
        raw[22:26] = b"\x00\x00\x00\x00"
        pages.append({
            "offset": i, "header_type": data[i + 5], "granule": granule, "serial": serial, "seq": seq,
            "crc_ok": ogg_crc(bytes(raw)) == crc, "truncated": False,
        })
        i = end
    return pages


# ---- native-rate loading ----------------------------------------------------------------------
def container_info(path: Path) -> Optional[Dict[str, Any]]:
    """Native {sample_rate, bits, channels, format} from the container header (None if unknown)."""
    head, _tail, size = read_windows(path, window=WINDOW)
    fmt = sniff_audio_format(head)
    if fmt == "wav":
        riff = walk_riff(path)
        for ch in riff["chunks"]:
            if ch["id"] == "fmt ":
                f = parse_wav_fmt(riff_chunk_body(path, ch))
                if f:
                    return {"format": "wav", "sample_rate": f["sample_rate"], "bits": f["bits"], "channels": f["channels"],
                            "format_tag": f["format_tag"]}
    elif fmt == "flac":
        si = flac_streaminfo(head)
        if si:
            return {"format": "flac", "sample_rate": si["sample_rate"], "bits": si["bits"], "channels": si["channels"]}
    elif fmt == "mp3":
        start = 0
        id3 = parse_id3v2(head)
        if id3:
            start = id3["total_len"]
        fr = mp3_first_frame(head, start)
        if fr:
            return {"format": "mp3", "sample_rate": fr["sample_rate"], "bits": None, "channels": fr["channels"],
                    "bitrate_kbps": fr["bitrate_kbps"]}
    elif fmt == "ogg":
        i = head.find(b"OpusHead")
        if i != -1:
            return {"format": "ogg-opus", "sample_rate": 48000, "bits": None, "channels": head[i + 9] if len(head) > i + 9 else None}
        j = head.find(b"\x01vorbis")
        if j != -1 and len(head) >= j + 15:
            ch = head[j + 11]
            (sr,) = struct.unpack("<I", head[j + 12 : j + 16])
            return {"format": "ogg-vorbis", "sample_rate": sr, "bits": None, "channels": ch}
    return None


def load_native_mono(path: Path, max_seconds: float = 30.0) -> Optional[Tuple[np.ndarray, int, Optional[int]]]:
    """
    Mono float32 samples at the file's native sample rate (first ``max_seconds``), plus the container's
    bit depth. WAV PCM is read natively; other formats need ffmpeg. Returns None if neither works.
    """
    info = container_info(path)
    if not info or not info.get("sample_rate"):
        return None
    sr, bits = int(info["sample_rate"]), info.get("bits")
    if info["format"] == "wav" and info.get("format_tag") == 1 and bits in (8, 16, 24, 32):
        try:
            with wave.open(str(path), "rb") as wf:
                n_frames = min(wf.getnframes(), int(max_seconds * sr))
                raw = wf.readframes(n_frames)
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
            x = pcm_to_float(raw, sw)
            if ch > 1:
                x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
            return x, sr, bits
        except Exception as exc:
            logger.debug("load_native_mono: ignored %s: %s", type(exc).__name__, exc)
            return None
    try:
        res = subprocess.run(
            ffmpeg_input(path, max_seconds=max_seconds) + ["-vn", "-ac", "1", "-f", "f32le", "-"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0 or not res.stdout:
        return None
    return np.frombuffer(res.stdout, dtype="<f4").astype(np.float32), sr, bits


# ---- free-text metadata collection -------------------------------------------------------------
def parse_vorbis_comment(body: bytes) -> Dict[str, str]:
    """Vorbis-comment payload (vendor string + user comments) -> {NAME: value}."""
    out: Dict[str, str] = {}
    try:
        (vlen,) = struct.unpack("<I", body[:4])
        out["VENDOR"] = body[4 : 4 + vlen].decode("utf-8", errors="replace")
        i = 4 + vlen
        (n,) = struct.unpack("<I", body[i : i + 4])
        i += 4
        for _ in range(min(n, 512)):
            (ln,) = struct.unpack("<I", body[i : i + 4])
            kv = body[i + 4 : i + 4 + ln].decode("utf-8", errors="replace")
            i += 4 + ln
            if "=" in kv:
                k, v = kv.split("=", 1)
                out[k.upper()] = v[:20000]
    except Exception as exc:
        logger.debug("parse_vorbis_comment: ignored %s: %s", type(exc).__name__, exc)
    return out


def collect_text_fields(path: Path) -> Dict[str, str]:
    """Free-text metadata from ID3v2, RIFF LIST INFO / bext, FLAC and Ogg Vorbis/Opus comments."""
    head, _tail, _size = read_windows(path)
    fmt = sniff_audio_format(head)
    fields: Dict[str, str] = {}
    id3 = parse_id3v2(head)
    if id3:
        for k, v in id3["text"].items():
            fields[f"id3:{k}"] = v[:20000]
    if fmt == "wav":
        riff = walk_riff(path)
        for ch in riff["chunks"]:
            if ch["id"] == "LIST":
                for k, v in parse_list_info(riff_chunk_body(path, ch)).items():
                    fields[f"riff:{k}"] = v
            elif ch["id"] == "bext":
                bx = parse_bext(riff_chunk_body(path, ch, limit=1 << 20))
                for k in ("description", "originator", "originator_reference", "coding_history"):
                    if bx.get(k):
                        fields[f"bext:{k}"] = bx[k]
    elif fmt == "flac":
        i = 4
        while i + 4 <= len(head):
            hdr = head[i]
            length = int.from_bytes(head[i + 1 : i + 4], "big")
            if (hdr & 0x7F) == 4:
                for k, v in parse_vorbis_comment(head[i + 4 : i + 4 + length]).items():
                    fields[f"vorbis:{k}"] = v
            i += 4 + length
            if hdr & 0x80:
                break
    elif fmt == "ogg":
        for marker in (b"\x03vorbis", b"OpusTags"):
            j = head.find(marker)
            if j != -1:
                for k, v in parse_vorbis_comment(head[j + len(marker) : j + len(marker) + 65536]).items():
                    fields[f"vorbis:{k}"] = v
                break
    return fields
