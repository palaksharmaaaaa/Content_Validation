"""
video_detector.dimension_checks._common: ISOBMFF (MP4/MOV), EBML (MKV/WebM) and RIFF-AVI helpers for the
video dimension checks. Pure stdlib + numpy; every parser is bounded and returns partial results or None
instead of raising on malformed input.
"""
from __future__ import annotations

from core.ffmpeg import open_video
from core.filecache import stat_cached

import re
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.forensics.bytescan import extract_xmp, read_windows

EXT_TO_FORMAT = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov", ".avi": "avi", ".mkv": "mkv", ".webm": "mkv", ".3gp": "mp4"}
MAX_BOXES = 20000
MOOV_LIMIT = 32 * 1024 * 1024
MAX_TABLE_ENTRIES = 2_000_000
MAC_EPOCH = datetime(1904, 1, 1)

C2PA_UUID = bytes.fromhex("d8fec3d61b0e483c92975828877ec481")

_CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta", b"mvex", b"moof", b"traf", b"ilst"}


def sniff_video_format(head: bytes) -> str:
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "mov" if head[8:12] == b"qt  " else "mp4"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "mkv"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "avi"
    return "unknown"


# ---- box walking ---------------------------------------------------------------------------------
@stat_cached(16)
def walk_top_level(path: Path, max_boxes: int = MAX_BOXES) -> Dict[str, Any]:
    """Top-level ISOBMFF boxes read by seeking (no full-file load)."""
    size = path.stat().st_size
    boxes: List[Dict[str, Any]] = []
    issues: List[str] = []
    pos = 0
    with open(path, "rb") as f:
        while pos + 8 <= size and len(boxes) < max_boxes:
            f.seek(pos)
            hdr = f.read(16)
            if len(hdr) < 8:
                break
            sz, typ = struct.unpack(">I4s", hdr[:8])
            header = 8
            if sz == 1:
                if len(hdr) < 16:
                    issues.append("truncated 64-bit box header")
                    break
                (sz,) = struct.unpack(">Q", hdr[8:16])
                header = 16
            elif sz == 0:
                sz = size - pos
            if sz < header:
                issues.append(f"invalid box size at offset {pos}")
                break
            t = typ.decode("latin-1")
            boxes.append({"type": t, "offset": pos, "size": sz, "header": header})
            if pos + sz > size:
                issues.append(f"box '{t}' at offset {pos} extends {pos + sz - size:,} bytes past the end of the file (truncated)")
                pos = size
                break
            pos += sz
    return {"boxes": boxes, "issues": issues, "end_offset": pos, "file_size": size}


def read_range(path: Path, offset: int, length: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(length)


def _parse_children(buf: bytes, start: int, end: int, depth: int = 0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pos = start
    while pos + 8 <= end and len(out) < 4096:
        sz, typ = struct.unpack(">I4s", buf[pos : pos + 8])
        header = 8
        if sz == 1 and pos + 16 <= end:
            (sz,) = struct.unpack(">Q", buf[pos + 8 : pos + 16])
            header = 16
        elif sz == 0:
            sz = end - pos
        if sz < header or pos + sz > end:
            break
        node = {"type": typ, "start": pos + header, "end": pos + sz, "children": []}
        if depth < 8:
            if typ in _CONTAINER_BOXES:
                node["children"] = _parse_children(buf, node["start"], node["end"], depth + 1)
            elif typ == b"meta":
                s = node["start"]
                if buf[s + 8 : s + 12] == b"hdlr":
                    s += 4  # ISO full-box meta
                node["children"] = _parse_children(buf, s, node["end"], depth + 1)
        out.append(node)
        pos += sz
    return out


def _find(nodes: List[Dict[str, Any]], typ: bytes) -> Optional[Dict[str, Any]]:
    for n in nodes:
        if n["type"] == typ:
            return n
    return None


def _find_all(nodes: List[Dict[str, Any]], typ: bytes) -> List[Dict[str, Any]]:
    return [n for n in nodes if n["type"] == typ]


def mac_time(seconds: int) -> Optional[datetime]:
    if seconds <= 0:
        return None
    try:
        return MAC_EPOCH + timedelta(seconds=int(seconds))
    except OverflowError:
        return None


def _u32s(buf: bytes, start: int, count: int) -> np.ndarray:
    count = max(0, min(count, MAX_TABLE_ENTRIES, (len(buf) - start) // 4))
    return np.frombuffer(buf[start : start + 4 * count], dtype=">u4").astype(np.uint64)


def _text_from_item(buf: bytes, node: Dict[str, Any]) -> Optional[str]:
    # iTunes-style: <key> -> <data> (8 bytes type/locale, then text)
    data = _find(node["children"], b"data") if node["children"] else None
    if data is None and node["end"] - node["start"] >= 8 and node["type"].startswith(b"\xa9"):
        # QuickTime-style: 2-byte length, 2-byte language, text
        body = buf[node["start"] : node["end"]]
        if len(body) >= 4:
            (ln,) = struct.unpack(">H", body[:2])
            return body[4 : 4 + ln].decode("utf-8", errors="replace").strip("\x00 ")
        return None
    if data is not None:
        raw = buf[data["start"] + 8 : data["end"]]
        return raw.decode("utf-8", errors="replace").strip("\x00 ")
    return None


def _collect_text(buf: bytes, nodes: List[Dict[str, Any]], out: Dict[str, str]) -> None:
    for n in nodes:
        if n["type"] in (b"udta", b"meta"):
            _collect_text(buf, n["children"], out)
        elif n["type"] == b"ilst":
            for item in n["children"]:
                # ilst children are boxes whose own children (e.g. 'data') are not auto-parsed: parse here
                item["children"] = _parse_children(buf, item["start"], item["end"], 7)
                txt = _text_from_item(buf, item)
                if txt:
                    out[item["type"].decode("latin-1")] = txt[:20000]
        else:
            txt = _text_from_item(buf, n)
            if txt and n["type"][:1] == b"\xa9":
                out[n["type"].decode("latin-1")] = txt[:20000]


def _fullbox_u32_u64(buf: bytes, ps: int, n_fields: List[str]) -> Tuple[int, ...]:
    version = buf[ps]
    off = ps + 4
    vals = []
    for width in n_fields:
        if width == "t":  # time field: 4 or 8 bytes by version
            if version == 1:
                vals.append(struct.unpack(">Q", buf[off : off + 8])[0])
                off += 8
            else:
                vals.append(struct.unpack(">I", buf[off : off + 4])[0])
                off += 4
        else:  # fixed 4-byte
            vals.append(struct.unpack(">I", buf[off : off + 4])[0])
            off += 4
    return tuple(vals)


def parse_moov(buf: bytes) -> Dict[str, Any]:
    """Interprets a moov payload (without its own header). Never raises; returns what could be read."""
    out: Dict[str, Any] = {"mvhd": None, "tracks": [], "text_items": {}, "issues": []}
    try:
        top = _parse_children(buf, 0, len(buf))
        mvhd = _find(top, b"mvhd")
        if mvhd:
            c, m, ts, dur = _fullbox_u32_u64(buf, mvhd["start"], ["t", "t", "u", "t"])
            out["mvhd"] = {"creation": c, "modification": m, "timescale": ts, "duration": dur}
        _collect_text(buf, top, out["text_items"])
        for trak in _find_all(top, b"trak"):
            out["tracks"].append(_parse_track(buf, trak))
    except Exception as exc:  # malformed moov: report, keep partial result
        out["issues"].append(f"moov parse stopped: {type(exc).__name__}")
    return out


def _parse_track(buf: bytes, trak: Dict[str, Any]) -> Dict[str, Any]:
    t: Dict[str, Any] = {"handler": None, "handler_name": "", "sample_entry": None, "mdhd": None, "tkhd": None,
                         "stts": None, "ctts_negative": False, "stsz": None, "stss": None, "stco": None, "stsc": None,
                         "fiel": None, "issues": []}
    kids = trak["children"]
    tkhd = _find(kids, b"tkhd")
    if tkhd:
        try:
            c, m, tid = _fullbox_u32_u64(buf, tkhd["start"], ["t", "t", "u"])
            t["tkhd"] = {"creation": c, "modification": m, "track_id": tid}
        except struct.error:
            pass
    mdia = _find(kids, b"mdia")
    if not mdia:
        return t
    mk = mdia["children"]
    mdhd = _find(mk, b"mdhd")
    if mdhd:
        try:
            c, m, ts, dur = _fullbox_u32_u64(buf, mdhd["start"], ["t", "t", "u", "t"])
            t["mdhd"] = {"creation": c, "modification": m, "timescale": ts, "duration": dur}
        except struct.error:
            pass
    hdlr = _find(mk, b"hdlr")
    if hdlr:
        body = buf[hdlr["start"] : hdlr["end"]]
        if len(body) >= 24:
            t["handler"] = body[8:12].decode("latin-1")
            raw = body[24:]
            if raw and raw[0] == len(raw) - 1:
                raw = raw[1:]
            t["handler_name"] = raw.split(b"\x00")[0].decode("utf-8", errors="replace").strip()
    minf = _find(mk, b"minf")
    stbl = _find(minf["children"], b"stbl") if minf else None
    if not stbl:
        return t
    sk = stbl["children"]
    stsd = _find(sk, b"stsd")
    if stsd:
        s = stsd["start"] + 8  # version/flags + entry_count
        if s + 8 <= stsd["end"]:
            esz, etype = struct.unpack(">I4s", buf[s : s + 8])
            t["sample_entry"] = etype.decode("latin-1")
            if t["handler"] == "vide" and esz >= 86:
                for child in _parse_children(buf, s + 86, min(stsd["end"], s + esz)):
                    if child["type"] == b"fiel" and child["end"] - child["start"] >= 2:
                        t["fiel"] = {"count": buf[child["start"]], "order": buf[child["start"] + 1]}
    stts = _find(sk, b"stts")
    if stts:
        (n,) = struct.unpack(">I", buf[stts["start"] + 4 : stts["start"] + 8])
        arr = _u32s(buf, stts["start"] + 8, n * 2).reshape(-1, 2) if n else np.zeros((0, 2), np.uint64)
        t["stts"] = arr
    ctts = _find(sk, b"ctts")
    if ctts:
        ver = buf[ctts["start"]]
        (n,) = struct.unpack(">I", buf[ctts["start"] + 4 : ctts["start"] + 8])
        arr = _u32s(buf, ctts["start"] + 8, n * 2).reshape(-1, 2) if n else np.zeros((0, 2), np.uint64)
        if ver == 1 and len(arr):
            t["ctts_negative"] = bool(np.any(arr[:, 1].astype(np.int64) > 0x7FFFFFFF))
    stsz = _find(sk, b"stsz")
    if stsz:
        sample_size, count = struct.unpack(">II", buf[stsz["start"] + 4 : stsz["start"] + 12])
        t["stsz"] = {"sample_size": sample_size, "count": count}
    stss = _find(sk, b"stss")
    if stss:
        (n,) = struct.unpack(">I", buf[stss["start"] + 4 : stss["start"] + 8])
        t["stss"] = _u32s(buf, stss["start"] + 8, n)
    stco = _find(sk, b"stco")
    co64 = _find(sk, b"co64")
    if stco:
        (n,) = struct.unpack(">I", buf[stco["start"] + 4 : stco["start"] + 8])
        t["stco"] = _u32s(buf, stco["start"] + 8, n)
    elif co64:
        (n,) = struct.unpack(">I", buf[co64["start"] + 4 : co64["start"] + 8])
        n = min(n, MAX_TABLE_ENTRIES, (co64["end"] - co64["start"] - 8) // 8)
        t["stco"] = np.frombuffer(buf[co64["start"] + 8 : co64["start"] + 8 + 8 * n], dtype=">u8").astype(np.uint64)
    stsc = _find(sk, b"stsc")
    if stsc:
        (n,) = struct.unpack(">I", buf[stsc["start"] + 4 : stsc["start"] + 8])
        t["stsc"] = _u32s(buf, stsc["start"] + 8, n * 3).reshape(-1, 3) if n else np.zeros((0, 3), np.uint64)
    return t


@stat_cached(16)
def load_moov(path: Path, top: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parsed moov for a file whose top-level walk is ``top`` (None if absent or too large)."""
    for b in top["boxes"]:
        if b["type"] == "moov" and b["size"] <= MOOV_LIMIT and b["offset"] + b["size"] <= top["file_size"]:
            data = read_range(path, b["offset"] + b["header"], b["size"] - b["header"])
            return parse_moov(data)
    return None


def total_samples_from_stsc(stsc: np.ndarray, n_chunks: int) -> Optional[int]:
    if stsc is None or len(stsc) == 0 or n_chunks <= 0:
        return None
    total = 0
    for i, (first, per, _d) in enumerate(stsc):
        nxt = int(stsc[i + 1][0]) if i + 1 < len(stsc) else n_chunks + 1
        total += (nxt - int(first)) * int(per)
    return int(total)


# ---- bitstream strings -------------------------------------------------------------------------------
_X264 = re.compile(rb"x264 - core \d+[ -~]{0,400}")
_X265 = re.compile(rb"x265 \(build \d+\)[ -~]{0,400}")
_LAVC = re.compile(rb"Lavc[0-9]+\.[0-9]+\.[0-9]+")


def find_encoder_strings(data: bytes) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, rx in (("x264", _X264), ("x265", _X265), ("libavcodec", _LAVC)):
        m = rx.search(data)
        if m:
            out[key] = m.group(0).decode("latin-1")[:300]
    return out


# ---- EBML (MKV / WebM) --------------------------------------------------------------------------------
def _ebml_string_after(head: bytes, id_bytes: bytes) -> Optional[str]:
    i = head.find(id_bytes)
    if i == -1 or i + len(id_bytes) >= len(head):
        return None
    first = head[i + len(id_bytes)]
    length = 1
    mask = 0x80
    while length <= 8 and not (first & mask):
        length += 1
        mask >>= 1
    if length > 4:
        return None
    size = first & (mask - 1)
    for k in range(1, length):
        size = (size << 8) | head[i + len(id_bytes) + k]
    start = i + len(id_bytes) + length
    if size > 256 or start + size > len(head):
        return None
    return head[start : start + size].decode("utf-8", errors="replace").strip("\x00 ")


def ebml_info(head: bytes) -> Dict[str, Optional[str]]:
    return {
        "doc_type": _ebml_string_after(head[:512], b"\x42\x82"),
        "muxing_app": _ebml_string_after(head, b"\x4d\x80"),
        "writing_app": _ebml_string_after(head, b"\x57\x41"),
    }


# ---- text collection -----------------------------------------------------------------------------------
def collect_text_fields(path: Path) -> Dict[str, str]:
    """Free-text metadata from MP4/MOV udta/ilst items and any XMP packet."""
    head, tail, _size = read_windows(path)
    fields: Dict[str, str] = {}
    if sniff_video_format(head) in ("mp4", "mov"):
        top = walk_top_level(path)
        moov = load_moov(path, top)
        if moov:
            for k, v in moov["text_items"].items():
                fields[f"mp4:{k}"] = v
    xmp = extract_xmp(head) or extract_xmp(tail)
    if xmp:
        fields["xmp"] = xmp
    return fields


# ---- frame reading -------------------------------------------------------------------------------------
def read_consecutive_gray(path: Path, n: int = 90, max_width: int = 480) -> List[np.ndarray]:
    """First ``n`` consecutive frames as grayscale uint8 arrays no wider than ``max_width`` (empty list on failure)."""
    import cv2

    cap = open_video(path)
    frames: List[np.ndarray] = []
    try:
        if not cap.isOpened():
            return frames
        while len(frames) < n:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if g.shape[1] > max_width:
                scale = max_width / g.shape[1]
                g = cv2.resize(g, (max_width, max(1, int(g.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            frames.append(g)
    finally:
        cap.release()
    return frames


def read_spread_gray(path: Path, n: int = 16, max_width: int = 320) -> List[np.ndarray]:
    """Up to ``n`` evenly spaced grayscale frames across the whole clip (empty list on failure)."""
    import cv2

    cap = open_video(path)
    out: List[np.ndarray] = []
    try:
        if not cap.isOpened():
            return out
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sorted(set(int(i) for i in np.linspace(0, max(0, total - 2), n))) if total > 1 else [0]
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if g.shape[1] > max_width:
                scale = max_width / g.shape[1]
                g = cv2.resize(g, (max_width, max(1, int(g.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            out.append(g)
    finally:
        cap.release()
    return out


def phash64_gray(gray: np.ndarray) -> int:
    import cv2

    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(g)[:8, :8].flatten()
    bits = dct > float(np.median(dct[1:]))
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming64(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def stream_info(path: Path) -> Dict[str, Any]:
    """fps, frame count, width, height, duration via OpenCV (zeros on failure)."""
    import cv2

    cap = open_video(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    dur = frames / fps if fps > 0 else 0.0
    return {"fps": fps, "frames": frames, "width": w, "height": h, "duration": dur}
