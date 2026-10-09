"""
image_detector.dimension_checks._common: byte-level helpers shared by the image dimension checks.

Pure stdlib + Pillow + numpy + cv2. All scans are bounded (WINDOW bytes head/tail).
"""
from __future__ import annotations

import logging

import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from core.forensics.bytescan import read_windows

logger = logging.getLogger(__name__)

FULL_READ_LIMIT = 64 * 1024 * 1024

EXT_TO_FORMAT = {
    ".jpg": "jpeg", ".jpeg": "jpeg", ".jpe": "jpeg", ".jfif": "jpeg",
    ".png": "png", ".gif": "gif", ".webp": "webp", ".bmp": "bmp",
    ".tif": "tiff", ".tiff": "tiff", ".svg": "svg",
}


def sniff_format(head: bytes) -> str:
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:2] == b"BM":
        return "bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftypavif", b"ftyphevc", b"ftypmsf1"):
        return "heif"
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")[:2048].lower()
    if stripped.startswith(b"<?xml") or stripped.startswith(b"<svg") or b"<svg" in stripped:
        return "svg"
    return "unknown"


def jpeg_end_offset(data: bytes) -> Optional[int]:
    """Offset just past the first image's EOI marker (None if the stream is malformed)."""
    if not data.startswith(b"\xff\xd8"):
        return None
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0x01,) or 0xD0 <= marker <= 0xD8:
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        i += 2 + length
        if marker == 0xDA:  # SOS: entropy-coded data follows; first FFD9 after it is the EOI
            end = data.find(b"\xff\xd9", i)
            return end + 2 if end != -1 else None
    return None


def png_end_offset(data: bytes) -> Optional[int]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    i, n = 8, len(data)
    while i + 12 <= n:
        (length,) = struct.unpack(">I", data[i : i + 4])
        ctype = data[i + 4 : i + 8]
        i += 12 + length
        if ctype == b"IEND":
            return i
    return None


_PNG_TEXT_CHUNKS = {"tEXt", "zTXt", "iTXt", "eXIf"}


def parse_png_chunks(path: Path, max_chunks: int = 4096, body_limit: int = 1_000_000) -> List[Tuple[str, bytes]]:
    """
    PNG chunk walk that SEEKS through the file, so text/metadata chunks located after a large IDAT are seen
    regardless of file size. Only text/EXIF chunk bodies are read (up to ``body_limit`` bytes); other chunks
    are reported with an empty body.
    """
    out: List[Tuple[str, bytes]] = []
    size = Path(path).stat().st_size
    with open(path, "rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            return out
        pos = 8
        while pos + 12 <= size and len(out) < max_chunks:
            f.seek(pos)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            (length,) = struct.unpack(">I", hdr[:4])
            ctype = hdr[4:8].decode("latin-1")
            body = f.read(min(length, body_limit)) if ctype in _PNG_TEXT_CHUNKS else b""
            out.append((ctype, body))
            pos += 12 + length
            if ctype == "IEND":
                break
    return out


def _inflate_bounded(data: bytes, max_len: int) -> bytes:
    """zlib-decompress at most ``max_len`` bytes: a few kilobytes of a crafted text chunk can otherwise expand to gigabytes."""
    return zlib.decompressobj().decompress(data, max_len)


def png_text_fields(chunks: List[Tuple[str, bytes]], max_len: int = 200_000) -> Dict[str, str]:
    """tEXt / zTXt / iTXt key -> text (decoded leniently, truncated)."""
    out: Dict[str, str] = {}
    for ctype, body in chunks:
        try:
            if ctype == "tEXt":
                key, _, val = body.partition(b"\x00")
                out[key.decode("latin-1")] = val.decode("latin-1", errors="replace")[:max_len]
            elif ctype == "zTXt":
                key, _, rest = body.partition(b"\x00")
                out[key.decode("latin-1")] = _inflate_bounded(rest[1:], max_len).decode("utf-8", errors="replace")
            elif ctype == "iTXt":
                key, _, rest = body.partition(b"\x00")
                comp_flag = rest[0:1]
                rest = rest[2:]
                _lang, _, rest = rest.partition(b"\x00")
                _tkey, _, text = rest.partition(b"\x00")
                if comp_flag == b"\x01":
                    text = _inflate_bounded(text, max_len)
                out[key.decode("latin-1")] = text.decode("utf-8", errors="replace")[:max_len]
        except Exception:  # malformed chunk: skip, never raise
            continue
    return out


def extract_xmp(data: bytes) -> str:
    start = data.find(b"<x:xmpmeta")
    if start == -1:
        start = data.find(b"<?xpacket")
    if start == -1:
        return ""
    end = data.find(b"</x:xmpmeta>", start)
    end = end + len(b"</x:xmpmeta>") if end != -1 else min(len(data), start + 262144)
    return data[start:end].decode("utf-8", errors="replace")


def collect_text_fields(path: Path) -> Dict[str, str]:
    """Free-text metadata fields from PNG chunks, EXIF string tags, and XMP."""
    fields: Dict[str, str] = {}
    head, _tail, _size = read_windows(path)
    fmt = sniff_format(head)
    if fmt == "png":
        for k, v in png_text_fields(parse_png_chunks(path)).items():
            fields[f"png:{k}"] = v
    xmp = extract_xmp(head)
    if xmp:
        fields["xmp"] = xmp
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            for tag, name in ((0x9286, "UserComment"), (0x010E, "ImageDescription"), (0x013B, "Artist"),
                              (0x8298, "Copyright"), (0x0131, "Software"), (0x9C9C, "XPComment")):
                val = exif.get(tag)
                if val is None:
                    try:
                        val = exif.get_ifd(0x8769).get(tag)
                    except Exception:
                        val = None
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace").replace("\x00", "")
                if val:
                    fields[f"exif:{name}"] = str(val)[:20000]
    except Exception as exc:
        logger.debug("collect_text_fields: ignored %s: %s", type(exc).__name__, exc)
    return fields


def extract_exif_thumbnail(path: Path) -> Optional[bytes]:
    """Returns the IFD1 embedded JPEG thumbnail bytes from a JPEG's Exif APP1 segment, if any."""
    with open(path, "rb") as f:
        data = f.read(512 * 1024)
    if not data.startswith(b"\xff\xd8"):
        return None
    i, n = 2, len(data)
    while i + 4 <= n and data[i] == 0xFF:
        marker = data[i + 1]
        if marker in (0xDA, 0xD9):
            break
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        seg = data[i + 4 : i + 2 + length]
        i += 2 + length
        if marker != 0xE1 or not seg.startswith(b"Exif\x00\x00"):
            continue
        tiff = seg[6:]
        if tiff[:2] not in (b"II", b"MM"):
            return None
        e = "<" if tiff[:2] == b"II" else ">"
        try:
            (ifd0,) = struct.unpack(e + "I", tiff[4:8])
            (n0,) = struct.unpack(e + "H", tiff[ifd0 : ifd0 + 2])
            (ifd1,) = struct.unpack(e + "I", tiff[ifd0 + 2 + n0 * 12 : ifd0 + 6 + n0 * 12])
            if ifd1 == 0 or ifd1 >= len(tiff):
                return None
            (n1,) = struct.unpack(e + "H", tiff[ifd1 : ifd1 + 2])
            offset = length_ = None
            for k in range(min(n1, 64)):
                base = ifd1 + 2 + k * 12
                tag, _typ, _cnt, val = struct.unpack(e + "HHII", tiff[base : base + 12])
                if tag == 0x0201:
                    offset = val
                elif tag == 0x0202:
                    length_ = val
            if offset is None or length_ is None or offset + length_ > len(tiff):
                return None
            return bytes(tiff[offset : offset + length_])
        except struct.error:
            return None
    return None


# ---- JPEG quantization helpers -------------------------------------------------------------
_STD_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]
_STD_CHROMA = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
]
_ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
]


def libjpeg_table(base: List[int], quality: int) -> List[int]:
    q = max(1, min(100, int(quality)))
    scale = 5000 // q if q < 50 else 200 - 2 * q
    return [max(1, min(255, (b * scale + 50) // 100)) for b in base]


def estimate_libjpeg_quality(tables: Dict[int, List[int]]) -> Optional[int]:
    """Quality (1-100) if the luma table equals the libjpeg-scaled standard table, else None."""
    luma = tables.get(0)
    if not luma or len(luma) != 64:
        return None
    luma = [int(v) for v in luma]
    zig = [luma[_ZIGZAG.index(i)] for i in range(64)]  # tolerate zigzag-ordered storage
    for q in range(1, 101):
        ref = libjpeg_table(_STD_LUMA, q)
        if luma == ref or zig == ref:
            return q
    return None


# ---- perceptual hashes ---------------------------------------------------------------------
def _gray_array(im: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(im.convert("L").resize(size, Image.LANCZOS), dtype=np.float32)


def dhash64(im: Image.Image) -> int:
    g = _gray_array(im, (9, 8))
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def phash64(im: Image.Image) -> int:
    g = _gray_array(im, (32, 32))
    dct = cv2.dct(g)[:8, :8].flatten()
    med = float(np.median(dct[1:]))
    bits = dct > med
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming64(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def to_hex64(v: int) -> str:
    return f"{int(v):016x}"
