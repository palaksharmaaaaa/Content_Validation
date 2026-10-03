"""Synthetic video fixtures: hand-built ISOBMFF boxes (no codec needed) and cv2/ffmpeg clip writers."""
import struct
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def box(t: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I4s", 8 + len(payload), t) + payload


def fullbox(t: bytes, payload: bytes = b"", version: int = 0, flags: int = 0) -> bytes:
    return box(t, bytes([version]) + flags.to_bytes(3, "big") + payload)


def mvhd(timescale=1000, duration=1000, creation=0, modification=0) -> bytes:
    return fullbox(b"mvhd", struct.pack(">IIII", creation, modification, timescale, duration) + b"\x00" * 80)


def tkhd(track_id=1, creation=0, modification=0) -> bytes:
    return fullbox(b"tkhd", struct.pack(">IIII", creation, modification, track_id, 0) + b"\x00" * 68, flags=3)


def mdhd(timescale=25000, duration=25000, creation=0, modification=0) -> bytes:
    return fullbox(b"mdhd", struct.pack(">IIII", creation, modification, timescale, duration) + b"\x55\xc4\x00\x00")


def hdlr(handler: bytes = b"vide", name: str = "VideoHandler") -> bytes:
    return fullbox(b"hdlr", b"\x00" * 4 + handler + b"\x00" * 12 + name.encode() + b"\x00")


def stts(entries: Sequence[Tuple[int, int]]) -> bytes:
    return fullbox(b"stts", struct.pack(">I", len(entries)) + b"".join(struct.pack(">II", c, d) for c, d in entries))


def stsz(count: int, sample_size: int = 100) -> bytes:
    return fullbox(b"stsz", struct.pack(">II", sample_size, count))


def stss(indices: Sequence[int]) -> bytes:
    return fullbox(b"stss", struct.pack(">I", len(indices)) + b"".join(struct.pack(">I", i) for i in indices))


def stco(offsets: Sequence[int]) -> bytes:
    return fullbox(b"stco", struct.pack(">I", len(offsets)) + b"".join(struct.pack(">I", o) for o in offsets))


def stsc(entries: Sequence[Tuple[int, int, int]]) -> bytes:
    return fullbox(b"stsc", struct.pack(">I", len(entries)) + b"".join(struct.pack(">III", *e) for e in entries))


def stsd(fourcc: bytes = b"avc1", children: bytes = b"") -> bytes:
    entry = box(fourcc, b"\x00" * 78 + children)
    return fullbox(b"stsd", struct.pack(">I", 1) + entry)


def fiel(count=2, order=1) -> bytes:
    return box(b"fiel", bytes([count, order]))


def trak(handler=b"vide", fourcc=b"avc1", handler_name="VideoHandler", timescale=25000, stts_entries=((25, 1000),),
         sample_count: Optional[int] = None, stss_idx=(1,), chunk_offsets=(0,), sample_per_chunk: Optional[int] = None,
         entry_children=b"", track_creation=0, track_modification=0, media_creation=0, media_modification=0) -> bytes:
    total = sum(c for c, _ in stts_entries)
    count = total if sample_count is None else sample_count
    per = count if sample_per_chunk is None else sample_per_chunk
    stbl = box(b"stbl", stsd(fourcc, entry_children) + stts(stts_entries) + stsz(count) + stss(stss_idx)
               + stsc([(1, per, 1)]) + stco(chunk_offsets))
    minf = box(b"minf", stbl)
    mdia = box(b"mdia", mdhd(timescale, total * 1000, media_creation, media_modification) + hdlr(handler, handler_name) + minf)
    return box(b"trak", tkhd(1, track_creation, track_modification) + mdia)


def text_atom(key: bytes, text: str) -> bytes:
    """QuickTime-style user-data text atom (2-byte length, 2-byte language, text)."""
    t = text.encode("utf-8")
    return box(key, struct.pack(">HH", len(t), 0x15C7) + t)


def ilst_item(key: bytes, text: str) -> bytes:
    data = box(b"data", struct.pack(">II", 1, 0) + text.encode("utf-8"))
    return box(key, data)


def build_mp4(tracks: Sequence[bytes] = (), moov_extra: bytes = b"", brand: bytes = b"isom", faststart: bool = False,
              mdat_payload: bytes = b"\x00" * 2000, mvhd_args: Optional[dict] = None, extra_top: bytes = b"") -> bytes:
    ftyp = box(b"ftyp", brand + struct.pack(">I", 512) + brand + b"iso2avc1mp41")
    mdat = box(b"mdat", mdat_payload)
    moov = box(b"moov", mvhd(**(mvhd_args or {})) + b"".join(tracks) + moov_extra)
    return ftyp + (moov + mdat if faststart else mdat + moov) + extra_top


def write_clip(path: Path, frames: List[np.ndarray], fps: float = 25.0) -> Path:
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        wr.write(f)
    wr.release()
    return Path(path)


def moving_square_frames(n=40, size=(160, 120), step=3) -> List[np.ndarray]:
    out = []
    for i in range(n):
        f = np.full((size[1], size[0], 3), 90, np.uint8)
        x = (i * step) % (size[0] - 30)
        f[40:70, x : x + 30] = (30, 200, 240)
        out.append(f)
    return out
