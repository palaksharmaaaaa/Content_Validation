"""core.ffmpeg: how this project calls ffmpeg/ffprobe on files it did not make, and how it reads PCM.

ffmpeg picks the demuxer from a file's content, not its extension, so an upload named ``x.mp3`` can be a playlist that points at
URLs or other files. Every call therefore allows only the ``file`` and ``pipe`` protocols, never reads the terminal, and can be
limited to the first seconds of the input. Only the container formats in ``SAFE_DEMUXERS`` are opened, because a playlist-type
demuxer (concat, HLS, DASH) would read other files named inside the upload.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

SAFE_PROTOCOLS = "file,pipe"
# The container formats this project accepts. Playlist-like demuxers (concat, HLS, DASH, ...) are left out on purpose: they read other files
# named inside the upload, and ``-protocol_whitelist file,pipe`` still lets them read local or network-share files.
SAFE_DEMUXERS = "wav,mp3,aac,flac,ogg,mov,mp4,m4a,matroska,webm,avi"


def ffmpeg_input(path: str | Path, max_seconds: Optional[float] = None) -> List[str]:
    """ffmpeg arguments up to and including ``-i <path>``; add the output options after them."""
    args = ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", SAFE_PROTOCOLS, "-format_whitelist", SAFE_DEMUXERS]
    if max_seconds is not None:
        args += ["-t", f"{float(max_seconds):.3f}"]
    return args + ["-i", str(path)]


def ffprobe_input(path: str | Path, binary: str = "ffprobe") -> List[str]:
    """ffprobe arguments up to and including the input; add ``-show_entries`` and friends after them."""
    return [binary, "-v", "error", "-protocol_whitelist", SAFE_PROTOCOLS, "-format_whitelist", SAFE_DEMUXERS, "-i", str(path)]


EBML_MAGIC = bytes([0x1A, 0x45, 0xDF, 0xA3])      # Matroska / WebM
_ISO_BOX_TYPES = (b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"styp", b"pnot")


def looks_like_video_container(path: str | Path) -> bool:
    """True if the file starts like an MP4/MOV (ISO base media), Matroska/WebM or AVI container. OpenCV's FFmpeg backend decides the
    format from the content, not the name, and would follow a playlist (HLS, concat, ...) to the files or URLs it names, so a file that is
    not one of the supported containers is never handed to it."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return (len(head) >= 8 and head[4:8] in _ISO_BOX_TYPES) or head[:4] == EBML_MAGIC or (head[:4] == b"RIFF" and head[8:12] == b"AVI ")


def open_video(path: str | Path):
    """``cv2.VideoCapture`` for a local file in a supported container; for anything else an unopened capture (``isOpened()`` is False).
    Every place that opens a video goes through here."""
    import cv2

    return cv2.VideoCapture(str(path)) if looks_like_video_container(path) else cv2.VideoCapture()


def pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    """Little-endian integer PCM of 1, 2, 3 or 4 bytes per sample as float32 in [-1, 1]. Any other width is an error, never a guess."""
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        b = np.frombuffer(raw[: len(raw) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        return np.where(v & 0x800000, v - 0x1000000, v).astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw[: len(raw) // 4 * 4], dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")
