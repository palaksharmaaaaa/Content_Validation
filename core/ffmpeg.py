"""core.ffmpeg: how this project calls ffmpeg/ffprobe on files it did not make, and how it reads PCM.

ffmpeg picks the demuxer from a file's content, not its extension, so an upload named ``x.mp3`` can be a playlist that points at
URLs or other files. Every call therefore allows only the ``file`` and ``pipe`` protocols, never reads the terminal, and can be
limited to the first seconds of the input.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

SAFE_PROTOCOLS = "file,pipe"


def ffmpeg_input(path: str | Path, max_seconds: Optional[float] = None) -> List[str]:
    """ffmpeg arguments up to and including ``-i <path>``; add the output options after them."""
    args = ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", SAFE_PROTOCOLS]
    if max_seconds is not None:
        args += ["-t", f"{float(max_seconds):.3f}"]
    return args + ["-i", str(path)]


def ffprobe_input(path: str | Path, binary: str = "ffprobe") -> List[str]:
    """ffprobe arguments up to and including the input; add ``-show_entries`` and friends after them."""
    return [binary, "-v", "error", "-protocol_whitelist", SAFE_PROTOCOLS, "-i", str(path)]


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
