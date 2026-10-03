"""Synthetic audio fixtures (no external assets): WAV writer, ID3v2 builder, Ogg page builder."""
import struct
import wave
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from audio_detector.dimension_checks._common import ogg_crc


def tone(seconds=1.0, sr=16000, freq=440.0, amp=0.3, noise=0.0, seed=0):
    t = np.arange(int(seconds * sr)) / sr
    x = amp * np.sin(2 * np.pi * freq * t)
    if noise:
        x = x + noise * np.random.default_rng(seed).standard_normal(len(t))
    return np.clip(x, -1, 1).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sr=16000, bits=16) -> Path:
    path = Path(path)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setframerate(sr)
        if bits == 16:
            wf.setsampwidth(2)
            wf.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        elif bits == 24:
            wf.setsampwidth(3)
            v = (np.clip(samples, -1, 1) * 8388607).astype(np.int32)
            raw = np.stack([(v & 0xFF), ((v >> 8) & 0xFF), ((v >> 16) & 0xFF)], axis=1).astype(np.uint8).tobytes()
            wf.writeframes(raw)
        else:
            raise ValueError(bits)
    return path


def write_wav_with_chunks(path: Path, samples: np.ndarray, sr=16000, extra: Optional[Dict[bytes, bytes]] = None) -> Path:
    """16-bit mono WAV with additional RIFF chunks (e.g. bext, LIST) inserted before data."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, sr, sr * 2, 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    for cid, payload in (extra or {}).items():
        body += cid + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) & 1 else b"")
    body += b"data" + struct.pack("<I", len(pcm)) + pcm
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return Path(path)


def synchsafe(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def id3v23(frames: Dict[str, bytes], padding=0) -> bytes:
    body = b""
    for fid, payload in frames.items():
        body += fid.encode("ascii") + struct.pack(">I", len(payload)) + b"\x00\x00" + payload
    body += b"\x00" * padding
    return b"ID3" + bytes([3, 0, 0]) + synchsafe(len(body)) + body


def text_frame(text: str) -> bytes:
    return b"\x03" + text.encode("utf-8")  # UTF-8 encoding byte 3 is ID3v2.4, 0 (latin-1) is safe for v2.3


def text_frame_v23(text: str) -> bytes:
    return b"\x00" + text.encode("latin-1")


def txxx_v23(desc: str, value: str) -> bytes:
    return b"\x00" + desc.encode("latin-1") + b"\x00" + value.encode("latin-1")


def ogg_page(payload: bytes, serial=1, seq=0, granule=0, header_type=0, corrupt_crc=False) -> bytes:
    assert len(payload) < 255
    seg = bytes([len(payload)])
    hdr = b"OggS" + bytes([0, header_type]) + struct.pack("<q", granule) + struct.pack("<III", serial, seq, 0) + b"\x01" + seg
    page = bytearray(hdr + payload)
    crc = ogg_crc(bytes(page))
    if corrupt_crc:
        crc ^= 0xDEADBEEF
    page[22:26] = struct.pack("<I", crc)
    return bytes(page)
