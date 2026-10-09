"""
audio_detector.dimension_checks.context: audio re-use fingerprinting for "real but misleading" checks
. The recording itself cannot reveal a false attribution; this check only computes
a coarse Haitsma-Kalker-style fingerprint (32 bits per 32 ms hop from 33 log-spaced bands, 300-3000 Hz)
and, if the operator maintains a local reference index, matches against it by best-offset bit error rate.

Index format (JSON lines): {"fp": "<hex of little-endian uint32 array>", "label": "...", "source": "..."}.
Build entries with ``fingerprint_hex(samples_16k_mono)``. Set OMNI_AUDIO_FP_INDEX or place the file at
audio_detector/data/audio_fingerprint_index.jsonl. This fingerprint is coarse: always verify matches by ear.
"""
from __future__ import annotations

import logging

from pathlib import Path
from typing import Any, Dict

import numpy as np

from core.forensics.config import audio_fp_index_file
from core.forensics.jsonl_index import load_index
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity

logger = logging.getLogger(__name__)

DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "data" / "audio_fingerprint_index.jsonl"
FRAME, HOP, BANDS = 4096, 512, 33
FMIN, FMAX = 300.0, 3000.0
MAX_SECONDS = 120
MATCH_BER = 0.30
MIN_OVERLAP_FRAMES = 100
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def fingerprint(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)[: int(MAX_SECONDS * sr)]
    n = (len(x) - FRAME) // HOP + 1
    if n < 3:
        return np.zeros(0, dtype=np.uint32)
    win = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / sr)
    edges = np.geomspace(FMIN, FMAX, BANDS + 1)
    idx = np.searchsorted(freqs, edges)
    energies = np.zeros((n, BANDS))
    for i in range(n):
        spec = np.abs(np.fft.rfft(x[i * HOP : i * HOP + FRAME] * win)) ** 2
        for b in range(BANDS):
            lo, hi = idx[b], max(idx[b + 1], idx[b] + 1)
            energies[i, b] = spec[lo:hi].sum()
    d = energies[:, :-1] - energies[:, 1:]
    bits = (d[1:] - d[:-1]) > 0
    weights = (1 << np.arange(32, dtype=np.uint64)).astype(np.uint64)
    return (bits.astype(np.uint64) * weights).sum(axis=1).astype(np.uint32)


def fingerprint_hex(x: np.ndarray, sr: int = 16000) -> str:
    return fingerprint(x, sr).astype("<u4").tobytes().hex()


def best_ber(query: np.ndarray, ref: np.ndarray) -> float:
    """Lowest bit error rate over all frame offsets with at least MIN_OVERLAP_FRAMES (1.0 if none)."""
    nq, nr = len(query), len(ref)
    min_overlap = min(MIN_OVERLAP_FRAMES, nq, nr)
    if min_overlap < 20:
        return 1.0
    best = 1.0
    for off in range(-(nq - min_overlap), nr - min_overlap + 1):
        a0, b0 = max(0, -off), max(0, off)
        length = min(nq - a0, nr - b0)
        if length < min_overlap:
            continue
        diff = (query[a0 : a0 + length] ^ ref[b0 : b0 + length]).view(np.uint8)
        ber = float(_POP[diff].sum()) / (32.0 * length)
        if ber < best:
            best = ber
    return best


def _prepare(row: Dict[str, Any]) -> None:
    row["_fp"] = np.frombuffer(bytes.fromhex(row["fp"]), dtype="<u4").astype(np.uint32)


@registry.register("audio", "audio_fingerprint", phase="post")
def check_audio_fingerprint(ctx: CheckContext) -> Finding:
    s = ctx.extra.get("samples")
    if s is None:
        from audio_detector.validator import AudioValidator

        s = AudioValidator().extract_pcm_samples(ctx.path)
        ctx.extra["samples"] = s
    x, sr, _dur = s
    base = dict(check_id="audio_fingerprint", dimension="Section21.3", stage="context", title="Reuse / context fingerprint",
                evidence_class=EvidenceClass.CONTEXT)
    fp = fingerprint(x, sr) if x is not None else np.zeros(0, np.uint32)
    if len(fp) < 20:
        return Finding(status=FindingStatus.NOT_APPLICABLE, severity=Severity.NONE,
                       detail="Recording too short to fingerprint.", data={"fingerprint_frames": len(fp)}, **base)
    index = load_index(context_index_path(), _prepare)
    matches = []
    for row in index:
        ber = best_ber(fp, row["_fp"])
        if ber <= MATCH_BER:
            matches.append({"label": row.get("label", ""), "source": row.get("source", ""), "ber": round(ber, 3)})
    data = {"fingerprint_frames": int(len(fp)), "index_loaded": bool(index), "matches": matches}
    if matches:
        m = matches[0]
        return Finding(status=FindingStatus.WARN, severity=Severity.MEDIUM, detail=(
            f"Acoustically matches a known reference ('{m['label']}', bit error rate {m['ber']}). The sound may be authentic while its "
            "claimed context is not; verify date, speaker and setting independently and confirm the match by ear."), data=data, **base)
    if index:
        return Finding(status=FindingStatus.INFO, severity=Severity.NONE, detail="No match in the local reference index.", data=data, **base)
    return Finding(status=FindingStatus.INFO, severity=Severity.NONE, detail=(
        "Fingerprint computed. No local reference index is configured, so re-use cannot be checked here (set OMNI_AUDIO_FP_INDEX)."),
        data=data, **base)


def context_index_path() -> Path:
    return audio_fp_index_file(DEFAULT_INDEX)
