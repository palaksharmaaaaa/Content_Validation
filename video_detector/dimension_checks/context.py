"""
video_detector.dimension_checks.context: frame-hash fingerprint for "real but misleading" re-use checks
(report Section 21.3). The footage itself cannot reveal a false caption; this check computes up to 16
evenly spaced perceptual (DCT) frame hashes and, if the operator maintains a local reference index,
matches by set overlap (>= 50% of query frames, at least 4, within Hamming distance 8).

Index format (JSON lines): {"phashes": ["<16 hex>", ...], "label": "...", "source": "..."}.
Build entries with ``video_fingerprint_hashes(path)``. Set OMNI_VIDEO_FP_INDEX or place the file at
video_detector/data/video_fingerprint_index.jsonl. Always verify matches by eye.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from core.forensics.config import video_fp_index_file
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from video_detector.dimension_checks import _common as C

DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "data" / "video_fingerprint_index.jsonl"
MATCH_HAMMING = 8
MIN_MATCHED = 4
MATCH_FRACTION = 0.5


def video_fingerprint_hashes(path: Union[str, Path], n: int = 16) -> List[str]:
    return [f"{C.phash64_gray(g):016x}" for g in C.read_spread_gray(Path(path), n=n)]


def _load_index(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            row["_h"] = [int(h, 16) for h in row["phashes"]]
            rows.append(row)
        except Exception:
            continue
    return rows


@registry.register("video", "video_fingerprint", phase="post")
def check_video_fingerprint(ctx: CheckContext) -> Finding:
    base = dict(check_id="video_fingerprint", dimension="Section21.3", stage="context", title="Reuse / context fingerprint",
                evidence_class=EvidenceClass.CONTEXT)
    hashes = video_fingerprint_hashes(ctx.path)
    if len(hashes) < 4:
        return Finding(status=FindingStatus.NOT_APPLICABLE, severity=Severity.NONE, detail="Too few decodable frames to fingerprint.",
                       data={"frames": len(hashes)}, **base)
    query = [int(h, 16) for h in hashes]
    index = _load_index(video_fp_index_file(DEFAULT_INDEX))
    matches = []
    for row in index:
        hit = sum(1 for q in query if any(C.hamming64(q, r) <= MATCH_HAMMING for r in row["_h"]))
        if hit >= MIN_MATCHED and hit / len(query) >= MATCH_FRACTION:
            matches.append({"label": row.get("label", ""), "source": row.get("source", ""), "frames_matched": hit})
    data = {"frames": len(hashes), "index_loaded": bool(index), "matches": matches, "phashes": hashes}
    if matches:
        m = matches[0]
        return Finding(status=FindingStatus.WARN, severity=Severity.MEDIUM, detail=(
            f"{m['frames_matched']} of {len(hashes)} sampled frames match a known reference ('{m['label']}'). The footage may be authentic "
            "while its claimed date, place or caption is not; verify context independently and confirm the match by eye."), data=data, **base)
    if index:
        return Finding(status=FindingStatus.INFO, severity=Severity.NONE, detail="No match in the local reference index.", data=data, **base)
    return Finding(status=FindingStatus.INFO, severity=Severity.NONE, detail=(
        "Frame hashes computed. No local reference index is configured, so re-use cannot be checked here (set OMNI_VIDEO_FP_INDEX)."),
        data=data, **base)
