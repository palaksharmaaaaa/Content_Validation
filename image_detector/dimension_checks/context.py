"""
image_detector.dimension_checks.context: perceptual-hash fingerprinting for "real but misleading"
re-use detection. The file's own pixels cannot reveal a false caption; this check
only computes hashes and, if the operator maintains a local reference index, matches against it.

Index format (JSON lines): {"phash": "<16 hex>", "label": "...", "source": "..."}.
Set OMNI_CONTEXT_HASH_INDEX or place the file at image_detector/data/context_hash_index.jsonl.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PIL import Image

from core.forensics.config import context_index_file
from core.forensics.jsonl_index import load_index
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from image_detector.dimension_checks import _common as C

DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "data" / "context_hash_index.jsonl"
MATCH_HAMMING = 6


def _prepare(row: Dict[str, Any]) -> None:
    int(row["phash"], 16)


@registry.register("image", "perceptual_hash", phase="post")
def check_perceptual_hash(ctx: CheckContext) -> Finding:
    with Image.open(ctx.path) as im:
        im.load()
        ph, dh = C.phash64(im), C.dhash64(im)
    index = load_index(context_index_file(DEFAULT_INDEX), _prepare)
    matches = []
    for row in index:
        dist = C.hamming64(ph, int(row["phash"], 16))
        if dist <= MATCH_HAMMING:
            matches.append({"label": row.get("label", ""), "source": row.get("source", ""), "hamming": dist})
    data = {"phash": C.to_hex64(ph), "dhash": C.to_hex64(dh), "index_loaded": bool(index), "matches": matches}
    base = dict(check_id="perceptual_hash", dimension="Section22.3", stage="context", title="Reuse / context fingerprint",
                severity=Severity.MEDIUM, evidence_class=EvidenceClass.CONTEXT, data=data)
    if matches:
        m = matches[0]
        return Finding(status=FindingStatus.WARN, detail=(
            f"Perceptually matches a known reference ('{m['label']}', Hamming {m['hamming']}). The pixels may be authentic "
            "while the claimed context is not; verify date, place and caption independently."), **base)
    base["severity"] = Severity.NONE
    if index:
        return Finding(status=FindingStatus.INFO, detail="No match in the local reference index.", **base)
    return Finding(status=FindingStatus.INFO, detail=(
        "Hashes computed. No local reference index is configured, so context re-use cannot be checked here "
        "(set OMNI_CONTEXT_HASH_INDEX)."), **base)
