"""
services.blind_test: run the full image pipeline over folders of photos without ever looking at labels, then score it.

    python -m services.blind_test run     --out reports/blind.jsonl  "<photo dir 1>" "<photo dir 2>"
    python -m services.blind_test summary --out reports/blind.jsonl [--truth "Human Faces Dataset/AI-Generated Images=ai"]

``run`` is blind: it records only what the engine says (path, verdict AI / REAL / UNSURE, scores, faces, scene) and is
resumable. Nothing is copied; only file paths and numbers are written. ``summary`` compares the verdicts with the labels you
supply afterwards (``--truth <path fragment>=ai|real``) and reports accuracy, how often the engine abstained, and the
worst mistakes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".jfif"}
_AI_STATES = {"FULLY_AI_GENERATED", "AI_ENHANCED_COMPOSITE", "PROCEDURAL_CGI_SYNTHETIC",
              "AI_GENERATED_SCREENSHOT", "AI_ENHANCED_SCREENSHOT"}
_REAL_STATES = {"AUTHENTIC_REAL_PHOTOGRAPH", "AUTHENTIC_EDITED", "AUTHENTIC_RECAPTURED_SCREEN", "AUTHENTIC_SCREENSHOT"}


def verdict_of(taxonomy_state: Optional[str]) -> str:
    """Collapse the taxonomy into the three answers a person wants: AI, REAL or UNSURE."""
    if taxonomy_state in _AI_STATES:
        return "AI"
    if taxonomy_state in _REAL_STATES:
        return "REAL"
    return "UNSURE"


def iter_images(folders: Iterable[Path]) -> Iterator[Path]:
    """Every image file below the folders, in a stable order."""
    for folder in folders:
        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                if Path(name).suffix.lower() in _EXT:
                    yield Path(root) / name


def _record(path: Path, item: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    decision = item.get("decision") or {}
    probs = decision.get("authenticity_probabilities") or {}
    content = item.get("content_res") or {}
    humans = (content.get("entities") or {}).get("humans", {})
    face_finding = next((f for f in ((item.get("dimension_report") or {}).get("findings_by_stage") or {}).get("faces", [])), {})
    return {
        "path": str(path),
        "verdict": verdict_of(decision.get("taxonomy_state")),
        "taxonomy": decision.get("taxonomy_state"),
        "p_ai": probs.get("p_ai"), "p_real": probs.get("p_real"),
        "band": (item.get("confidence_band") or {}).get("band"),
        "faces": humans.get("faces_count", 0), "persons": humans.get("persons_count", 0),
        "face_ai_like": bool((face_finding.get("data") or {}).get("ai_like")),
        "face_worst_p_ai": (face_finding.get("data") or {}).get("worst_p_ai"),
        "minor_review": bool((content.get("minors") or {}).get("review_required")),
        "possible_minor": bool((content.get("minors") or {}).get("contains_possible_minor")),
        "youngest_age": (content.get("minors") or {}).get("youngest_age"),
        "age_status": (content.get("minors") or {}).get("status"),
        "scene": (content.get("environment") or {}).get("setting"),
        "seconds": round(seconds, 2),
    }


def run(folders: List[Path], out: Path, limit: int = 0) -> int:
    """Analyse every image not already in ``out``; append one JSON line per image. Returns the number analysed."""
    from services.forensic_service import ForensicService
    from ui.adapters import process_single_image

    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        done = {json.loads(line)["path"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    svc = ForensicService.get_instance()
    todo = [p for p in iter_images(folders) if str(p) not in done]
    if limit:
        todo = todo[:limit]
    started, count = time.time(), 0
    with out.open("a", encoding="utf-8") as fh:
        for path in todo:
            t0 = time.time()
            try:
                item = process_single_image(str(path), path.name, svc.image_detector, svc.content_analyzer, svc.attribution_engine)
                rec = _record(path, item, time.time() - t0) if item.get("success") else {"path": str(path), "verdict": "ERROR", "error": str(item.get("error"))[:200]}
            except Exception as exc:  # one bad file must not stop a multi-hour run
                rec = {"path": str(path), "verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}"[:200]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            count += 1
            if count % 100 == 0:
                rate = (time.time() - started) / count
                print(f"{count}/{len(todo)}  {rate:.2f}s/img  eta {(len(todo) - count) * rate / 60:.0f} min", flush=True)
    return count


def summary(out: Path, truth: Dict[str, str], split: str = "all") -> Dict[str, Any]:
    """Score recorded verdicts against ``truth`` ({path fragment: 'ai'|'real'}); unlabelled groups get a verdict histogram."""
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    if split != "all":   # "val": only files the face model never trained on (same content-hash split as face_training)
        from core.hashing import file_sha256
        from core.media_library import partition

        rows = [r for r in rows if r["verdict"] == "ERROR" or partition(file_sha256(r["path"], cached=False)) == split]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        label = next((v for k, v in truth.items() if k.replace("\\", "/") in r["path"].replace("\\", "/")), None)
        groups[label or "unlabelled"].append(r)
    report: Dict[str, Any] = {"total": len(rows), "errors": sum(r["verdict"] == "ERROR" for r in rows), "groups": {}}
    for label, rs in groups.items():
        hist = Counter(r["verdict"] for r in rs)
        entry: Dict[str, Any] = {"n": len(rs), "verdicts": dict(hist)}
        if label in ("ai", "real"):
            want = label.upper()
            decided = [r for r in rs if r["verdict"] in ("AI", "REAL")]
            entry["correct"] = sum(r["verdict"] == want for r in rs)
            entry["wrong"] = sum(r["verdict"] in ("AI", "REAL") and r["verdict"] != want for r in rs)
            entry["unsure"] = hist.get("UNSURE", 0)
            entry["accuracy_all"] = round(entry["correct"] / len(rs), 4)
            entry["accuracy_when_decided"] = round(sum(r["verdict"] == want for r in decided) / len(decided), 4) if decided else None
        report["groups"][label] = entry
    return report


def main(argv=None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("folders", nargs="+", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("summary")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--split", choices=("all", "val", "train"), default="all", help="restrict to the hash-based validation/training split")
    s.add_argument("--truth", action="append", default=[], help="'path fragment=ai' or 'path fragment=real'")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(f"analysed {run(a.folders, a.out, a.limit)} image(s)")
        return 0
    truth = {t.rsplit("=", 1)[0]: t.rsplit("=", 1)[1].lower() for t in a.truth}
    print(json.dumps(summary(a.out, truth, a.split), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
