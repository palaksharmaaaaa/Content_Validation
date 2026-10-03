"""
services.calibration_cli: measure real accuracy/calibration of a modality's detector on held-out labeled media.

    python -m services.calibration_cli --modality image [--split val|train|all] [--limit N] [--out report.json]

Uses the copy-free media library (<modality>_detector/data/library.json). The default split is the
validation partition, which retraining never trains on, so the numbers are not inflated by training data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from core.calibration_report import evaluate
from core.media_library import library_for


def _scorer(modality: str) -> Callable[[Path], float]:
    from services.forensic_service import ForensicService

    svc = ForensicService.get_instance()
    if modality == "image":
        return lambda p: float(svc.image_detector.predict(str(p))["ai_percentage"])
    if modality == "audio":
        return lambda p: float(svc.audio_detector.analyze_audio_file(str(p))["ai_percentage"])
    return lambda p: float(svc.video_detector.analyze_video(str(p))["ai_percentage"])


def collect_pairs(modality: str, split: str, limit: int, scorer: Callable[[Path], float]) -> List[Tuple[str, float]]:
    lib = library_for(modality)
    samples = lib.samples(split=None if split == "all" else split)
    pairs: List[Tuple[str, float]] = []
    for _sha, path, label in samples[: limit or None]:
        try:
            pairs.append((label, scorer(path)))
        except Exception as exc:  # one unreadable file must not abort the run
            print(f"skip {path}: {exc}", file=sys.stderr)
    return pairs


def main(argv=None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--modality", choices=("image", "audio", "video"), required=True)
    ap.add_argument("--split", choices=("val", "train", "all"), default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    report = evaluate(collect_pairs(a.modality, a.split, a.limit, _scorer(a.modality)))
    text = json.dumps(report, indent=2)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text)
    return report


if __name__ == "__main__":
    main()
