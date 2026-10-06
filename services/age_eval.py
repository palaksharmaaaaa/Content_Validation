"""services.age_eval: measure the minor screening on faces with known ages (UTKFace layout: ``<age>_<gender>_<race>_<date>.jpg``).

    python -m services.age_eval run     --faces <folder of UTKFace images> --out reports/age_eval.jsonl [--per-minor-band 300 --per-adult-band 150]
    python -m services.age_eval summary --out reports/age_eval.jsonl

``run`` ages a stratified sample, treating each pre-cropped image as one face (no body). ``summary`` prints the error per age
band and, for the rule in core.perception.age, the share of under-18s flagged (recall) and of adults flagged (cost).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List

BANDS = [(0, 4), (5, 9), (10, 12), (13, 15), (16, 17), (18, 20), (21, 25), (26, 35), (36, 60), (61, 200)]


def band_of(age: int) -> str:
    """Label of the age band an age falls in."""
    return next(f"{lo}-{hi}" for lo, hi in BANDS if lo <= age <= hi)


def run(faces: Path, out: Path, per_minor_band: int, per_adult_band: int, seed: int = 21) -> int:
    """Age a stratified sample of the images; one JSON line each. Returns the number analysed."""
    from core.imageio import imread
    from core.perception.age import get_age_estimator

    by: Dict[str, List[Path]] = collections.defaultdict(list)
    for f in sorted(faces.glob("*.jpg")):
        try:
            by[band_of(int(f.name.split("_")[0]))].append(f)
        except (ValueError, StopIteration):
            continue
    rng = random.Random(seed)
    chosen = [f for b, files in by.items() for f in rng.sample(files, min(len(files), per_minor_band if int(b.split("-")[1]) < 18 else per_adult_band))]
    rng.shuffle(chosen)
    est = get_age_estimator()
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with out.open("w", encoding="utf-8") as fh:
        for i, f in enumerate(chosen, 1):
            img = imread(str(f))
            h, w = img.shape[:2]
            subject = est.assess(img, faces=[(0, 0, w, h)], persons=[])["subjects"][0]
            fh.write(json.dumps({"file": f.name, "true": int(f.name.split("_")[0]), "est": subject["age"],
                                 "share": subject["minor_group_share"], "assessment": subject["assessment"]}) + "\n")
            if i % 200 == 0:
                print(f"{i}/{len(chosen)}  {(time.time() - started) / i:.2f}s/img", flush=True)
    return len(chosen)


def summary(out: Path) -> str:
    """Error per band, then recall on under-18s and false-flag rate on adults for the current rule."""
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    bands: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in rows:
        bands[band_of(r["true"])].append(r)
    lines = ["band        n  median |error|  mean signed error  flagged (review)"]
    for lo, hi in BANDS:
        rs = bands.get(f"{lo}-{hi}", [])
        if not rs:
            continue
        err = [r["est"] - r["true"] for r in rs]
        flagged = sum(r["assessment"] in ("LIKELY_MINOR", "POSSIBLE_MINOR") for r in rs)
        lines.append(f"{lo}-{hi:<6}{len(rs):5}{statistics.median(abs(e) for e in err):12.1f}{statistics.mean(err):18.1f}{100 * flagged / len(rs):15.1f}%")
    minors = [r for r in rows if r["true"] < 18]
    if minors:
        caught = sum(r["assessment"] in ("LIKELY_MINOR", "POSSIBLE_MINOR") for r in minors)
        lines.append(f"\nunder-18s flagged for review: {caught}/{len(minors)} = {100 * caught / len(minors):.1f}%")
    return "\n".join(lines)


def main(argv=None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--faces", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--per-minor-band", type=int, default=300)
    r.add_argument("--per-adult-band", type=int, default=150)
    s = sub.add_parser("summary")
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(f"analysed {run(a.faces, a.out, a.per_minor_band, a.per_adult_band)} image(s)")
    else:
        print(summary(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
