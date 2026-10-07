"""services.age_eval: measure the minor screening on faces with known ages (UTKFace layout: ``<age>_<gender>_<race>_<date>.jpg``).

    python -m services.age_eval run     --faces <folder of UTKFace images> --out reports/age_eval.jsonl [--per-minor-band 300 --per-adult-band 150]
    python -m services.age_eval summary --out reports/age_eval.jsonl
    python -m services.age_eval wild    --images <folder> --annotations <lagenda_annotation.csv> --out reports/age_wild.jsonl
    python -m services.age_eval wild-summary --out reports/age_wild.jsonl
    python -m services.age_eval fairface --parquet <FairFace validation .parquet> --out reports/age_fairface.jsonl
    python -m services.age_eval fairface-summary --out reports/age_fairface.jsonl

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

import numpy as np

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


def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _matches(subject: dict, face: list, person: list) -> bool:
    """A reported subject is the annotated person if its face centre lies in the (slightly grown) annotated face, or its person
    box overlaps the annotated person box well."""
    fb, pb = subject.get("face_box"), subject.get("person_box")
    if fb and face:
        cx, cy = fb[0] + fb[2] / 2, fb[1] + fb[3] / 2
        gw, gh = (face[2] - face[0]) * 0.3, (face[3] - face[1]) * 0.3
        if face[0] - gw <= cx <= face[2] + gw and face[1] - gh <= cy <= face[3] + gh:
            return True
    if pb and person and _iou([pb[0], pb[1], pb[0] + pb[2], pb[1] + pb[3]], person) >= 0.5:
        return True
    return False


def wild(images: Path, annotations: Path, out: Path) -> int:
    """End-to-end check on LAGENDA photographs (real face and person detection, nothing pre-cropped): for every annotated
    person with a known age, was a matching subject found, and was it sent to review?"""
    import csv

    from core.imageio import imread
    from core.perception.age import get_age_estimator

    gt: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in csv.DictReader(annotations.open(encoding="utf-8")):
        name = Path(r["img_name"]).name
        if r["age"] != "-1" and r["face_x0"] != "-1" and (images / name).exists():
            gt[name].append({"age": int(r["age"]), "face": [int(r[k]) for k in ("face_x0", "face_y0", "face_x1", "face_y1")],
                             "person": [int(r[k]) for k in ("person_x0", "person_y0", "person_x1", "person_y1")] if r["person_x0"] != "-1" else None})
    est = get_age_estimator()
    out.parent.mkdir(parents=True, exist_ok=True)
    started, n = time.time(), 0
    with out.open("w", encoding="utf-8") as fh:
        for i, (name, people) in enumerate(sorted(gt.items()), 1):
            result = est.assess(imread(str(images / name)))
            for p in people:
                hits = [s for s in result["subjects"] if _matches(s, p["face"], p["person"])]
                best = hits[0] if hits else None
                fh.write(json.dumps({"image": name, "true": p["age"], "face_w": p["face"][2] - p["face"][0],
                                     "found": best is not None, "est": best and best["age"], "assessment": best and best["assessment"],
                                     "flagged": bool(best and best["assessment"] != "ADULT")}) + "\n")
                n += 1
            if i % 10 == 0:
                fh.flush()
                print(f"{i}/{len(gt)} images  {(time.time() - started) / i:.2f}s/img", flush=True)
    return n


def wild_summary(out: Path) -> str:
    """Per face size: how many annotated minors were found, and how many were sent to review; plus the adult review rate."""
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    size_bins = [(0, 24), (24, 40), (40, 80), (80, 10000)]
    lines = ["annotated minors (age < 18) by face width    n   found  sent to review  not found  found but called adult"]
    for lo, hi in size_bins:
        rs = [r for r in rows if r["true"] < 18 and lo <= r["face_w"] < hi]
        if rs:
            found = sum(r["found"] for r in rs)
            flagged = sum(r["flagged"] for r in rs)
            lines.append(f"  {lo:>3}-{hi if hi < 10000 else 'up':<6}px {len(rs):26}{100 * found / len(rs):7.1f}%{100 * flagged / len(rs):14.1f}%{len(rs) - found:11}{found - flagged:12}")
    minors = [r for r in rows if r["true"] < 18]
    if minors:
        lines.append(f"  all minors: {len(minors)}, sent to review {100 * sum(r['flagged'] for r in minors) / len(minors):.1f}%")
    adults = [r for r in rows if r["true"] >= 18 and r["found"]]
    if adults:
        lines.append(f"adults found: {len(adults)}, sent to review {100 * sum(r['flagged'] for r in adults) / len(adults):.1f}%")
    return "\n".join(lines)


FAIRFACE_GROUPS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]


def fairface(parquet: Path, out: Path, per_young_group: int = 400, per_other_group: int = 120, seed: int = 4) -> int:
    """Age a stratified sample of FairFace (CC BY 4.0) faces. FairFace only gives 10-year groups, so 0-2 and 3-9 are certain
    minors, 10-19 mixes minors with 18-19 year-olds, and 20+ are adults. MiVOLO v2 was not trained on FairFace."""
    import io

    import cv2
    import pandas as pd

    from core.perception.age import get_age_estimator

    df = pd.read_parquet(parquet)
    rng = random.Random(seed)
    rows = []
    for gid, name in enumerate(FAIRFACE_GROUPS):
        idx = list(df.index[df["age"] == gid])
        rows += rng.sample(idx, min(len(idx), per_young_group if gid < 3 else per_other_group))
    rng.shuffle(rows)
    est = get_age_estimator()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i, ix in enumerate(rows, 1):
            img = cv2.imdecode(np.frombuffer(df.loc[ix, "image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            subject = est.assess(img, faces=[(0, 0, w, h)], persons=[])["subjects"][0]
            fh.write(json.dumps({"group": FAIRFACE_GROUPS[int(df.loc[ix, "age"])], "est": subject["age"], "share": subject["minor_group_share"],
                                 "assessment": subject["assessment"]}) + "\n")
            if i % 200 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    return len(rows)


def fairface_summary(out: Path) -> str:
    """Share of each FairFace age group sent to review, with the median estimated age."""
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = ["group           n   median est. age   sent to review"]
    for g in FAIRFACE_GROUPS:
        rs = [r for r in rows if r["group"] == g]
        if rs:
            lines.append(f"{g:<14}{len(rs):5}{statistics.median(r['est'] for r in rs):14.1f}{100 * sum(r['assessment'] != 'ADULT' for r in rs) / len(rs):15.1f}%")
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
    w = sub.add_parser("wild")
    w.add_argument("--images", type=Path, required=True)
    w.add_argument("--annotations", type=Path, required=True)
    w.add_argument("--out", type=Path, required=True)
    ws = sub.add_parser("wild-summary")
    ws.add_argument("--out", type=Path, required=True)
    ff = sub.add_parser("fairface")
    ff.add_argument("--parquet", type=Path, required=True)
    ff.add_argument("--out", type=Path, required=True)
    ffs = sub.add_parser("fairface-summary")
    ffs.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "fairface":
        print(f"analysed {fairface(a.parquet, a.out)} face(s)")
    elif a.cmd == "fairface-summary":
        print(fairface_summary(a.out))
    elif a.cmd == "wild":
        print(f"annotated people: {wild(a.images, a.annotations, a.out)}")
    elif a.cmd == "wild-summary":
        print(wild_summary(a.out))
    elif a.cmd == "run":
        print(f"analysed {run(a.faces, a.out, a.per_minor_band, a.per_adult_band)} image(s)")
    else:
        print(summary(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
