"""Seeded 300-scene golden for face localisation (person-anchored and standalone modes)."""
import json
from pathlib import Path
import random

import cv2
import numpy as np

from image_detector.face import FaceDeepfakeDetector


def scene(r, rng):
    h, w = r.choice([(240, 320), (480, 360), (300, 300), (1400, 1100), (20, 20), (12, 40)])
    img = rng.integers(60, 120, (h, w, 3), dtype=np.uint8) if r.random() < .5 else np.full((h, w, 3), r.randint(40, 200), np.uint8)
    boxes = []
    for _ in range(r.randint(0, 3)):
        cx, cy = r.randint(w // 5, 4 * w // 5), r.randint(h // 5, 4 * h // 5)
        rx, ry = max(3, int(w * r.uniform(.08, .25))), max(3, int(h * r.uniform(.1, .3)))
        col = (r.randint(90, 130), r.randint(130, 170), r.randint(170, 220))  # BGR skin-ish
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, col, -1)
        if r.random() < .6:
            img[cy - ry // 2: cy - ry // 4, cx - rx // 2: cx + rx // 2] = (30, 30, 30)
        boxes.append((max(0, cx - rx - 5), max(0, cy - ry - 5), min(w, 2 * rx + 10), min(h, 3 * ry + 20)))
    return img, boxes



def test_face_detection_matches_golden():
    expected = json.loads((Path(__file__).resolve().parents[2] / "tests" / "data" / "image_faces_golden.json").read_text(encoding="utf-8"))

    r = random.Random(31)
    rng = np.random.default_rng(31)
    det = FaceDeepfakeDetector()
    res = []
    for _ in range(300):
        img, boxes = scene(r, rng)
        mode = r.choice(["none", "boxes", "empty", "weird"])
        pb = None if mode == "none" else ([] if mode == "empty" else (boxes if mode == "boxes" else [(0, 0, 5, 5), (-10, -10, 40, 60), (1, 1, 10000, 10000)]))
        res.append(det.detect_faces(img, person_boxes=pb))
    res.append(det.detect_faces(None))
    res.append(det.detect_faces(np.zeros((3,), np.uint8)))
    assert json.loads(json.dumps(res, default=lambda o: list(o))) == expected
