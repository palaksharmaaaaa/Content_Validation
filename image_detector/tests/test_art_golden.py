"""Seeded 160-case golden for digital-art / painting detection."""
import json
import random

import cv2
import numpy as np
from PIL import Image

from image_detector.features import detect_digital_art_and_painting


from tests.golden_support import assert_golden


def make(r, rng, kind):
    h, w = r.choice([(120, 160), (300, 200), (64, 64)])
    if kind == "photo":
        return np.clip(rng.normal(120, 25, (h, w, 3)), 0, 255).astype(np.uint8)
    if kind == "saturated":
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (0, 220, 255)
        img[h // 2:] = (255, 40, 40)
        return np.clip(img.astype(int) + rng.integers(0, 4, img.shape), 0, 255).astype(np.uint8)
    if kind == "ink":
        img = np.full((h, w, 3), 235, np.uint8)
        for i in range(0, h, 6):
            cv2.line(img, (0, i), (w, i + r.randint(-5, 5)), (10, 10, 10), 1)
        cv2.circle(img, (w // 2, h // 2), min(h, w) // 3, (0, 0, 0), 2)
        return img
    if kind == "flat":
        return np.full((h, w, 3), r.randint(0, 255), np.uint8)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)



def test_digital_art_detection_matches_golden(tmp_path):
    tmp = tmp_path

    r = random.Random(3)
    rng = np.random.default_rng(3)
    res = []
    for i in range(160):
        kind = r.choice(["photo", "saturated", "ink", "flat", "noise"])
        img = make(r, rng, kind)
        meta = r.choice([None, {}, {"camera_make": "Canon"}])
        if r.random() < .25:
            p = tmp / f"a{i}.png"
            rgba = np.dstack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.where(rng.random(img.shape[:2]) < .5, 255, 0).astype(np.uint8)])
            Image.fromarray(rgba).save(p)
            res.append(detect_digital_art_and_painting(p, None, meta))
        else:
            res.append(detect_digital_art_and_painting("x.png", img, meta))
    res.append(detect_digital_art_and_painting(tmp / "none.png"))
    assert_golden("image_art_golden", json.loads(json.dumps(res, sort_keys=True)))
