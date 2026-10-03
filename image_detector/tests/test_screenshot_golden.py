"""Seeded 300-case golden for screenshot detection (canonical sizes, UI bars, taskbars, noise, metadata)."""
import json
from pathlib import Path
import random

import cv2
import numpy as np

from image_detector.config import CANONICAL_SCREEN_RESOLUTIONS
from image_detector.features import detect_screenshot


from tests.golden_support import assert_golden


def make(r, w, h, kind):
    if kind == "noise":
        return r.integers(0, 255, (h, w, 3), dtype=np.uint8)
    img = np.full((h, w, 3), 245, np.uint8)
    if kind == "ui":
        cv2.rectangle(img, (0, 0), (w, int(h * .05)), (30, 30, 30), -1)
        for i in range(8):
            cv2.line(img, (int(w * .1), int(h * (.1 + i * .1))), (int(w * .9), int(h * (.1 + i * .1))), (20, 20, 20), 2)
        cv2.rectangle(img, (int(w * .35), h - 12), (int(w * .65), h - 8), (10, 10, 10), -1)
    if kind == "taskbar":
        cv2.rectangle(img, (0, h - int(h * .06)), (w, h), (20, 20, 20), -1)
        cv2.rectangle(img, (w // 10, h // 10), (w // 2, h // 2), (200, 120, 60), 2)
    if kind == "smooth":
        img = cv2.GaussianBlur(r.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 20)
    return img



def test_screenshot_detection_matches_golden():

    r = np.random.default_rng(9)
    rr = random.Random(4)
    sizes = list(CANONICAL_SCREEN_RESOLUTIONS)[:12] + [(400, 400), (500, 900), (900, 500), (800, 600), (1920, 1080), (300, 700), (700, 300)]
    names = ["a.png", "Screenshot_2024.png", "snip.png", "capture_1.png", "IMG_1.jpg", "screen shot.png"]
    metas = [None, {}, {"screenshot_software_found": True, "screenshot_software_name": "Snagit"},
             {"camera_make": "Canon", "camera_model": "R5"}, {"has_optical_parameters": True}]
    res = []
    for _ in range(300):
        w, h = rr.choice(sizes)
        w, h = max(64, min(w, 1200)), max(64, min(h, 1200))
        kind = rr.choice(["noise", "ui", "taskbar", "smooth", "flat"])
        img = make(r, w, h, kind)
        meta = rr.choice(metas)
        res.append(detect_screenshot(rr.choice(names), img, meta))
    res.append(detect_screenshot("nope.png", None))
    assert_golden("image_screenshot_golden", json.loads(json.dumps(res, sort_keys=True)))
