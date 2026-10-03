"""Golden for ImageContentAnalyzer on synthetic scenes (noise, silhouette, text, large, sky/ground)."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from image_detector.content import ImageContentAnalyzer


def imgs():
    rng = np.random.default_rng(2)
    noise = rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)
    blank_bg = np.full((500, 400, 3), 220, np.uint8)
    cv2.ellipse(blank_bg, (200, 260), (60, 180), 0, 0, 360, (40, 30, 90), -1)
    text = np.full((300, 800, 3), 255, np.uint8)
    for i in range(6):
        cv2.putText(text, "HELLO WORLD %d" % i, (20, 40 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    big = rng.integers(0, 255, (1500, 2000, 3), dtype=np.uint8)
    sky = np.zeros((300, 400, 3), np.uint8); sky[:150] = (235, 190, 120); sky[150:] = (60, 150, 60)
    return {"noise": noise, "character": blank_bg, "text": text, "big": big, "sky": sky}



def test_content_analysis_matches_golden(tmp_path):
    expected = json.loads((Path(__file__).resolve().parents[2] / "tests" / "data" / "image_content_golden.json").read_text(encoding="utf-8"))
    tmp = tmp_path

    an = ImageContentAnalyzer()
    res = {}
    for k, a in imgs().items():
        p = tmp / f"{k}.png"
        cv2.imwrite(str(p), a)
        res[k] = an.analyze_image_content(p)
        res[k + "_arr"] = an.analyze_image_content(a)
    res["missing"] = an.analyze_image_content(tmp / "none.png")
    res["badtype"] = an.analyze_image_content(123)
    assert json.loads(json.dumps(res, sort_keys=True, default=str)) == expected
