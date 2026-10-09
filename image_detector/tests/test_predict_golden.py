"""Golden regression for the image scoring pipeline."""
import json


import cv2
import numpy as np
from PIL import Image

from image_detector import ImageAIDetector
from tests.golden_support import assert_golden, skip_unless_state_matches


def exif(make="Canon", model="Canon EOS R5"):
    ex = Image.Exif()
    ex[271] = make
    ex[272] = model
    return ex


def arrays():
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:512, 0:512]
    grad = np.stack([(xx / 2).astype(np.uint8), (yy / 2).astype(np.uint8), ((xx + yy) / 4).astype(np.uint8)], axis=-1)
    noise = rng.integers(60, 190, (512, 512, 3), dtype=np.uint8)
    grain = np.clip(rng.normal(128, 9, (512, 512, 3)), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(noise, (0, 0), 6)
    photo = np.clip(cv2.GaussianBlur(rng.normal(120, 40, (512, 512, 3)).astype(np.float32), (0, 0), 3) + rng.normal(0, 3, (512, 512, 3)), 0, 255).astype(np.uint8)
    flat = np.full((1024, 1024, 3), 200, np.uint8)
    flat[100:300, 100:400] = (30, 80, 160)
    return {"noise": noise, "grad": grad, "grain": grain, "blurred": blurred, "photo": photo, "flat1024": flat}


def clean(r):
    r = dict(r)
    r.pop("heatmap_rgb", None)
    return r




def _run(tmp):
    det = ImageAIDetector()
    det.load()
    res = {}
    for name, arr in arrays().items():
        for fmt in ("jpg", "png"):
            for with_exif in (False, True):
                p = tmp / f"{name}_{int(with_exif)}.{fmt}"
                kw = {"exif": exif()} if (with_exif and fmt == "jpg") else {}
                Image.fromarray(arr).save(p, "JPEG" if fmt == "jpg" else "PNG", **({"quality": 92} if fmt == "jpg" else {}), **kw)
                res[f"{name}|{fmt}|exif={with_exif}"] = clean(det.predict(str(p)))
        p = tmp / f"{name}_x.jpg"
        Image.fromarray(arr).save(p, "JPEG", quality=90)
        res[f"{name}|extra"] = clean(det.predict(str(p), sensitivity="high", provenance={"c2pa_present": True}, extra_log_lrs={"a": 0.25, "b": -0.1}))
        res[f"{name}|ndarray"] = clean(det.predict(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)))
        res[f"{name}|bytes"] = clean(det.predict(p.read_bytes(), sensitivity="aggressive"))
        res[f"{name}|pil"] = clean(det.predict(Image.fromarray(arr)))
    res["unreadable"] = clean(det.predict(b"not an image"))
    return json.loads(json.dumps(res, default=str))


def test_predict_matches_golden_baseline(tmp_path):
    """Golden regression for ImageAIDetector.predict (49 inputs). Regenerate deliberately if scoring is meant to change."""
    skip_unless_state_matches("image_predict_golden.meta.json", ["image_detector/models/ai_detector.pt", "image_detector/data/image_calibration.json"])
    assert_golden("image_predict_golden", _run(tmp_path))
