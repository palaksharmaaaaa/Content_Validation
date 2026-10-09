"""Seeded 1500-case golden for image generator attribution."""
import json
import random

from PIL import Image

from image_detector.attribution import KNOWN_IMAGE_GENERATORS, ImageModelAttributionEngine

SOFT = ["", "midjourney v6", "DALL-E 3", "automatic1111", "comfyui", "ideogram", "recraft", "magnific", "firefly", "leonardo",
        "grok", "xai", "nano banana", "gemini 2.5 flash image", "seedream", "hunyuan", "qwen", "kolors", "gpt-image", "gpt image",
        "remini", "topaz photo ai", "canva", "photoshop"]
STATES = [None, "AUTHENTIC_REAL_PHOTOGRAPH", "AUTHENTIC_RECAPTURED_SCREEN", "AUTHENTIC_SCREENSHOT", "AUTHENTIC_EDITED", "FULLY_AI_GENERATED"]




from tests.golden_support import assert_golden


def _run(tmp):

    r = random.Random(5)
    sizes = [(64, 64)] + [s for g in KNOWN_IMAGE_GENERATORS.values() for s in g["resolutions"][:2]][:30]
    names = ["a.png", "gemini_x.png", "face-swap.png", "faceswap.png"]
    files = []
    for i, (w, h) in enumerate(sizes[:20]):
        p = tmp / f"{r.choice(names)[:-4]}_{i}.png"
        Image.new("RGB", (w, h)).save(p)
        files.append(p)
    eng = ImageModelAttributionEngine()
    res = []
    for _ in range(1500):
        p = r.choice(files)
        fd = None
        if r.random() < .8:
            fd = {"taxonomy_state": r.choice(STATES), "watermark_detected": r.random() < .15, "watermark_details": "wm",
                  "face_swap_detected": r.random() < .15, "face_swap_details": "fs",
                  "spectral_features": {"spectral_decay_slope": r.uniform(1.0, 3.0)},
                  "surface_smoothness": r.uniform(0.5, 4), "digital_art_detected": r.random() < .2,
                  "forensic_metrics": {"is_digital_art": r.random() < .2}}
        pd = None
        if r.random() < .8:
            pd = {"metadata": {"software": r.choice(SOFT), "creator_tool": r.choice(["", "Canva", "Topaz Labs"]),
                               "photoshop_credit": r.choice(["", "Made with Google AI"]),
                               "iptc_digital_source_type": r.choice(["", "trainedAlgorithmicMedia"])},
                  "c2pa_present": r.random() < .3}
        a = eng.attribute_image(p, forensic_data=fd, provenance_data=pd)
        res.append(json.loads(json.dumps(a, sort_keys=True)))
    res.append(eng.attribute_image(tmp / "missing.png"))
    return res


def test_attribution_matches_golden(tmp_path):
    assert_golden("image_attribution_golden", _run(tmp_path))
