"""Seeded golden for image metadata extraction (EXIF, optical params, XMP/IPTC markers, software signatures)."""
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
from PIL.TiffImagePlugin import IFDRational

from image_detector.features import extract_image_metadata

XMPS = ["", "<x>trainedAlgorithmicMedia</x>", "<x>compositeWithTrainedAlgorithmicMedia</x>", "Made with Google AI", "Topaz Photo AI",
        "Canva", "Attrib:Ads", "midjourney", "plain text no signals", "trainedAlgorithmicMedia Canva"]
SOFT = ["", "Adobe Photoshop 25", "Topaz Photo AI 3", "Canva", "GIMP", "midjourney v6", "Snipping Tool", "ShareX", "Pixlr", "Flux export", "Lumia 950"]



def test_metadata_extraction_matches_golden(tmp_path):
    expected = json.loads((Path(__file__).resolve().parents[2] / "tests" / "data" / "image_metadata_golden.json").read_text(encoding="utf-8"))
    tmp = tmp_path

    r = random.Random(14)
    arr = np.zeros((40, 40, 3), np.uint8)
    res = {}
    for i in range(120):
        ex = Image.Exif()
        if r.random() < .8:
            if r.random() < .7:
                ex[271] = r.choice(["Canon", "Apple", "SONY"]); ex[272] = r.choice(["R5", "iPhone 15", "A7"])
            s = r.choice(SOFT)
            if s:
                ex[305] = s
            if r.random() < .5:
                ex[306] = "2024:05:01 10:00:00"
            sub = ex.get_ifd(0x8769)
            if r.random() < .6:
                sub[0x829D] = IFDRational(28, 10); sub[0x920A] = IFDRational(50, 1); sub[0x8827] = 100; sub[0x829A] = IFDRational(1, 250)
            if r.random() < .3:
                sub[0xA434] = "RF 50mm"
        xmp = r.choice(XMPS)
        p = tmp / f"m{i}.png"
        info = PngImagePlugin.PngInfo()
        if xmp:
            info.add_itxt("XML:com.adobe.xmp", xmp)
        if r.random() < .2:
            info.add_text("Creation Time", "2024-01-01")
        Image.fromarray(arr).save(p, "PNG", pnginfo=info, exif=ex)
        res[f"png{i}"] = extract_image_metadata(p)
    j = tmp / "x.jpg"
    Image.fromarray(arr).save(j, "JPEG")
    res["jpg_plain"] = extract_image_metadata(j)
    res["bytesio"] = extract_image_metadata(io.BytesIO(j.read_bytes()))
    res["ndarray"] = extract_image_metadata(arr)
    res["missing"] = extract_image_metadata(tmp / "none.png")
    assert json.loads(json.dumps(res, sort_keys=True, default=str)) == expected
