"""Seeded 1200-case golden for video generator attribution."""
import json
import random

from video_detector.attribution import VideoModelAttributionEngine
from video_detector.provenance import KNOWN_VIDEO_GENERATOR_SIGNATURES



from tests.golden_support import assert_golden


def test_video_attribution_matches_golden(tmp_path):
    tmp = tmp_path

    r = random.Random(8)
    f = tmp / "v.mp4"
    f.write_bytes(b"x")
    eng = VideoModelAttributionEngine()
    sigs = list(KNOWN_VIDEO_GENERATOR_SIGNATURES) + ["unknown vendor", "Tongyi", "KLING"]
    res = []
    for _ in range(1200):
        pd = None
        if r.random() < .85:
            pd = {"vendor_signatures_found": r.sample(sigs, r.randint(0, 3)), "c2pa_present": r.random() < .3}
        td = None
        if r.random() < .85:
            td = {"temporal_consistency": {"motion_variance": r.choice([0, 3, 10, 29, 60, 141, 300, r.uniform(0, 400)])},
                  "diffusion_flicker": {"has_diffusion_flicker": r.random() < .4}}
        res.append(json.loads(json.dumps(eng.attribute_video(f, temporal_data=td, provenance_data=pd), sort_keys=True)))
    res.append(eng.attribute_video(tmp / "nope.mp4"))
    assert_golden("video_attribution_golden", json.loads(json.dumps(res, sort_keys=True)))
