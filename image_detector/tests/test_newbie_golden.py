"""Seeded 1000-case golden for the plain-English image narrative (also pins deterministic entity ordering)."""
import json
import random
from pathlib import Path

from image_detector.explain import generate_newbie_explanation

VERDICTS = ["LIKELY REAL", "LIKELY AI-GENERATED", "AUTHENTIC (CONVENTIONALLY EDITED)", "AUTHENTIC SCREENSHOT", "AI-GENERATED SCREENSHOT",
            "AUTHENTIC (RECAPTURED SCREEN)", "UNDECIDED", "LIKELY_SYNTHETIC", "LIKELY_AUTHENTIC", "PARTIALLY_SYNTHETIC_OR_EDITED", "GRAPHIC", ""]


from tests.golden_support import assert_golden


def case(r):
    def put(d, k, v, p=.6):
        if r.random() < p:
            d[k] = v

    geom = {}
    put(geom, "width", r.randint(64, 4000)); put(geom, "height", r.randint(64, 4000)); put(geom, "aspect_ratio_str", "16:9")
    prof = {"spatial_geometry": geom}
    put(prof, "display_attributes", {"dpi_str": "300 x 300 DPI"})
    put(prof, "raw_physical_signals", {"flat_region_noise_mean": r.uniform(0, 3), "surface_smoothness_index": r.uniform(0, 5), "dark_line_art_pct": 1.5})
    put(prof, "pixel_color_profile", {"dominant_palette": [{"color_name": "Teal", "hex": "#008080", "coverage_pct": 41.2}]}, .5)
    put(prof, "width", 10, .3); put(prof, "height", 11, .3)
    inv = {}
    ents = {}
    put(ents, "humans", {"persons_count": r.choice([0, 1, 2, 5]), "faces_count": r.choice([0, 1, 3]), "is_stylized_character": r.random() < .2})
    put(ents, "animals", {"animal_types": r.choice([[], ["dog"], ["dog", "cat", "dog"]])})
    put(inv, "living_entities", ents, .8); put(inv, "persons_count", 2, .3); put(inv, "faces_count", 2, .3)
    put(inv, "vehicles", {"vehicle_types": r.choice([[], ["car"], ["bus", "car"]])}, .6)
    put(inv, "contents_and_items", {"identified_items": r.choice([[], ["cup"], ["cup", "laptop", "book", "pen"]])}, .6)
    put(inv, "purpose_and_depiction", {"primary_genre": "Street Photography"}, .5)
    put(inv, "tone_and_mood", {"atmospheric_mood": "Moody"}, .6)
    put(inv, "lighting_and_daytime", {"daytime": "Golden Hour"}, .6)
    ai = {}
    put(ai, "visual_medium", "Watercolor", .5); put(ai, "subject_genre", "Landscape", .3)
    put(ai, "forensic_metrics", {"noise_residual_mean": r.uniform(0, 3), "surface_smoothness": r.uniform(0, 5)}, .6)
    dec = {"final_status": r.choice(VERDICTS), "taxonomy_label": r.choice([None, "Fully AI Generated", "Authentic Real Capture"]),
           "authenticity_probabilities": {"p_ai": round(r.uniform(0, 100), 1), "p_real": round(r.uniform(0, 100), 1)}}
    return "img_%d.png" % r.randint(0, 99), prof, inv, ai, dec



def test_newbie_narrative_matches_golden():
    r = random.Random(77)
    actual = []
    for _ in range(1000):
        f, p, i, a, d = case(r)
        actual.append(generate_newbie_explanation(f, p, i, a, d))
    assert_golden("image_newbie_golden", actual)
