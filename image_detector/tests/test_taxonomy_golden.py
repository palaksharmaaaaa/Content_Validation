"""Seeded 4000-case fuzz golden for the taxonomy decision tree (covers all 9 reachable states)."""
import random

from image_detector.scoring import evaluate_taxonomy_classification as ev
from tests.golden_support import assert_golden


def rnd_case(r):
    def b(p=0.25):
        return r.random() < p

    meta = {}
    if b(.4): meta["camera_make"] = r.choice(["Canon", "Apple"]); meta["camera_model"] = r.choice(["R5", "iPhone"])
    if b(.15): meta["ai_signature_found"] = True; meta["signature_details"] = "sig"
    if b(.15): meta["ai_enhancer_signature_found"] = True
    if b(.1): meta["iptc_digital_source_type"] = r.choice(["trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia"])
    if b(.05): meta["photoshop_credit"] = "Made with Google AI"
    if b(.15): meta["graphic_editor_signature_found"] = True
    if b(.2): meta.update(has_optical_parameters=True, focal_length=50, f_number=2.8, iso=100)
    kw = dict(
        ai_pct=round(r.uniform(0, 100), 1), real_pct=round(r.uniform(0, 100), 1), metadata=meta,
        watermark_data={"watermark_detected": b(.12), "details": "wm"},
        cutout_data={"is_cutout": b(.2), "details": "cut"},
        scanned_data={"is_scanned": b(.12), "details": "scan"},
        art_data={"is_digital_art": b(.2), "visual_medium": r.choice(["Digital 3D CGI / AI Neural Painting", "Other"]), "details": "art"},
        screenshot_data={"is_screenshot": b(.15), "device_type": "Phone", "orientation": "Portrait", "screen_resolution": "1x1", "details": "ss"},
        inpainting_data={"is_manipulated": b(.12), "details": "inp"},
        noise_mean=round(r.uniform(0.5, 3.0), 2), smoothness=round(r.uniform(0.5, 4.0), 2),
        is_square_gen=b(.15), is_canonical_gen=b(.15),
    )
    if b(.5): kw["metadata_absent"] = b(.6); kw["synthetic_signal_count"] = r.randint(0, 4)
    if b(.2): kw["screen_recapture_data"] = {"is_screen_recapture": True, "details": "rc"}
    if b(.1): kw["screen_recapture_detected"] = True
    if b(.3): kw["text_regions_count"] = r.randint(0, 6)
    if b(.1): kw["prob_ai"] = r.random()
    return kw




def test_taxonomy_matches_golden_fuzz():
    r = random.Random(11)
    results = []
    for _ in range(4000):
        s, label, desc, reasons = ev(**rnd_case(r))
        results.append([str(s), label, desc, reasons])
    assert len({x[0] for x in results}) == 9  # every reachable taxonomy state is exercised
    assert_golden("taxonomy_fuzz_golden", results)
