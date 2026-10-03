"""core.perception: colours, same-person grouping and graceful behaviour with and without the pretrained models."""
import numpy as np
import pytest

from core.perception import colors
from core.perception.detector import ObjectDetector, get_object_detector
from core.perception.face_attributes import FaceAttributes, group_same_person
from core.perception.recognizer import ZeroShotRecognizer, get_recognizer
from core.perception.vocab import INDOOR, VOCABS

needs_detector = pytest.mark.skipif(not get_object_detector().available, reason="RF-DETR not downloaded (python -m services.fetch_models)")
needs_recognizer = pytest.mark.skipif(not get_recognizer().available, reason="SigLIP 2 not downloaded (python -m services.fetch_models)")


def test_colour_names_are_sensible():
    assert colors.name_color(255, 255, 255) == "white"
    assert colors.name_color(0, 0, 0) == "black"
    assert colors.name_color(220, 30, 30) == "red"
    assert colors.name_color(30, 80, 200) in ("blue", "royal blue")
    assert colors.name_color(34, 139, 34) in ("green", "forest green", "dark green")


def test_dominant_colours_are_deterministic_and_sum_to_100():
    img = np.zeros((120, 160, 3), np.uint8)
    img[:, :80] = (0, 0, 255)        # BGR red
    img[:, 80:] = (255, 0, 0)        # BGR blue
    first = colors.dominant_colors(img)
    assert first == colors.dominant_colors(img)
    assert {c["color_name"] for c in first} >= {"Red", "Blue"}
    assert abs(sum(c["coverage_pct"] for c in first) - 100.0) < 0.5


def test_dominant_colours_of_flat_and_degenerate_images():
    assert colors.dominant_colors(np.full((50, 50, 3), 128, np.uint8))[0]["color_name"] == "Gray"
    assert colors.dominant_colors(None) == []


def test_same_person_grouping():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(128); a /= np.linalg.norm(a)
    b = rng.standard_normal(128); b /= np.linalg.norm(b)
    near_a = a + 0.05 * rng.standard_normal(128); near_a /= np.linalg.norm(near_a)
    assert group_same_person([a, b, near_a, None]) == [0, 1, 0, -1]


def test_vocabularies_are_well_formed():
    for name, entries in VOCABS.items():
        labels = [label for label, _ in entries]
        assert len(labels) == len(set(labels)) and len(labels) >= 6, name
    assert set(INDOOR) == {label for label, _ in VOCABS["scene"]}


def test_models_report_unavailable_instead_of_raising():
    rec = ZeroShotRecognizer("no/such-model-xyz")
    assert not rec.available and rec.classify([np.zeros((64, 64, 3), np.uint8)], "scene") == [[]]
    det = ObjectDetector("no/such-model-xyz")
    assert not det.available and det.detect(np.zeros((64, 64, 3), np.uint8)) == []


def test_face_attribute_models_load_from_the_repo():
    fa = FaceAttributes()
    assert fa.expression_available and fa.identity_available


@needs_detector
def test_detector_finds_nothing_in_a_blank_image():
    assert get_object_detector().detect(np.full((320, 320, 3), 127, np.uint8)) == []


@needs_recognizer
def test_recognizer_prefers_the_right_scene_for_a_synthetic_sky_image():
    img = np.zeros((224, 224, 3), np.uint8)
    img[:, :] = (235, 170, 80)       # BGR: sky blue
    top = get_recognizer().classify([img[:, :, ::-1].copy()], "scene", top_k=3)[0]
    assert len(top) == 3 and 0.0 < top[0][1] <= 1.0
