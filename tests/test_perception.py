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


def test_shipped_vocabulary_embeddings_match_the_current_vocabularies():
    """Fails until ``python -m services.build_vocab_embeddings`` is run after any change to a vocabulary or the pinned model."""
    from core.perception.recognizer import EMBEDDINGS_FILE, vocabulary_key

    with np.load(EMBEDDINGS_FILE, allow_pickle=False) as data:
        assert str(data["key"]) == vocabulary_key()
        for name, items in VOCABS.items():
            assert data[name].shape[0] == len(items)
            assert np.allclose(np.linalg.norm(data[name], axis=1), 1.0, atol=1e-4)


@needs_recognizer
def test_shipped_embeddings_equal_a_fresh_computation_and_the_text_tower_is_released():
    import torch

    from core.perception.recognizer import EMBEDDINGS_FILE

    rec = get_recognizer()
    assert rec._model.text_model is None                      # about 1 GB freed after the vocabularies were embedded
    fresh = ZeroShotRecognizer()
    fresh._embed_vocabularies_and_free_text_tower = lambda *a, **k: None
    assert fresh._ensure() is not None
    with np.load(EMBEDDINGS_FILE, allow_pickle=False) as data:
        for name in VOCABS:
            computed = fresh._text_features(name).numpy()
            assert np.allclose(data[name], computed, atol=1e-5), name
    img = np.full((224, 224, 3), (200, 160, 80), np.uint8)
    assert rec.classify([img], "scene", top_k=1)[0]                 # image classification still works without the tower


def test_face_finder_survives_an_extremely_wide_strip():
    """A 64-megapixel strip scales to a zero-pixel side; it must be handled, not crash cv2.resize."""
    from core.face_detection import FaceFinder

    strip = np.zeros((20, 40000, 3), np.uint8)
    assert FaceFinder().find(strip) == []
    assert FaceFinder().find(np.zeros((40000, 20, 3), np.uint8)) == []


def test_screening_face_finder_survives_extreme_strips_and_keeps_the_short_side():
    from core.perception.face_scan import ScreeningFaceFinder

    finder = ScreeningFaceFinder()
    assert finder.find(np.zeros((20, 40000, 3), np.uint8)) == []
    assert finder.find(np.zeros((40000, 20, 3), np.uint8)) == []
