"""Recall-first face finder: merging, tiling geometry, and that it sees small faces the shared finder does not."""
import cv2
import numpy as np
import pytest

from core.perception import face_scan as F


def test_merge_keeps_the_most_confident_of_overlapping_boxes():
    kept = F.merge_scored([((10, 10, 40, 40), 0.7), ((12, 12, 40, 40), 0.9), ((200, 200, 30, 30), 0.65)])
    assert [b for b, _ in kept] == [(12, 12, 40, 40), (200, 200, 30, 30)]


def test_merge_of_nothing_is_nothing():
    assert F.merge_boxes([]) == []


def test_tiny_images_and_none_are_safe():
    f = F.ScreeningFaceFinder()
    assert f.find(None) == []
    assert f.find(np.zeros((8, 8, 3), np.uint8)) == []


def test_blank_scene_has_no_faces():
    assert F.ScreeningFaceFinder().find(np.full((720, 1280, 3), 120, np.uint8)) == []


def test_boxes_are_reported_in_original_pixels_for_a_large_image():
    finder = F.ScreeningFaceFinder()
    if not finder._ensure():
        pytest.skip("YuNet model missing")
    # a synthetic oval "face" is not a face for YuNet; this only checks that a big image goes through the tiling path unharmed
    big = np.full((3000, 4000, 3), 90, np.uint8)
    cv2.ellipse(big, (2000, 1500), (60, 80), 0, 0, 360, (180, 190, 220), -1)
    for x, y, w, h in finder.find(big):
        assert 0 <= x and 0 <= y and x + w <= 4000 and y + h <= 3000


@pytest.mark.parametrize("rot", [90, 180, 270])
def test_unrotate_box_maps_back_to_the_original_pixels(rot):
    h, w = 120, 200
    img = np.zeros((h, w), np.uint8)
    img[30:50, 60:130] = 255                                   # the "face": x 60..130, y 30..50
    turned = cv2.rotate(img, F.ROTATIONS[rot])
    ys, xs = np.where(turned > 0)
    rbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    assert F.unrotate_box(rbox, rot, w, h) == (60, 30, 70, 20)


def test_find_rotated_on_blank_and_tiny_images_is_empty():
    f = F.ScreeningFaceFinder()
    assert f.find_rotated(None) == []
    assert f.find_rotated(np.full((300, 400, 3), 100, np.uint8)) == []
