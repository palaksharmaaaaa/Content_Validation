"""The generic AI-generator metadata match must not fire on ordinary words or camera fields."""
import pytest

from image_detector.features import _apply_generic_ai_signature, _blank_metadata_info

XMP_IMAGE_NUMBER = '<x:xmpmeta><rdf:Description aux:ImageNumber="6195" aux:LensID="154" aux:SerialNumber="123"/></x:xmpmeta>'


def _run(xmp="", tags=None):
    info = _blank_metadata_info()
    info["raw_tags"] = tags or {}
    _apply_generic_ai_signature(xmp, info)
    return info


def test_camera_image_number_is_not_the_imagen_generator():
    assert _run(XMP_IMAGE_NUMBER)["ai_signature_found"] is False


@pytest.mark.parametrize("tags", [
    {"ImageDescription": "tilt-shift of a runway at dusk"},
    {"UserComment": "my flux capacitor and a luma key"},
    {"Artist": "Gemini Studios, sora no iro"},
])
def test_ambiguous_names_in_free_text_do_not_count(tags):
    assert _run(tags=tags)["ai_signature_found"] is False


@pytest.mark.parametrize("tags,xmp", [
    ({"Software": "Midjourney v6"}, ""),
    ({"Software": "Google Imagen 3"}, ""),
    ({"ProcessingSoftware": "Flux.1 dev"}, ""),
    ({}, '<rdf:Description xmp:CreatorTool="Stable Diffusion XL"/>'),
    ({}, '<rdf:Description xmp:CreatorTool="Gemini"/>'),
    ({"ImageDescription": "made with ComfyUI"}, ""),         # distinctive names still count anywhere
])
def test_real_generator_declarations_still_count(tags, xmp):
    assert _run(xmp, tags)["ai_signature_found"] is True


def test_distinctive_name_must_be_a_whole_word():
    assert _run(tags={"Software": "comfyuiXYZ"})["ai_signature_found"] is False
