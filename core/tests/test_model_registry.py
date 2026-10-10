"""The manifest is built from the files on this machine; nothing in it may be a typed-in number that the files do not support."""
import pytest

from core.model_registry import FINE_TUNED, HAND_WRITTEN, manifest


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_every_step_names_its_model_origin_data_and_licence(modality):
    info = manifest(modality)
    assert info["models"] and info["summary"]
    for m in info["models"]:
        for field in ("step", "name", "variant", "origin", "training_data", "training_size", "licence", "source"):
            assert str(m[field]).strip(), (modality, m["name"], field)
    assert info["models"][0]["origin"] == HAND_WRITTEN                  # the signal checks decide the verdict, and say they are hand-set


def test_image_training_total_is_read_from_the_checkpoint_not_typed_in(monkeypatch):
    import core.model_registry as reg

    monkeypatch.setattr(reg, "_read_checkpoint_meta", lambda path: {"train_samples": 70, "val_samples": 30, "unit": "images"})
    face = reg._face_model()
    assert face.trained_files == 100 and "100 images" in face.training_size and face.origin == FINE_TUNED
    monkeypatch.setattr(reg, "_read_checkpoint_meta", lambda path: None)
    assert reg._face_model().trained_files is None and reg._face_model().active is False


def test_a_checkpoint_saved_before_sizes_were_recorded_says_so():
    import core.model_registry as reg

    text, size, files = reg._trained_text({})
    assert files is None and "not recorded" in size


def test_unknown_modality_is_refused():
    with pytest.raises(ValueError):
        manifest("text")
