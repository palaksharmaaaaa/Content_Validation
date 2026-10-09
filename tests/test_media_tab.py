"""The upload/link tab: results survive unrelated interactions, files are written once, long names keep their extension."""

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from core.security import sanitize_filename
from ui import media_tab


def _link_page(img_path):
    from pathlib import Path as P

    import streamlit as st

    import ui.media_tab as mt
    from ui.media_tab import MediaTabSpec, render_media_tab

    mt.fetch_media_from_url = lambda url, expected_type, dest_dir: {"success": True, "file_path": img_path, "filename": "linked.jpg"}
    mt.run_batch_pipeline = lambda items, **k: [{"filename": i["filename"], "success": True, "path": i["path"]} for i in items]
    spec = MediaTabSpec(key="img", modality="image", noun="Image", file_types=["jpg"], hint="", url_placeholder="",
                        render_result=lambda it: st.write("RESULT:" + it["filename"]))
    render_media_tab(spec, {}, "balanced", P(img_path).parent)
    st.selectbox("some unrelated widget", ["a", "b"])


def _results(at):
    return [m.value for m in at.markdown if "RESULT" in m.value]


def test_a_fetched_link_result_survives_touching_another_widget_and_can_be_forgotten(tmp_path):
    p = tmp_path / "x.jpg"
    Image.fromarray(np.zeros((80, 80, 3), np.uint8)).save(p)
    at = AppTest.from_function(_link_page, args=(str(p),), default_timeout=60).run()
    at.text_area[0].set_value("https://example.com/a.jpg").run()
    at.button(key="btn_fetch_img").click()
    at = at.run()
    assert _results(at) == ["RESULT:linked.jpg"]
    at.selectbox[0].select("b").run()
    assert _results(at) == ["RESULT:linked.jpg"]                      # the run after any other click still shows it
    at.button(key="btn_forget_img").click()
    at = at.run()
    assert _results(at) == []


def test_a_link_whose_file_was_wiped_is_dropped_not_analysed(tmp_path):
    p = tmp_path / "x.jpg"
    Image.fromarray(np.zeros((80, 80, 3), np.uint8)).save(p)
    at = AppTest.from_function(_link_page, args=(str(p),), default_timeout=60).run()
    at.text_area[0].set_value("https://example.com/a.jpg").run()
    at.button(key="btn_fetch_img").click()
    at = at.run()
    p.unlink()                                                         # "Clear uploaded files" removed it
    at.selectbox[0].select("b").run()
    assert _results(at) == [] and not at.exception


class _Upload:
    def __init__(self, name, data, file_id):
        self.name, self.size, self.file_id, self._data = name, len(data), file_id, data

    def getbuffer(self):
        return self._data


def test_uploads_are_written_once_and_same_names_never_collide(tmp_path):
    a, b = _Upload("photo.jpg", b"AAAA", "id-1"), _Upload("photo.jpg", b"BBBBBB", "id-2")
    pa, pb = media_tab._save_upload("img", tmp_path, a, 0), media_tab._save_upload("img", tmp_path, b, 1)
    assert pa != pb and pa.read_bytes() == b"AAAA" and pb.read_bytes() == b"BBBBBB"
    before = pa.stat().st_mtime_ns
    pa.write_bytes(b"AAAA")                                            # same size on disk: a re-run must not rewrite it
    stamp = pa.stat().st_mtime_ns
    media_tab._save_upload("img", tmp_path, a, 0)
    assert pa.stat().st_mtime_ns == stamp and stamp >= before


def test_a_damaged_partial_file_is_rewritten(tmp_path):
    up = _Upload("clip.mp4", b"0123456789", "id-9")
    path = media_tab._save_upload("vid", tmp_path, up, 0)
    path.write_bytes(b"0123")                                          # interrupted earlier write
    assert media_tab._save_upload("vid", tmp_path, up, 0).read_bytes() == b"0123456789"


def test_a_very_long_name_keeps_its_extension_and_a_safe_length():
    name = "holiday " * 40 + ".JPEG"
    out = sanitize_filename(name, max_len=80)
    assert len(out) <= 80 and out.lower().endswith(".jpeg")
    assert sanitize_filename("short name.png") == "short_name.png"
    assert sanitize_filename("a" * 300) == "a" * 90                    # no extension to keep: plain truncation


def _picker_page(rows):
    from ui.batch_views import render_batch_overview

    chosen = render_batch_overview(rows, "image")
    import streamlit as st

    st.write("OPENED:" + str(chosen["path"]))


def test_two_files_with_the_same_name_and_verdict_open_independently():
    decision = {"final_status": "LIKELY_AUTHENTIC", "authenticity_probabilities": {"p_ai": 10.0, "p_real": 85.0, "p_undecided": 5.0}}
    rows = [{"filename": "photo.jpg", "success": True, "path": "first", "decision": decision},
            {"filename": "photo.jpg", "success": True, "path": "second", "decision": decision}]
    at = AppTest.from_function(_picker_page, args=(rows,), default_timeout=60).run()
    assert [m.value for m in at.markdown if "OPENED" in m.value] == ["OPENED:first"]
    at.selectbox[0].select(1).run()
    assert [m.value for m in at.markdown if "OPENED" in m.value] == ["OPENED:second"]


def test_profile_blocks_survive_a_none_container_and_auto_detect_covers_every_supported_extension():
    from audio_detector.config import SUPPORTED_EXTENSIONS as A
    from image_detector.config import SUPPORTED_EXTENSIONS as I
    from ui.profile_view import add_ui_profile_blocks, modality_of_extension as _modality_of
    from video_detector.config import SUPPORTED_EXTENSIONS as V

    assert all(_modality_of(e) == "image" for e in I) and all(_modality_of(e) == "video" for e in V) and all(_modality_of(e) == "audio" for e in A)
    assert _modality_of(".jfif") == "image" and _modality_of(".xyz") is None
    out = add_ui_profile_blocks("video", {"valid": True, "filename": "v", "container_format": None})
    assert out["file_identity"]["mime_type"] == "video/mp4"


def test_the_app_accepts_what_each_package_supports_up_to_its_own_size_limit(tmp_path):
    """A 150 MB video is within the documented 500 MB limit; the app used to reject it at a blanket 100 MB."""
    import cv2

    from ui import validators
    from video_detector.config import MAX_FILE_SIZE_MB as VIDEO_LIMIT

    assert validators.MAX_FILE_SIZE_MB_BY_TYPE == {"image": 100.0, "video": 500.0, "audio": 200.0}
    p = tmp_path / "big.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(30):
        vw.write(np.full((48, 64, 3), 100 + i, np.uint8))
    vw.release()
    with open(p, "ab") as f:
        f.write(b"\x00" * (150 * 1024 * 1024))
    res = validators.validate_file(p)
    assert res["readable"] is True and res["size_valid"] is True and VIDEO_LIMIT >= 150
    assert validators.SUPPORTED_AUDIO_EXTENSIONS == {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a"}        # .wma was never supported


def test_a_download_is_capped_at_the_limit_of_its_media_type(monkeypatch):
    from ui import validators

    seen = {}

    class Fake:
        def __init__(self, max_mb, timeout_seconds):
            seen["mb"] = max_mb

        def fetch(self, url, dest_dir=None, expected_type="image"):
            return {"success": False, "error": "stop here"}

    monkeypatch.setattr(validators, "SecureUrlFetcher", Fake)
    for kind, mb in (("image", 100), ("audio", 200), ("video", 500)):
        validators.fetch_media_from_url("https://example.com/x", expected_type=kind)
        assert seen["mb"] == mb
    validators.fetch_media_from_url("https://example.com/x", expected_type="video", max_mb=7)
    assert seen["mb"] == 7


def test_a_big_image_is_shrunk_for_the_preview_and_a_small_one_is_not(tmp_path):
    from ui.results import media

    big, small = tmp_path / "big.png", tmp_path / "small.png"
    Image.fromarray(np.zeros((2400, 3200, 3), np.uint8)).save(big)
    Image.fromarray(np.zeros((300, 400, 3), np.uint8)).save(small)
    shrunk = media._preview_source(big)
    assert max(shrunk.size) == media.PREVIEW_MAX_SIDE
    assert media._preview_source(small) == str(small)
    assert media._preview_source(tmp_path / "not_an_image.bin") == str(tmp_path / "not_an_image.bin")
