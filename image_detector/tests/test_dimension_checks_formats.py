"""Format/compression dimension checks: JPEG quantization tables + PNG chunk inventory."""
import json

import pytest
from PIL import Image, PngImagePlugin

from core.forensics.registry import CheckContext, build_report
from core.forensics.schemas import EvidenceClass, FindingStatus
from image_detector.dimension_checks import _common as C
from image_detector.dimension_checks import formats


def _ctx(path, camera=False):
    return CheckContext(path=path, modality="image", provenance={"has_camera_hardware": camera})


def _jpeg(tmp_path, quality=90, qtables=None, name="a.jpg"):
    p = tmp_path / name
    kwargs = {"qtables": qtables} if qtables else {"quality": quality}
    Image.new("RGB", (64, 64), (120, 130, 140)).save(p, "JPEG", **kwargs)
    return p


def _png(tmp_path, text=None, name="a.png"):
    p = tmp_path / name
    info = PngImagePlugin.PngInfo()
    for k, v in (text or {}).items():
        info.add_text(k, v)
    Image.new("RGB", (32, 32), (1, 2, 3)).save(p, "PNG", pnginfo=info)
    return p


@pytest.mark.parametrize("q", [50, 75, 90, 95])
def test_libjpeg_quality_estimated(tmp_path, q):
    p = _jpeg(tmp_path, quality=q)
    with Image.open(p) as im:
        assert C.estimate_libjpeg_quality(im.quantization) == q


def test_standard_tables_are_info_without_llr(tmp_path):
    f = formats.check_jpeg_quant_tables(_ctx(_jpeg(tmp_path, 85), camera=True))
    assert f.status == FindingStatus.INFO
    assert f.data["standard_libjpeg"] is True
    assert f.data["estimated_quality"] == 85
    assert f.llr is None
    assert f.evidence_class == EvidenceClass.METADATA_WEAK


def test_nonstandard_tables_are_informational_only(tmp_path):
    custom = [[7 + (i % 5) for i in range(64)], [9 + (i % 3) for i in range(64)]]
    p = _jpeg(tmp_path, qtables=custom)
    for camera in (True, False):
        f = formats.check_jpeg_quant_tables(_ctx(p, camera=camera))
        assert f.data["standard_libjpeg"] is False
        assert f.llr is None  # forgeable: never scored


def test_jpeg_check_not_applicable_for_png(tmp_path):
    assert formats.check_jpeg_quant_tables(_ctx(_png(tmp_path))).status == FindingStatus.NOT_APPLICABLE


def test_a1111_parameters_chunk(tmp_path):
    p = _png(tmp_path, {"parameters": "a cat\nNegative prompt: blur\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 5"})
    f = formats.check_png_chunks(_ctx(p))
    assert f.status == FindingStatus.WARN
    assert f.data["generator"] == "Stable Diffusion (A1111/Forge)"
    assert f.llr == pytest.approx(0.40)
    report = build_report([f])
    assert report.score_terms["png_chunks"] == pytest.approx(0.40)  # explicit cap override honored


def test_comfyui_prompt_chunk(tmp_path):
    prompt = json.dumps({"3": {"class_type": "KSampler", "inputs": {"seed": 1}}})
    p = _png(tmp_path, {"prompt": prompt})
    f = formats.check_png_chunks(_ctx(p))
    assert f.data["generator"] == "ComfyUI"
    assert f.llr == pytest.approx(0.40)


def test_novelai_software_chunk(tmp_path):
    p = _png(tmp_path, {"Software": "NovelAI", "Comment": "{}"})
    assert formats.check_png_chunks(_ctx(p)).data["generator"] == "NovelAI"


def test_plain_png_is_inventory_only(tmp_path):
    f = formats.check_png_chunks(_ctx(_png(tmp_path)))
    assert f.status == FindingStatus.INFO
    assert f.llr is None
    assert "IHDR" in f.data["chunks"]


def test_png_check_not_applicable_for_jpeg(tmp_path):
    assert formats.check_png_chunks(_ctx(_jpeg(tmp_path))).status == FindingStatus.NOT_APPLICABLE


def test_png_text_chunk_after_large_idat_is_found(tmp_path):
    """A text chunk placed after > 4 MB of image data (beyond the old head window) must still be read."""
    import struct
    import zlib

    import numpy as np

    arr = np.random.default_rng(0).integers(0, 255, (1400, 1400, 3), dtype=np.uint8)  # incompressible: IDAT > 4 MB
    base = tmp_path / "big.png"
    Image.fromarray(arr).save(base, "PNG", compress_level=0)
    data = base.read_bytes()
    assert len(data) > 4 * 1024 * 1024
    text = b"parameters\x00a cat\nSteps: 20, Sampler: Euler, CFG scale: 7"
    chunk = struct.pack(">I", len(text)) + b"tEXt" + text + struct.pack(">I", zlib.crc32(b"tEXt" + text) & 0xFFFFFFFF)
    iend = data.rfind(b"IEND") - 4
    p = tmp_path / "late.png"
    p.write_bytes(data[:iend] + chunk + data[iend:])
    f = formats.check_png_chunks(_ctx(p))
    assert f.data["generator"] == "Stable Diffusion (A1111/Forge)"
    assert f.llr == pytest.approx(0.40)
