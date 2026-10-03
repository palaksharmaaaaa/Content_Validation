"""
image_detector.dimension_checks.formats: format/compression forensics (report Dimension O, §18).

Findings are METADATA_WEAK: they may add small capped log-odds, never when the signal is merely
absent. The explicit generator-parameter PNG chunk is the single case allowed the 0.40 cap.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Optional

from PIL import Image, JpegImagePlugin

from core.forensics.config import EXPLICIT_GENERATOR_LLR_CAP
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from image_detector.dimension_checks import _common as C

STAGE = "formats"
DIM = "O"

EXPLICIT_GENERATOR_LLR = 0.40


def _f(check_id, title, status, severity, detail, data=None, llr: Optional[float] = None, cap: Optional[float] = None) -> Finding:
    kw = {}
    if cap is not None:
        kw["llr_cap"] = cap
    return Finding(
        check_id=check_id, dimension=DIM, stage=STAGE, title=title, status=status, severity=severity,
        evidence_class=EvidenceClass.METADATA_WEAK, detail=detail, data=data or {}, llr=llr, **kw,
    )


@registry.register("image", "jpeg_quant_tables")
def check_jpeg_quant_tables(ctx: CheckContext) -> Finding:
    head, _tail, _size = C.read_windows(ctx.path)
    if C.sniff_format(head) != "jpeg":
        return _f("jpeg_quant_tables", "JPEG quantization tables", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  "Not a JPEG file.")
    with Image.open(ctx.path) as im:
        tables = dict(getattr(im, "quantization", {}) or {})
        try:
            subsampling = JpegImagePlugin.get_sampling(im)
        except Exception:
            subsampling = -1
        progressive = bool(im.info.get("progressive") or im.info.get("progression"))
    quality = C.estimate_libjpeg_quality(tables)
    standard = quality is not None
    sub_names = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}
    data = {
        "standard_libjpeg": standard,
        "estimated_quality": quality,
        "subsampling": sub_names.get(subsampling, "unknown"),
        "progressive": progressive,
        "table_count": len(tables),
    }
    if standard:
        return _f("jpeg_quant_tables", "JPEG quantization tables", FindingStatus.INFO, Severity.NONE,
                  f"Standard libjpeg-scaled tables (quality about {quality}): typical of a software re-save or web export.", data)
    return _f("jpeg_quant_tables", "JPEG quantization tables", FindingStatus.INFO, Severity.NONE,
              "Non-standard quantization tables (camera ISP, Photoshop, or a custom encoder). Informational only: tables are "
              "trivially copied, so no score credit is given.", data)


_A1111_MARKERS = ("Steps:", "Sampler:", "CFG scale:")


def _generator_from_text(fields: dict) -> Optional[str]:
    params = fields.get("parameters", "")
    if params and sum(m in params for m in _A1111_MARKERS) >= 2:
        return "Stable Diffusion (A1111/Forge)"
    for key in ("prompt", "workflow"):
        raw = fields.get(key, "")
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if key == "prompt" and isinstance(obj, dict) and any(isinstance(v, dict) and "class_type" in v for v in obj.values()):
            return "ComfyUI"
        if key == "workflow" and isinstance(obj, dict) and "nodes" in obj:
            return "ComfyUI"
    if fields.get("Software", "").strip().lower() == "novelai":
        return "NovelAI"
    if "invokeai_metadata" in fields or "sd-metadata" in fields:
        return "InvokeAI"
    return None


@registry.register("image", "png_chunks")
def check_png_chunks(ctx: CheckContext) -> Finding:
    head, _tail, _size = C.read_windows(ctx.path)
    if C.sniff_format(head) != "png":
        return _f("png_chunks", "PNG chunk inventory", FindingStatus.NOT_APPLICABLE, Severity.NONE, "Not a PNG file.")
    chunks = C.parse_png_chunks(ctx.path)
    inventory = dict(Counter(name for name, _ in chunks))
    fields = C.png_text_fields(chunks)
    generator = _generator_from_text(fields)
    data = {"chunks": inventory, "text_keys": sorted(fields.keys()), "generator": generator}
    if generator:
        return _f("png_chunks", "PNG chunk inventory", FindingStatus.WARN, Severity.HIGH,
                  f"PNG text chunks contain {generator} generation parameters (self-declared AI generation).",
                  data, llr=EXPLICIT_GENERATOR_LLR, cap=EXPLICIT_GENERATOR_LLR_CAP)
    return _f("png_chunks", "PNG chunk inventory", FindingStatus.INFO, Severity.NONE,
              f"{len(chunks)} chunk(s); no generator parameter text found.", data)
