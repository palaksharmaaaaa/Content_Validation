"""
image_detector.dimension_checks.metadata: metadata & container forensics (report Section 21).

Metadata is forgeable and routinely stripped by benign platforms, so every signal here is weak.
Absence never yields log-odds. The thumbnail mismatch is the only PHYSICAL_SIGNAL (it compares
pixels, not labels).
"""
from __future__ import annotations

import logging

import io
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from image_detector.dimension_checks import _common as C

logger = logging.getLogger(__name__)

STAGE = "metadata"
DIM = "Section21"

INCOHERENT_LLR = 0.25
TIMESTAMP_PARADOX_LLR = 0.15
THUMBNAIL_MISMATCH_LLR = 0.25
THUMBNAIL_MAD_THRESHOLD = 20.0  # mean absolute gray-level difference (0-255) on a 32x24 grid

_BRANDS = {
    "apple": ("apple", "iphone", "ipad"),
    "samsung": ("samsung", "sm-", "galaxy"),
    "google": ("google", "pixel"),
    "canon": ("canon", "eos"),
    "nikon": ("nikon",),
    "sony": ("sony", "ilce", "dsc-", "ilme", "zv-"),
    "fujifilm": ("fujifilm", "fuji", "x-t", "gfx"),
    "panasonic": ("panasonic", "dmc-", "dc-"),
    "olympus": ("olympus", "om system", "e-m"),
    "huawei": ("huawei",),
    "xiaomi": ("xiaomi", "redmi"),
    "oneplus": ("oneplus",),
    "leica": ("leica",),
    "dji": ("dji",),
    "gopro": ("gopro",),
}


def _brand_of(text: str) -> Optional[str]:
    low = (text or "").lower()
    for brand, tokens in _BRANDS.items():
        if any(t in low for t in tokens):
            return brand
    return None


def _read_exif(path) -> Dict[str, Dict[int, Any]]:
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            ifd0 = dict(ex)
            sub = dict(ex.get_ifd(0x8769))
            gps = dict(ex.get_ifd(0x8825))
    except Exception as exc:
        logger.debug("_read_exif: ignored %s: %s", type(exc).__name__, exc)
        return {"ifd0": {}, "exif": {}, "gps": {}}
    return {"ifd0": ifd0, "exif": sub, "gps": gps}


def _s(v: Any) -> str:
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    return str(v).replace("\x00", "").strip()


def _f(check_id, title, status, severity, detail, data=None, llr=None, cls=EvidenceClass.METADATA_WEAK) -> Finding:
    return Finding(check_id=check_id, dimension=DIM, stage=STAGE, title=title, status=status, severity=severity,
                   evidence_class=cls, detail=detail, data=data or {}, llr=llr)


@registry.register("image", "exif_consistency")
def check_exif_consistency(ctx: CheckContext) -> Finding:
    ex = _read_exif(ctx.path)
    ifd0, sub = ex["ifd0"], ex["exif"]
    make, model = _s(ifd0.get(271, "")), _s(ifd0.get(272, ""))
    present = {
        "Make": bool(make), "Model": bool(model), "DateTimeOriginal": 0x9003 in sub,
        "ExposureTime": 0x829A in sub, "FNumber": 0x829D in sub, "ISO": 0x8827 in sub, "FocalLength": 0x920A in sub,
    }
    data = {"make": make, "model": model, "present": present}
    if not any(present.values()):
        return _f("exif_consistency", "EXIF coherence", FindingStatus.INFO, Severity.NONE,
                  "No camera EXIF present (common after platform re-encoding); no inference drawn.", data)
    bm, bo = _brand_of(make), _brand_of(model)
    if bm and bo and bm != bo:
        data["brand_conflict"] = {"make_brand": bm, "model_brand": bo}
        return _f("exif_consistency", "EXIF coherence", FindingStatus.WARN, Severity.MEDIUM,
                  f"EXIF Make '{make}' and Model '{model}' belong to different brands; typical of grafted or forged metadata.",
                  data, llr=INCOHERENT_LLR)
    if all(present.values()):
        # Coherent metadata is NOT rewarded: it is exactly what a forger supplies, and the detector already credits
        # camera EXIF (subject to its untrusted-metadata policy). Only incoherence adds suspicion.
        return _f("exif_consistency", "EXIF coherence", FindingStatus.PASS, Severity.NONE,
                  "Full camera-grade EXIF is internally coherent (unauthenticated metadata can be forged; no score credit).", data)
    return _f("exif_consistency", "EXIF coherence", FindingStatus.INFO, Severity.NONE,
              "EXIF is partial; coherent as far as present, no inference drawn.", data)


def _parse_dt(v: Any) -> Optional[datetime]:
    try:
        return datetime.strptime(_s(v)[:19], "%Y:%m:%d %H:%M:%S")
    except Exception as exc:
        logger.debug("_parse_dt: ignored %s: %s", type(exc).__name__, exc)
        return None


@registry.register("image", "timestamp_sanity")
def check_timestamp_sanity(ctx: CheckContext) -> Finding:
    ex = _read_exif(ctx.path)
    orig = _parse_dt(ex["exif"].get(0x9003))
    digi = _parse_dt(ex["exif"].get(0x9004))
    mod = _parse_dt(ex["ifd0"].get(306))
    gps_date = None
    try:
        gps_date = datetime.strptime(_s(ex["gps"].get(29, ""))[:10], "%Y:%m:%d")
    except Exception as exc:
        logger.debug("check_timestamp_sanity: ignored %s: %s", type(exc).__name__, exc)
    if not any((orig, digi, mod, gps_date)):
        return _f("timestamp_sanity", "Timestamp plausibility", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  "No EXIF timestamps present.")
    paradoxes: List[str] = []
    slack = timedelta(seconds=60)
    if orig and digi and digi < orig - slack:
        paradoxes.append("DateTimeDigitized precedes DateTimeOriginal")
    if orig and mod and mod < orig - slack:
        paradoxes.append("ModifyDate precedes DateTimeOriginal")
    now = datetime.utcnow()
    for name, v in (("DateTimeOriginal", orig), ("DateTimeDigitized", digi), ("ModifyDate", mod)):
        if v and v > now + timedelta(days=1):
            paradoxes.append(f"{name} is in the future")
    if orig and gps_date and abs((orig.date() - gps_date.date()).days) > 1:
        paradoxes.append("GPSDateStamp disagrees with DateTimeOriginal by more than a day")
    data = {"paradoxes": paradoxes}
    if paradoxes:
        return _f("timestamp_sanity", "Timestamp plausibility", FindingStatus.WARN, Severity.MEDIUM,
                  "Impossible or inconsistent timestamps: " + "; ".join(paradoxes) + ".", data, llr=TIMESTAMP_PARADOX_LLR)
    return _f("timestamp_sanity", "Timestamp plausibility", FindingStatus.PASS, Severity.NONE,
              "EXIF timestamps are ordered and plausible.", data)


def _gray_grid(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L").resize((32, 24), Image.LANCZOS), dtype=np.float32)


@registry.register("image", "thumbnail_match")
def check_thumbnail_match(ctx: CheckContext) -> Finding:
    head, _t, _s_ = C.read_windows(ctx.path)
    if C.sniff_format(head) != "jpeg":
        return _f("thumbnail_match", "Embedded thumbnail vs. main image", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  "Not a JPEG.", cls=EvidenceClass.PHYSICAL_SIGNAL)
    thumb_bytes = C.extract_exif_thumbnail(ctx.path)
    if not thumb_bytes:
        return _f("thumbnail_match", "Embedded thumbnail vs. main image", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  "No embedded EXIF thumbnail.", cls=EvidenceClass.PHYSICAL_SIGNAL)
    try:
        thumb = Image.open(io.BytesIO(thumb_bytes))
        thumb.load()
        with Image.open(ctx.path) as main:
            main.load()
            w, h = main.size
            tw, th = thumb.size
            r, tr = w / max(1, h), tw / max(1, th)
            aspect_ok = abs(r - tr) / r < 0.05
            swapped = (not aspect_ok) and abs(r - 1.0 / max(tr, 1e-6)) / r < 0.05
            mad = None
            d_ham = None
            if aspect_ok:
                a, b = _gray_grid(main), _gray_grid(thumb)
                mad = float(np.abs(a - b).mean())
                d_ham = C.hamming64(C.dhash64(main), C.dhash64(thumb))
    except Exception as exc:
        return _f("thumbnail_match", "Embedded thumbnail vs. main image", FindingStatus.INFO, Severity.NONE,
                  f"Thumbnail could not be compared ({type(exc).__name__}).", cls=EvidenceClass.PHYSICAL_SIGNAL)
    data = {"main_size": [w, h], "thumb_size": [tw, th], "aspect_match": aspect_ok, "mean_abs_diff": mad, "dhash_hamming": d_ham}
    if swapped:
        return _f("thumbnail_match", "Embedded thumbnail vs. main image", FindingStatus.INFO, Severity.NONE,
                  "Thumbnail is rotated relative to the main image (orientation tag); content not compared.", data,
                  cls=EvidenceClass.PHYSICAL_SIGNAL)
    if not aspect_ok or (mad is not None and mad > THUMBNAIL_MAD_THRESHOLD):
        why = "different aspect ratio (cropped/resized after capture)" if not aspect_ok else "different visual content"
        return _f("thumbnail_match", "Embedded thumbnail vs. main image", FindingStatus.WARN, Severity.HIGH,
                  f"Embedded thumbnail disagrees with the main image: {why}. The file was likely edited without updating its preview.",
                  data, llr=THUMBNAIL_MISMATCH_LLR, cls=EvidenceClass.PHYSICAL_SIGNAL)
    return _f("thumbnail_match", "Embedded thumbnail vs. main image", FindingStatus.PASS, Severity.NONE,
              "Embedded thumbnail agrees with the main image.", data, cls=EvidenceClass.PHYSICAL_SIGNAL)


@registry.register("image", "icc_profile")
def check_icc_profile(ctx: CheckContext) -> Finding:
    icc = None
    try:
        with Image.open(ctx.path) as im:
            icc = im.info.get("icc_profile")
    except Exception as exc:
        logger.debug("check_icc_profile: ignored %s: %s", type(exc).__name__, exc)
    if not icc:
        return _f("icc_profile", "ICC color profile", FindingStatus.INFO, Severity.NONE,
                  "No embedded ICC profile (common for screenshots, web exports and stripped files).", {"present": False})
    desc = None
    try:
        from PIL import ImageCms

        desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
    except Exception as exc:
        logger.debug("check_icc_profile: ignored %s: %s", type(exc).__name__, exc)
    return _f("icc_profile", "ICC color profile", FindingStatus.INFO, Severity.NONE,
              f"Embedded ICC profile present{f' ({desc})' if desc else ''}.", {"present": True, "descriptor": desc, "bytes": len(icc)})
