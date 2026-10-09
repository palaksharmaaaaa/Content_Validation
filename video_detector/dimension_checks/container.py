"""
video_detector.dimension_checks.container: container / metadata forensics.

All findings are METADATA_WEAK: boxes, tags and sample tables are trivially forged and are routinely
rewritten by benign editors and platforms. The only score effect is an explicit self-declaration of a
known generative-video tool in a software/comment-type field (+0.40, the explicit-generator cap).
Absence of any marker is never scored.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from core.forensics.bytescan import extract_xmp
from core.forensics.config import EXPLICIT_GENERATOR_LLR_CAP
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from video_detector.dimension_checks import _common as C
from video_detector.provenance import KNOWN_VIDEO_GENERATOR_SIGNATURES

STAGE = "container"
DIM = "Section20"
EXPLICIT_GENERATOR_LLR = 0.40

_GEN_RX = [(sig, re.compile(r"\b" + re.escape(sig) + r"\b", re.I)) for sig in KNOWN_VIDEO_GENERATOR_SIGNATURES]
_SOFTWARE_KEYS = ("\xa9too", "\xa9swr", "\xa9cmt", "\xa9enc", "\xa9inf", "desc", "ldes", "\xa9des", "\xa9dsc", "\xa9src")

_CAPTURE = {
    "obs": r"\blibobs\b|\bobs[- ]studio\b|\bobs\b", "streamlabs": r"streamlabs", "manycam": r"manycam", "xsplit": r"xsplit",
    "droidcam": r"droidcam", "snap-camera": r"snap camera", "camtasia": r"camtasia", "screenflow": r"screenflow",
    "quicktime-screen-recording": r"screen ?recording",
}
_EDITORS = {
    "premiere": r"premiere", "davinci": r"davinci", "final-cut": r"final cut", "capcut": r"capcut", "kdenlive": r"kdenlive",
    "handbrake": r"handbrake", "shotcut": r"shotcut", "imovie": r"imovie", "filmora": r"filmora", "inshot": r"inshot",
}
_LIBS = {"libavformat": r"\blavf\d|libavformat", "libavcodec": r"\blavc\d|libavcodec", "ffmpeg": r"ffmpeg", "gpac": r"\bgpac\b|mp4box",
         "apple-coremedia": r"core media"}

_TELEMETRY = {
    "gpmd": "GoPro GPMF telemetry (accelerometer / gyro / GPS)", "rtmd": "Sony real-time lens/camera metadata",
    "camm": "Google camera-motion metadata (camm)", "djmd": "DJI flight metadata", "dbgi": "DJI debug metadata",
    "tx3g": "timed-text track (e.g. DJI flight-log subtitles)", "mett": "timed metadata track", "fdsc": "GoPro device-info track",
}


def _f(check_id, title, status, severity, detail, data=None, llr=None, cap=None) -> Finding:
    kw = {"llr_cap": cap} if cap is not None else {}
    return Finding(check_id=check_id, dimension=DIM, stage=STAGE, title=title, status=status, severity=severity,
                   evidence_class=EvidenceClass.METADATA_WEAK, detail=detail, data=data or {}, llr=llr, **kw)


def _na(check_id, title, why) -> Finding:
    return _f(check_id, title, FindingStatus.NOT_APPLICABLE, Severity.NONE, why)


def _is_isobmff(ctx: CheckContext) -> bool:
    head, _t, _s = C.read_windows(ctx.path)
    return C.sniff_video_format(head) in ("mp4", "mov")


def _moov(ctx: CheckContext):
    top = C.walk_top_level(ctx.path)
    return top, C.load_moov(ctx.path, top)


def _generator_in(texts: List[str]) -> Optional[str]:
    for text in texts:
        for sig, rx in _GEN_RX:
            if rx.search(text):
                return sig
    return None


def _tools(texts: List[str]) -> Dict[str, List[str]]:
    blob = " \n".join(texts).lower()
    return {
        "capture_tools": [k for k, v in _CAPTURE.items() if re.search(v, blob)],
        "editors": [k for k, v in _EDITORS.items() if re.search(v, blob)],
        "libraries": [k for k, v in _LIBS.items() if re.search(v, blob)],
    }


# ---------------------------------------------------------------------------------------------
@registry.register("video", "isobmff_boxes")
def check_isobmff_boxes(ctx: CheckContext) -> Finding:
    title = "ISOBMFF box structure"
    if not _is_isobmff(ctx):
        return _na("isobmff_boxes", title, "Not an MP4/MOV (ISOBMFF) file.")
    top = C.walk_top_level(ctx.path)
    types = [b["type"] for b in top["boxes"]]
    ftyp = next((b for b in top["boxes"] if b["type"] == "ftyp"), None)
    brands: List[str] = []
    major = None
    if ftyp:
        body = C.read_range(ctx.path, ftyp["offset"] + ftyp["header"], min(ftyp["size"] - ftyp["header"], 256))
        major = body[:4].decode("latin-1")
        brands = [body[i : i + 4].decode("latin-1") for i in range(8, len(body) - 3, 4)]
    idx_moov = types.index("moov") if "moov" in types else -1
    idx_mdat = types.index("mdat") if "mdat" in types else -1
    layout = "NO_MOOV" if idx_moov < 0 else ("FASTSTART" if idx_mdat < 0 or idx_moov < idx_mdat else "MOOV_AT_END")
    c2pa = False
    for b in top["boxes"]:
        if b["type"] == "uuid" and b["size"] <= 4 * 1024 * 1024:
            if C.read_range(ctx.path, b["offset"] + b["header"], 16) == C.C2PA_UUID:
                c2pa = True
    data = {"box_types": types[:50], "major_brand": major, "compatible_brands": brands[:8], "layout": layout,
            "fragmented": "moof" in types, "c2pa_box": c2pa, "issues": top["issues"]}
    problems = list(top["issues"])
    if layout == "NO_MOOV" and "moof" not in types:
        problems.append("no 'moov' box: the file has no movie index and may be unplayable or cut")
    if problems:
        return _f("isobmff_boxes", title, FindingStatus.WARN, Severity.MEDIUM, "; ".join(problems) + ".", data)
    layout_note = {
        "FASTSTART": "moov before mdat (fast-start layout: typical of web export or `ffmpeg -movflags faststart`)",
        "MOOV_AT_END": "moov after mdat (typical of direct camera/phone recording or default ffmpeg muxing)",
    }.get(layout, "fragmented layout")
    extra = " A C2PA manifest box is present (presence only; not cryptographically validated)." if c2pa else ""
    return _f("isobmff_boxes", title, FindingStatus.INFO, Severity.NONE,
              f"Brand '{major}', {layout_note}.{extra} Box layout is weak evidence of origin only.", data)


# ---------------------------------------------------------------------------------------------
@registry.register("video", "isobmff_tables")
def check_isobmff_tables(ctx: CheckContext) -> Finding:
    title = "Sample-table consistency"
    if not _is_isobmff(ctx):
        return _na("isobmff_tables", title, "Not an MP4/MOV (ISOBMFF) file.")
    top, moov = _moov(ctx)
    if not moov or not moov["tracks"]:
        return _na("isobmff_tables", title, "No parseable movie index (moov) was found.")
    size = top["file_size"]
    tracks_out: List[Dict[str, Any]] = []
    any_issue = any_vfr = False
    for t in moov["tracks"]:
        if t["handler"] != "vide":
            continue
        issues: List[str] = []
        stts = t["stts"] if t["stts"] is not None else np.zeros((0, 2), np.uint64)
        n_stts = int(stts[:, 0].sum()) if len(stts) else 0
        ts = (t["mdhd"] or {}).get("timescale") or 0
        cfr, fps = False, None
        if len(stts):
            deltas = stts[:, 1].astype(np.int64)
            counts = stts[:, 0].astype(np.int64)
            if np.any((deltas == 0) & (counts > 0)):
                issues.append("zero-duration samples in stts (duplicate timestamps)")
            if n_stts:
                mode_delta = int(deltas[np.argmax(counts)])
                if counts[deltas == mode_delta].sum() >= 0.99 * n_stts and mode_delta > 0:
                    cfr = True
                    fps = round(ts / mode_delta, 3) if ts else None
        if t["stsz"] and t["stsz"]["count"] != n_stts:
            issues.append(f"stts/stsz sample-count mismatch (stts {n_stts}, stsz {t['stsz']['count']})")
        count = t["stsz"]["count"] if t["stsz"] else n_stts
        if t["stss"] is not None and len(t["stss"]):
            arr = t["stss"].astype(np.int64)
            if arr.min() < 1 or arr.max() > count or np.any(np.diff(arr) < 0):
                issues.append("sync-sample (stss) indices are out of range or not ascending")
        if t["stco"] is not None and len(t["stco"]):
            off = t["stco"].astype(np.int64)
            if off.max() > size or np.any(np.diff(off) < 0):
                issues.append("chunk offsets (stco) lie beyond the file or are not ascending")
            total = C.total_samples_from_stsc(t["stsc"], len(off))
            if total is not None and count and total != count:
                issues.append(f"stsc/stco imply {total} samples but stsz declares {count}")
        any_issue |= bool(issues)
        any_vfr |= not cfr
        tracks_out.append({"samples": count, "fps": fps, "cfr": cfr, "timescale": ts, "issues": issues,
                           "duration_s": round((t["mdhd"] or {}).get("duration", 0) / ts, 2) if ts else None})
    if not tracks_out:
        return _na("isobmff_tables", title, "No video track found.")
    data = {"tracks": tracks_out}
    if any_issue:
        msgs = "; ".join(i for tr in tracks_out for i in tr["issues"])
        return _f("isobmff_tables", title, FindingStatus.WARN, Severity.MEDIUM,
                  f"Sample tables are internally inconsistent: {msgs}. Typical of truncated, spliced, or hand-edited files.", data)
    if any_vfr:
        return _f("isobmff_tables", title, FindingStatus.INFO, Severity.NONE,
                  "Variable frame rate (typical of screen recordings, phone video and edited timelines); tables are otherwise consistent.", data)
    return _f("isobmff_tables", title, FindingStatus.PASS, Severity.NONE,
              f"Sample tables are consistent; constant frame rate about {tracks_out[0]['fps']} fps.", data)


# ---------------------------------------------------------------------------------------------
def _xmp_creator_tools(xmp: str) -> List[str]:
    return re.findall(r"CreatorTool[^>]*>([^<]{1,200})<", xmp) + re.findall(r'CreatorTool="([^"]{1,200})"', xmp)


@registry.register("video", "mp4_metadata")
def check_mp4_metadata(ctx: CheckContext) -> Finding:
    title = "MP4/MOV metadata & tool strings"
    if not _is_isobmff(ctx):
        return _na("mp4_metadata", title, "Not an MP4/MOV (ISOBMFF) file.")
    top, moov = _moov(ctx)
    head, tail, _size = C.read_windows(ctx.path)
    xmp = extract_xmp(head) or extract_xmp(tail)
    if not moov:
        return _na("mp4_metadata", title, "No parseable movie index (moov) was found.")
    mv = moov["mvhd"] or {}
    creation = C.mac_time(mv.get("creation", 0))
    modification = C.mac_time(mv.get("modification", 0))
    paradoxes: List[str] = []
    now = datetime.utcnow()
    if creation and modification and modification < creation - timedelta(seconds=60):
        paradoxes.append("modification time precedes creation time")
    for name, v in (("creation", creation), ("modification", modification)):
        if v and v > now + timedelta(days=1):
            paradoxes.append(f"{name} time is in the future")
    handler_names = [t["handler_name"] for t in moov["tracks"] if t["handler_name"]]
    texts_all = list(moov["text_items"].values()) + handler_names + _xmp_creator_tools(xmp)
    soft_texts = [v for k, v in moov["text_items"].items() if k in _SOFTWARE_KEYS] + handler_names + _xmp_creator_tools(xmp)
    gen = _generator_in(soft_texts)
    tools = _tools(texts_all)
    data = {"creation": creation.isoformat() if creation else None, "modification": modification.isoformat() if modification else None,
            "paradoxes": paradoxes, "handler_names": handler_names, "text_keys": sorted(moov["text_items"].keys()),
            "tools": tools["capture_tools"] + tools["editors"] + tools["libraries"], "capture_tools": tools["capture_tools"],
            "editors": tools["editors"], "libraries": tools["libraries"], "generator": gen}
    if gen:
        return _f("mp4_metadata", title, FindingStatus.WARN, Severity.HIGH,
                  f"Software/comment fields name a known generative video tool ({gen}); self-declared AI generation.",
                  data, llr=EXPLICIT_GENERATOR_LLR, cap=EXPLICIT_GENERATOR_LLR_CAP)
    if paradoxes:
        return _f("mp4_metadata", title, FindingStatus.WARN, Severity.MEDIUM,
                  "Implausible container timestamps: " + "; ".join(paradoxes) + ".", data)
    bits = []
    if tools["capture_tools"]:
        bits.append("screen/virtual-camera capture software strings: " + ", ".join(tools["capture_tools"]))
    if tools["editors"]:
        bits.append("editing/transcoding software: " + ", ".join(tools["editors"]))
    if tools["libraries"]:
        bits.append("muxer/encoder libraries: " + ", ".join(tools["libraries"]))
    if not creation:
        bits.append("creation time is unset (stripped or machine-produced)")
    detail = ("; ".join(bits) if bits else "No notable tool strings.") + ". Strings are forgeable and often rewritten by platforms."
    return _f("mp4_metadata", title, FindingStatus.INFO, Severity.NONE, detail[0].upper() + detail[1:], data)


# ---------------------------------------------------------------------------------------------
@registry.register("video", "telemetry_tracks")
def check_telemetry_tracks(ctx: CheckContext) -> Finding:
    title = "Hardware telemetry tracks"
    if not _is_isobmff(ctx):
        return _na("telemetry_tracks", title, "Not an MP4/MOV (ISOBMFF) file.")
    _top, moov = _moov(ctx)
    if not moov:
        return _na("telemetry_tracks", title, "No parseable movie index (moov) was found.")
    found = []
    for t in moov["tracks"]:
        desc = _TELEMETRY.get(t.get("sample_entry") or "")
        if desc:
            found.append(desc)
    loc = moov["text_items"].get("\xa9xyz")
    data = {"telemetry": sorted(set(found)), "location": loc}
    if found or loc:
        parts = []
        if found:
            parts.append("telemetry track(s): " + "; ".join(sorted(set(found))))
        if loc:
            parts.append(f"embedded location {loc}")
        return _f("telemetry_tracks", title, FindingStatus.INFO, Severity.NONE,
                  "Found " + " and ".join(parts) + ". This check reports presence only; comparing IMU data with optical flow "
                  "is not implemented.", data)
    return _f("telemetry_tracks", title, FindingStatus.INFO, Severity.NONE,
              "No hardware telemetry tracks (absence is common and is not evidence of synthesis).", data)


# ---------------------------------------------------------------------------------------------
@registry.register("video", "encoder_sei")
def check_encoder_sei(ctx: CheckContext) -> Finding:
    title = "Encoder strings in the bitstream"
    head, _t, size = C.read_windows(ctx.path)
    fmt = C.sniff_video_format(head)
    if fmt == "unknown":
        return _na("encoder_sei", title, "Unrecognized container.")
    if fmt in ("mp4", "mov"):
        top = C.walk_top_level(ctx.path)
        mdat = next((b for b in top["boxes"] if b["type"] == "mdat"), None)
        data_bytes = C.read_range(ctx.path, mdat["offset"] + mdat["header"], 256 * 1024) if mdat else b""
    else:
        data_bytes = head[: 1024 * 1024]
    enc = C.find_encoder_strings(data_bytes)
    if enc:
        names = ", ".join(enc.keys())
        return _f("encoder_sei", title, FindingStatus.INFO, Severity.NONE,
                  f"Software encoder string(s) embedded in the stream: {names}. Hardware camera encoders usually carry none; "
                  "re-encoded or exported files often do.", {"encoders": enc})
    return _f("encoder_sei", title, FindingStatus.INFO, Severity.NONE,
              "No x264/x265/libavcodec option strings found near the start of the stream.", {"encoders": {}})


# ---------------------------------------------------------------------------------------------
@registry.register("video", "ebml_info")
def check_ebml_info(ctx: CheckContext) -> Finding:
    title = "Matroska/WebM writer strings"
    head, _t, _s = C.read_windows(ctx.path)
    if C.sniff_video_format(head) != "mkv":
        return _na("ebml_info", title, "Not a Matroska/WebM file.")
    info = C.ebml_info(head[:65536])
    texts = [v for v in (info["muxing_app"], info["writing_app"]) if v]
    gen = _generator_in(texts)
    data = dict(info, generator=gen)
    if gen:
        return _f("ebml_info", title, FindingStatus.WARN, Severity.HIGH,
                  f"Writer strings name a known generative video tool ({gen}); self-declared AI generation.",
                  data, llr=EXPLICIT_GENERATOR_LLR, cap=EXPLICIT_GENERATOR_LLR_CAP)
    if texts:
        return _f("ebml_info", title, FindingStatus.INFO, Severity.NONE,
                  f"DocType {info['doc_type']}, muxing app '{info['muxing_app']}', writing app '{info['writing_app']}'. Strings are forgeable.", data)
    return _f("ebml_info", title, FindingStatus.INFO, Severity.NONE,
              f"DocType {info['doc_type']}; no muxing/writing app strings (stripped or hand-built file).", data)
