"""
audio_detector.dimension_checks.container: container / metadata forensics (report Section 20).

All findings are METADATA_WEAK: tags are trivially forged and routinely rewritten by benign
converters. The only score effect is the explicit self-declaration of a known generator in an
encoder/comment-type field (+0.40, the explicit-generator cap); absence is never scored.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

from audio_detector.dimension_checks import _common as C
from core.forensics.config import EXPLICIT_GENERATOR_LLR_CAP
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity

STAGE = "container"
DIM = "Section20"
EXPLICIT_GENERATOR_LLR = 0.40

_GENERATORS = {
    "elevenlabs": r"eleven\s?labs", "suno": r"\bsuno\b", "udio": r"\budio\b", "stable-audio": r"stable\s?audio",
    "musicgen": r"musicgen|audiocraft", "riffusion": r"riffusion", "coqui-xtts": r"\bxtts\b|coqui", "lyria": r"\blyria\b",
    "openai-voice": r"openai.{0,20}(tts|voice)", "playht": r"play\.?ht\b", "resemble": r"resemble\s?ai",
    "cosyvoice": r"cosyvoice", "chattts": r"chattts", "fish-speech": r"fish[\s-]?speech", "cartesia": r"\bcartesia\b",
    "murf": r"\bmurf\b", "wellsaid": r"wellsaid",
}
_GEN_RX = {k: re.compile(v, re.I) for k, v in _GENERATORS.items()}

# ID3 frames where a software/generator self-declaration is meaningful (not title/artist/album).
_ID3_SOFTWARE_FRAMES = ("TSSE", "TSS", "TENC", "TEN", "TPUB", "TCOP", "TXXX", "COMM")


def _generator_in(text_fields: Dict[str, str]) -> Optional[str]:
    for text in text_fields.values():
        for name, rx in _GEN_RX.items():
            if rx.search(text):
                return name
    return None


def _f(check_id, title, status, severity, detail, data=None, llr=None, cap=None) -> Finding:
    kw = {"llr_cap": cap} if cap is not None else {}
    return Finding(check_id=check_id, dimension=DIM, stage=STAGE, title=title, status=status, severity=severity,
                   evidence_class=EvidenceClass.METADATA_WEAK, detail=detail, data=data or {}, llr=llr, **kw)


def _na(check_id, title, why) -> Finding:
    return _f(check_id, title, FindingStatus.NOT_APPLICABLE, Severity.NONE, why)


# ---------------------------------------------------------------------------------------------
@registry.register("audio", "id3_tags")
def check_id3_tags(ctx: CheckContext) -> Finding:
    head, _tail, size = C.read_windows(ctx.path)
    fmt = C.sniff_audio_format(head)
    title = "ID3v2 tag forensics"
    if fmt not in ("mp3", "aiff"):
        return _na("id3_tags", title, f"ID3v2 not applicable to format '{fmt}'.")
    id3 = C.parse_id3v2(head)
    if not id3:
        return _f("id3_tags", title, FindingStatus.INFO, Severity.NONE,
                  "No ID3v2 tag (common for stripped or machine-produced files); no inference drawn.", {"present": False})
    data = {"present": True, "version": id3["version"], "frames": id3["frames"], "padding_bytes": id3["padding_bytes"],
            "has_cover_art": id3["has_apic"], "generator": None}
    if id3["total_len"] > size:
        return _f("id3_tags", title, FindingStatus.WARN, Severity.MEDIUM,
                  "ID3v2 header declares a tag larger than the file; malformed or tampered tag.", data)
    software = {k: v for k, v in id3["text"].items() if k.split(":")[0] in _ID3_SOFTWARE_FRAMES}
    gen = _generator_in(software)
    if gen:
        data["generator"] = gen
        return _f("id3_tags", title, FindingStatus.WARN, Severity.HIGH,
                  f"Encoder/comment tags name a known generative audio tool ({gen}); self-declared AI generation.",
                  data, llr=EXPLICIT_GENERATOR_LLR, cap=EXPLICIT_GENERATOR_LLR_CAP)
    return _f("id3_tags", title, FindingStatus.INFO, Severity.NONE,
              f"ID3v{id3['version']} with {len(id3['frames'])} frame(s); no generator declaration found.", data)


# ---------------------------------------------------------------------------------------------
@registry.register("audio", "riff_structure")
def check_riff_structure(ctx: CheckContext) -> Finding:
    head, _tail, size = C.read_windows(ctx.path)
    title = "RIFF / WAV structure"
    if C.sniff_audio_format(head) != "wav":
        return _na("riff_structure", title, "Not a RIFF/WAVE file.")
    riff = C.walk_riff(ctx.path)
    chunks = {c["id"]: c for c in riff["chunks"]}
    fmt = C.parse_wav_fmt(C.riff_chunk_body(ctx.path, chunks["fmt "])) if "fmt " in chunks else None
    bext = C.parse_bext(C.riff_chunk_body(ctx.path, chunks["bext"], limit=1 << 20)) if "bext" in chunks else None
    info: Dict[str, str] = {}
    for ch in riff["chunks"]:
        if ch["id"] == "LIST":
            info.update(C.parse_list_info(C.riff_chunk_body(ctx.path, ch)))
    data = {"chunks": [c["id"] for c in riff["chunks"]], "format": fmt, "bext": bext, "info": info, "issues": list(riff["issues"]),
            "generator": None}
    issues = list(riff["issues"])
    if fmt and fmt["format_tag"] not in (1, 3) and "fact" not in chunks:
        issues.append("non-PCM WAV lacks a 'fact' chunk (non-compliant writer)")
    data["issues"] = issues
    soft = {f"riff:{k}": v for k, v in info.items() if k in ("ISFT", "ICMT", "IENG", "ITCH")}
    if bext:
        soft.update({f"bext:{k}": v for k, v in bext.items() if k in ("originator", "description", "coding_history") and v})
    gen = _generator_in(soft)
    if gen:
        data["generator"] = gen
        return _f("riff_structure", title, FindingStatus.WARN, Severity.HIGH,
                  f"RIFF software/comment fields name a known generative audio tool ({gen}); self-declared AI generation.",
                  data, llr=EXPLICIT_GENERATOR_LLR, cap=EXPLICIT_GENERATOR_LLR_CAP)
    if issues:
        return _f("riff_structure", title, FindingStatus.WARN, Severity.MEDIUM, "RIFF structure issues: " + "; ".join(issues) + ".", data)
    extra = " Broadcast Wave `bext` chunk present." if bext else ""
    return _f("riff_structure", title, FindingStatus.INFO, Severity.NONE,
              f"Well-formed RIFF/WAVE with chunks {', '.join(data['chunks'])}.{extra} Chunk metadata is unauthenticated.", data)


# ---------------------------------------------------------------------------------------------
_FLAC_FMT = {16: "s16le", 24: "s24le"}


def _md5_of_decoded(path: Path, sample_fmt: str) -> Optional[str]:
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-f", sample_fmt, "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    h = hashlib.md5()
    try:
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(1 << 20), b""):
            h.update(chunk)
        proc.wait(timeout=120)
    except Exception:
        proc.kill()
        return None
    return h.hexdigest() if proc.returncode == 0 else None


@registry.register("audio", "flac_md5")
def check_flac_md5(ctx: CheckContext) -> Finding:
    head, _tail, _size = C.read_windows(ctx.path)
    title = "FLAC embedded MD5 signature"
    if C.sniff_audio_format(head) != "flac":
        return _na("flac_md5", title, "Not a FLAC file.")
    si = C.flac_streaminfo(head)
    if not si:
        return _f("flac_md5", title, FindingStatus.INFO, Severity.LOW, "STREAMINFO block could not be parsed.")
    if set(si["md5"]) == {"0"}:
        return _f("flac_md5", title, FindingStatus.INFO, Severity.NONE,
                  "The encoder did not store an MD5 of the audio (all zeros); integrity cannot be checked this way.", {"stored": si["md5"]})
    sample_fmt = _FLAC_FMT.get(si["bits"])
    if not sample_fmt:
        return _na("flac_md5", title, f"MD5 verification is only implemented for 16/24-bit FLAC (this file is {si['bits']}-bit).")
    actual = _md5_of_decoded(ctx.path, sample_fmt)
    if actual is None:
        return _na("flac_md5", title, "ffmpeg is unavailable or failed, so the audio could not be decoded for MD5 verification.")
    data = {"stored": si["md5"], "decoded": actual}
    if actual == si["md5"]:
        return _f("flac_md5", title, FindingStatus.PASS, Severity.NONE,
                  "Decoded audio matches the encoder's stored MD5: the FLAC stream is unmodified since encoding "
                  "(this does not prove the audio was unaltered before encoding).", data)
    return _f("flac_md5", title, FindingStatus.WARN, Severity.MEDIUM,
              "Decoded audio does not match the stored MD5: the stream was modified or corrupted after encoding "
              "(unless the MD5 was recomputed).", data)


# ---------------------------------------------------------------------------------------------
_ENCODER_RX = re.compile(rb"(LAME|Lavc|Lavf|GOGO|L3\.99r)[0-9A-Za-z\.]{0,9}")


@registry.register("audio", "mp3_encoder_tag")
def check_mp3_encoder_tag(ctx: CheckContext) -> Finding:
    head, _tail, _size = C.read_windows(ctx.path)
    title = "MP3 encoder / Xing-LAME tag"
    if C.sniff_audio_format(head) != "mp3":
        return _na("mp3_encoder_tag", title, "Not an MP3 file.")
    start = 0
    id3 = C.parse_id3v2(head)
    if id3:
        start = id3["total_len"]
    fr = C.mp3_first_frame(head, start)
    if not fr:
        return _f("mp3_encoder_tag", title, FindingStatus.INFO, Severity.LOW, "No decodable MPEG frame header found near the start.")
    frame = head[fr["offset"] : fr["offset"] + 512]
    xing = b"Xing" if b"Xing" in frame else (b"Info" if b"Info" in frame else None)
    m = _ENCODER_RX.search(frame)
    encoder = m.group(0).decode("latin-1") if m else None
    lowpass = None
    if m and encoder and encoder.startswith("LAME") and m.end() + 1 < len(frame) - 0:
        # LAME tag: 9-byte encoder string, 1 byte revision/VBR method, then lowpass in units of 100 Hz.
        pos = m.start() + 9
        if pos + 2 <= len(frame):
            lowpass = frame[pos + 1] * 100 or None
    data = {"encoder": encoder, "lowpass_hz": lowpass, "vbr_header": xing.decode() if xing else None,
            "sample_rate": fr["sample_rate"], "bitrate_kbps": fr["bitrate_kbps"], "mpeg_version": fr["mpeg_version"]}
    if encoder:
        return _f("mp3_encoder_tag", title, FindingStatus.INFO, Severity.NONE,
                  f"Encoder tag '{encoder}'" + (f", lowpass {lowpass} Hz" if lowpass else "") +
                  ". A valid tag attests only to the final encoding pass, not to the origin of the sound.", data)
    return _f("mp3_encoder_tag", title, FindingStatus.INFO, Severity.NONE,
              "No Xing/LAME encoder tag (hardware encoders, streams and re-wrapped files often lack it).", data)


# ---------------------------------------------------------------------------------------------
@registry.register("audio", "ogg_structure")
def check_ogg_structure(ctx: CheckContext) -> Finding:
    head, _tail, size = C.read_windows(ctx.path)
    title = "Ogg page integrity"
    if C.sniff_audio_format(head) != "ogg":
        return _na("ogg_structure", title, "Not an Ogg stream.")
    if size > C.MAX_OGG_BYTES:
        return _na("ogg_structure", title, "File exceeds the Ogg structural-scan size limit.")
    pages = [p for p in C.parse_ogg_pages(ctx.path.read_bytes()) if not p.get("truncated")]
    if not pages:
        return _f("ogg_structure", title, FindingStatus.INFO, Severity.LOW, "No complete Ogg pages could be parsed.")
    crc_fail = sum(1 for p in pages if not p["crc_ok"])
    by_serial: Dict[int, list] = {}
    for p in pages:
        by_serial.setdefault(p["serial"], []).append(p)
    gaps = regress = 0
    for plist in by_serial.values():
        for a, b in zip(plist, plist[1:]):
            if b["seq"] != a["seq"] + 1:
                gaps += 1
            if a["granule"] >= 0 and b["granule"] >= 0 and b["granule"] < a["granule"]:
                regress += 1
    data = {"pages": len(pages), "crc_failures": crc_fail, "sequence_gaps": gaps, "granule_regressions": regress,
            "serials": len(by_serial)}
    problems = []
    if crc_fail:
        problems.append(f"{crc_fail} page(s) fail their CRC (edited without CRC repair, or corrupted)")
    if gaps:
        problems.append(f"{gaps} page-sequence gap(s) (pages removed or reordered)")
    if regress:
        problems.append(f"{regress} granule-position regression(s) (non-monotonic timeline)")
    if problems:
        return _f("ogg_structure", title, FindingStatus.WARN, Severity.MEDIUM, "; ".join(problems) + ".", data)
    if len(by_serial) > 1:
        return _f("ogg_structure", title, FindingStatus.INFO, Severity.NONE,
                  f"{len(by_serial)} logical streams (chained or multiplexed Ogg); page CRCs and sequence are intact.", data)
    return _f("ogg_structure", title, FindingStatus.PASS, Severity.NONE,
              f"{len(pages)} pages, CRCs and page sequence intact.", data)
