# Video Dimension Checks — Design Spec

**Date:** 2026-10-02
**Builds on:** the image and audio specs (`2026-10-02-dimension-checks-foundation-image-design.md`, `2026-10-02-audio-dimension-checks-design.md`). All shared decisions carry over: Finding model, hybrid capped scoring, hard-block hash gate, recognition-only, advisory channels, five bands, OOD gate, `ui/stages.py`, no new dependencies, no automatic commits.

## 1. Decisions specific to video

| Topic | Decision |
|---|---|
| Score integration | `VideoAIDetector.analyze_video(..., extra_log_lrs=None)`. Video pools by weighted average, so after pooling `p_ai` is shifted in base-10 log-odds space (same rule as audio). Empty/None leaves results unchanged. Same caps (0.25 per finding, 0.40 explicit generator string, ±0.40 total). |
| What may score | Only explicit self-declared generator strings in container text fields (+0.40). Interlace/cadence/container-table findings are informational: they describe provenance and reliability, not AI-vs-real. Absence never scores. |
| Container parsing | Pure-Python ISOBMFF (MP4/MOV) box parser (top-level by seeking, `moov` fully parsed up to 32 MB), minimal EBML (MKV/WebM) info, AVI RIFF size check. |
| Frames | One bounded read of the first ≤ 90 consecutive frames (grayscale, ≤ 480 px wide) shared by the signal checks. |
| Recognition | Reuses the scientific-format gate (DICOM cine, HDF5/NetCDF, AEDAT event streams). |
| Commits | Not made automatically. |

## 2. Checks

`video_detector/dimension_checks/`: `_common.py`, `integrity.py`, `container.py`, `signal.py`, `context.py`, `legal.py`, `lifecycle.py`, `reliability.py`, `fit_ood.py`, `__init__.py` (`VideoDimensionAnalysis`, `check_video_gates`, `gate_short_circuit_result`, `summarize_for_evidence_trail`, `temporal_vector`).

| Stage | check_id | Class | llr |
|---|---|---|---|
| file_integrity | `format_sniff` (ftyp / EBML / RIFF-AVI vs extension) | SECURITY | – |
| file_integrity | `trailing_data` (ISOBMFF box sum vs file; AVI RIFF size) | SECURITY | – |
| file_integrity | `polyglot_signatures` (shared bytescan) | SECURITY | – |
| file_integrity | `prompt_injection_text` (udta/ilst text, XMP) | SECURITY | – |
| container | `isobmff_boxes` (brands, box order, fast-start vs trailing `moov`, fragmentation, `uuid`/C2PA box, overruns) | METADATA_WEAK | – |
| container | `isobmff_tables` (`stts`/`stsz`/`stss`/`stco` consistency, CFR vs VFR, fps from `mdhd`) | METADATA_WEAK | – |
| container | `mp4_metadata` (mvhd/tkhd/mdhd times, handler/encoder strings, capture-tool cues such as OBS; explicit generator strings) | METADATA_WEAK | +0.40 on explicit generator string |
| container | `telemetry_tracks` (GoPro `gpmd`, Sony `rtmd`, Google `camm`, DJI subtitle/meta tracks, `©xyz` location) | METADATA_WEAK | – |
| container | `encoder_sei` (x264/x265/libavcodec option strings in the bitstream) | METADATA_WEAK | – |
| container | `ebml_info` (DocType, MuxingApp, WritingApp for MKV/WebM) | METADATA_WEAK | – |
| signal | `interlacing` (container `fiel` + row-combing metric on moving frames) | PHYSICAL_SIGNAL | – |
| signal | `temporal_cadence` (duplicate frames, 3:2 pulldown / frame-rate-conversion cadence) | PHYSICAL_SIGNAL | – |
| context (post) | `video_fingerprint` (up to 16 frame pHashes; optional local index; set-overlap match) | CONTEXT | – |
| legal (post) | rights notice, location privacy (©xyz / telemetry), biometric notice (faces), AI-disclosure label | LEGAL_FLAG | – |
| lifecycle (post) | `transcoding_cascade` | RELIABILITY | – |
| reliability (post) | `confidence_limiters` (heavy compression via bits-per-pixel, low resolution, very short, interlaced, VFR, probable cascade) | RELIABILITY | – |

Interlacing: combing metric C = mean|adjacent-row diff| / mean|two-rows-apart diff| on frames with motion; progressive content gives about 0.5–0.8, combed (interlaced + motion) > 1.0. Cadence: from consecutive-frame differences, a "repeat" is d < 0.15 x median(d); repeats at a constant period of 5 frames (>= 80% of cycles) indicate 3:2 pulldown, other regular periods indicate frame-rate conversion, isolated repeats indicate duplication/dropped-frame repair.

## 3. Gates, bands, OOD, open set

Gates: hard-block SHA-256 + scientific-format recognition. Band from `ai_percentage`. OOD gate on a 5-dim temporal vector `[mean_motion_delta, log1p(motion_variance), flicker_score, flicker_ratio, mean_lum_jump]`, fitted by `python -m video_detector.dimension_checks.fit_ood` from the video media library. `UNKNOWN_SOURCE` when attribution `model_key == "unknown"` and `ai_percentage >= 60`.

## 4. Integration

`VideoForensicPipeline.analyze` and `ui.batch_ui.process_single_video` share the helpers: gates → validation → `VideoDimensionAnalysis.run_pre()` → detector (`extra_log_lrs`) → content/attribution/cross-modal → `run_post()`. Result adds `dimension_report`, `confidence_band`, `ood`, `attribution_open_set`, `gate`. UI: `render_linear_video_pipeline_results` gains the gate banner and Stages 1b, 3b, 4b, 6b via `ui/stages.py`.

## 5. Testing

TDD with hand-built ISOBMFF boxes (no codec needed), cv2 `mp4v` clips, and ffmpeg/libx264 clips (skip if ffmpeg missing). Identity regression for `analyze_video`. Full suite + AppTest render.

## 6. Docs

Video report Appendix F updated; README section.

## 7. Out of scope

Rolling-shutter and temporal PRNU analysis, optical-flow consistency, virtual-camera driver descriptors (not recoverable from a file), SEI NAL decoding beyond text strings, HLS/DASH segment analysis, telemetry-vs-optical-flow validation, watermark detectors, face-swap/lip-sync detection, 3D/VR/stereo analysis.
