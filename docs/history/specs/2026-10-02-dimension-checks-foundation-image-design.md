# Dimension Checks Foundation + Image Implementation — Design Spec

**Date:** 2026-10-02
**Status:** Approved by user (design presented in four parts; scope = foundation + image only).
**Governing documents:** `GLOBAL_IMAGE_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md` (REV5), and the Audio/Video reports for the shared foundation.

## 1. Goal

Make the image pipeline (and the shared `core/`) reflect the report's operational dimensions in proper stages, without disturbing the calibrated existing detector. Audio and video are separate later cycles that reuse the foundation built here.

## 2. Decisions (from brainstorming)

| Topic | Decision |
|---|---|
| Scope of this cycle | `core/` foundation + image modality end to end. Audio/video later. |
| Effect on verdict | **Hybrid.** Only `PHYSICAL_SIGNAL` / `METADATA_WEAK` findings may contribute capped log-odds. Security, legal, context, lifecycle and reliability findings never change P. |
| Hard-block (CSAM/NCII) | Pluggable SHA-256 hash-match hook against an operator-supplied local blocklist. No classifier. |
| Dimensions not judgeable from a file | Recognition-only (magic-byte sniff → "recognized, out of authenticity-scoring scope"). |
| New dependencies | None. Pillow, numpy, stdlib only. |
| Commits | Not made automatically; user commits. |

## 3. Architecture

### 3.1 `core/forensics/` (modality-agnostic)

- `schemas.py` — `Finding`, `DimensionReport`, enums `FindingStatus`, `EvidenceClass`.
- `config.py` — caps and paths (blocklist path via env `OMNI_HARDBLOCK_SHA256_FILE`, local hash index via `OMNI_CONTEXT_HASH_INDEX`).
- `registry.py` — `CheckRegistry.register(modality, stage, check_id)` decorator; `run_checks(modality, ctx)` runs every registered check under try/except (a failing check yields `ERROR`, never an exception), applies per-finding and total log-odds caps, and builds a `DimensionReport`.
- `gates.py` — `HardBlockGate` (SHA-256 vs blocklist file; inactive if no file), `recognize_scientific_format(path)` (FITS, DICOM, GeoTIFF, OpenEXR, HDF5, NetCDF, AEDAT/AER magic bytes).
- `ood.py` — `OODGate`: Mahalanobis distance on embeddings; `fit(embeddings)`, `save/load(stats_path)`, `score(embedding)`. Threshold = 99th percentile of in-sample distances. Without fitted stats the status is `NOT_CALIBRATED`; no threshold is invented.
- `bands.py` (at `core/bands.py`) — `classify_band(p_ai)` returning one of five bands per the report: `HIGH_CONFIDENCE_SYNTHETIC` (≥0.995), `LEANING_SYNTHETIC` (0.60<P<0.995), `INCONCLUSIVE` (0.40–0.60), `LEANING_AUTHENTIC` (0.005<P<0.40), `HIGH_CONFIDENCE_AUTHENTIC` (≤0.005). Inputs are fractions in [0,1] or percents; boundaries inclusive exactly as in the report.

### 3.2 Finding model

`Finding(check_id, dimension, stage, title, status, severity, evidence_class, detail, data, llr=None)`

- `status`: `PASS | WARN | FAIL | INFO | NOT_APPLICABLE | NOT_CALIBRATED | RECOGNIZED_OOS | ERROR`
- `severity`: `NONE | LOW | MEDIUM | HIGH | CRITICAL`
- `evidence_class`: `PHYSICAL_SIGNAL | METADATA_WEAK | SECURITY | LEGAL_FLAG | CONTEXT | RELIABILITY`
- `llr` (base-10 log-odds toward AI; negative = toward real) is **only honored** for `PHYSICAL_SIGNAL` and `METADATA_WEAK`; for any other class the registry forces it to `None`.
- Caps: per-finding ≤ 0.25 in magnitude (explicit generator-parameter chunk: 0.40 as the single allowed exception), total of all extra terms clamped to ±0.40.
- Absence is never evidence: a missing marker yields `llr=0`/`None`.

`DimensionReport.to_dict()` returns `{findings_by_stage, gates, score_terms, reliability, summary}` where `score_terms` is the dict passed to `ImageAIDetector.predict(extra_log_lrs=...)`.

### 3.3 Integration into the existing detector

- `ImageAIDetector.predict(..., extra_log_lrs: Optional[Dict[str, float]] = None)` — merged into `log_lrs` before `pool_bayesian_log_odds`. With `None`/empty input behavior is byte-for-byte unchanged (regression-tested).
- One shared helper `image_detector.dimension_checks.run_image_dimension_checks(path, profile, provenance, content, ai_result=None)` is used by **both** `ImageForensicPipeline.analyze` and `ui.batch_ui.process_single_image` so the two paths cannot drift. The UI path additionally starts passing `provenance=` to `predict` (it currently does not).
- Order: validator → hard-block gate + recognition gate → profiler → provenance → integrity/format/metadata checks (produce `score_terms`) → detector (with `extra_log_lrs`) → content analyzer → attribution (+ open-set `UNKNOWN_SOURCE`) → legal/context/lifecycle/reliability checks (need content + detector output) → band + OOD → dossier.
- A hard-block match short-circuits after the gate: result carries `final_status="HARD_BLOCK_ESCALATE"`, no probabilities.
- A recognized scientific format returns `final_status="RECOGNIZED_OUT_OF_SCOPE"` with the recognized type; no verdict.

### 3.4 Result additions

Pipeline/UI result dicts gain: `dimension_report`, `confidence_band`, `ood`, `attribution_open_set` (`UNKNOWN_SOURCE` flag), `gate`. Existing keys are untouched.

## 4. Image checks (this cycle)

Module layout: `image_detector/dimension_checks/{__init__,integrity,formats,metadata,context,legal,lifecycle,reliability}.py`, each registering via the core registry.

| Stage | check_id | Class | llr |
|---|---|---|---|
| file_integrity | `format_sniff` — true format vs extension | SECURITY | – |
| file_integrity | `trailing_data` — bytes after JPEG EOI / PNG IEND | SECURITY | – |
| file_integrity | `polyglot_signatures` — ZIP/PDF/HTML/PE/ELF signature inside payload | SECURITY | – |
| file_integrity | `svg_active_content` — `<script`, `on*=`, `<!ENTITY`, `javascript:` when content is XML/SVG | SECURITY | – |
| file_integrity | `prompt_injection_text` — instruction-like strings in EXIF/XMP/PNG text | SECURITY | – |
| formats | `jpeg_quant_tables` — standard libjpeg-scaled vs non-standard tables, est. quality, subsampling, progressive | METADATA_WEAK | −0.10 only if camera EXIF present and tables non-standard |
| formats | `png_chunks` — chunk inventory; A1111 `parameters` / ComfyUI `prompt`/`workflow` text | METADATA_WEAK | +0.40 for explicit generator-parameter chunk |
| metadata | `exif_consistency` — Make/Model/Lens coherence, required companion tags | METADATA_WEAK | ±0.25 (shared metadata cap) |
| metadata | `timestamp_sanity` — Original/Digitized/Modify ordering, GPS date vs capture | METADATA_WEAK | +0.15 on paradox |
| metadata | `thumbnail_match` — IFD1 thumbnail vs main image (aspect + dHash) | PHYSICAL_SIGNAL | +0.25 on mismatch |
| metadata | `icc_profile` — present/absent, descriptor | METADATA_WEAK | – (INFO only) |
| context | `perceptual_hash` — dHash/pHash; optional match against local index | CONTEXT | – |
| legal | `rights_and_licence` — Copyright/XMP rights/licence; GPS present; faces present; AI-disclosure label (C2PA/IPTC digitalSourceType) | LEGAL_FLAG | – |
| lifecycle | `platform_reencode` — likelihood of messaging/social re-encode | RELIABILITY | – |
| reliability | `confidence_limiters` — low resolution, heavy compression, small faces, laundering suspected | RELIABILITY | – |

**Superseded (2026-10-03 audit):** `exif_consistency` no longer emits −0.25 for coherent metadata, because coherent EXIF is exactly what a forger supplies; coherent EXIF is a PASS with no log-odds. It still emits +0.25 on incoherence (grafted/forged metadata) and never emits a value for absent metadata. Likewise `jpeg_quant_tables` is INFO only (no credit for non-standard tables), and the detector demotes camera EXIF that is contradicted by strong synthetic pixel evidence (`EXIF_CONTRADICTION_LR`).

Explicitly **not** implemented this cycle (stay spec-only in the report): presentation-attack / face-morph detection, double-JPEG coefficient analysis.

## 5. Open-set attribution

**As built:** `image_detector/attribution.py` is unchanged: it already returns `model_key="unknown"` when no generator profile scores at least 0.25. `ImageDimensionAnalysis.run_post()` derives `attribution_open_set.unknown_source = True` when `model_key == "unknown"` and the verdict is AI-leaning (P(AI) >= 60%). Authentic verdicts never raise it, and the existing authentic-attribution sanitizing in `core.decision` is untouched.

## 6. UI (`ui/stages.py`, modality-agnostic)

New renderers: `render_gate_banner(gate)`, `render_findings_stage(title, caption, findings, show_weight=False)`, `render_confidence_and_reliability(band, ood, reliability, attribution_open_set)`. The image linear flow gains: gate banner (top, only when triggered); Stage 1b File Integrity & Security; Stage 3b Metadata & Container Forensics (with evidence-weight column, labelled weak/capped); Stage 4b Confidence & Reliability (band badge, OOD status, UNKNOWN_SOURCE); Stage 6b Context, Rights & Lifecycle (each section marked "advisory — does not change the score"). Existing seven stages and their order are unchanged.

## 7. Error handling

- Each check isolated; failure → `Finding(status=ERROR)`; pipeline continues.
- Bounded I/O: head/tail windows ≤ 4 MB for scans; full-structure parsing only within the existing 100 MB validator limit.
- Missing blocklist/index/OOD stats → corresponding gate/check is `NOT_APPLICABLE`/`NOT_CALIBRATED`, never an error.

## 8. Testing (TDD)

- `core/tests/test_forensics_schemas_registry.py`, `test_forensics_gates.py`, `test_bands.py`, `test_ood.py`.
- `image_detector/tests/test_dimension_checks.py` with Pillow-generated fixtures: JPEG + appended ZIP, extension mismatch, PNG with A1111 `parameters` chunk, EXIF with inconsistent Make/Model, mismatched thumbnail, timestamp paradox, SVG with `<script>`, prompt-injection EXIF comment.
- Cap tests: llr forced to None for non-eligible classes; total clamp ±0.40; per-finding cap.
- Regression: `predict()` output unchanged when `extra_log_lrs` is empty; existing suite stays green.
- UI helper tests are logic-only (no Streamlit runtime): findings → rows transformation, band labels.

## 9. Documentation

- README: new "Dimension checks foundation" section.
- All three reports: an "Implementation Status" appendix per dimension (implemented / partial / spec-only). Image reflects this cycle's code; audio/video reflect current code and point to the next cycles.

## 10. Out of scope

Audio/video checks and UI parity (next cycles); new third-party dependencies; any CSAM/NCII classifier; trained OOD reference data (gate ships uncalibrated until fitted from the user's library).
