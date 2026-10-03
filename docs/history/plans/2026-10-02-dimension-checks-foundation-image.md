# Dimension Checks Foundation + Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. This plan is executed inline in the authoring session; tasks give exact files, interfaces and test cases, and the code is written test-first during execution.

**Goal:** Build a modality-agnostic dimension-check foundation in `core/` and wire the image modality end to end (checks, gates, bands, OOD, open-set attribution, UI stages, docs).

**Architecture:** A `core/forensics` registry runs isolated checks that return typed `Finding`s; only `PHYSICAL_SIGNAL`/`METADATA_WEAK` findings may add capped log-odds (passed to `ImageAIDetector.predict(extra_log_lrs=...)`). Gates (hard-block, scientific-format recognition) short-circuit; band/OOD/attribution flags sit beside P. A shared helper feeds both `ImageForensicPipeline` and `ui.batch_ui.process_single_image`; generic Streamlit stage components render the report.

**Tech Stack:** Python 3.14, Pillow, numpy, OpenCV (cv2), stdlib (`struct`, `zlib`, `re`, `hashlib`), pytest, Streamlit. No new dependencies.

## Global Constraints

- No new third-party dependencies (Pillow, numpy, cv2, stdlib only).
- Only `PHYSICAL_SIGNAL` and `METADATA_WEAK` findings may carry `llr`; registry forces `llr=None` otherwise.
- Per-finding cap 0.25 (explicit generator-parameter PNG chunk: 0.40); total of all extra terms clamped to ±0.40; log-odds are base-10, positive = toward AI.
- Absence of a marker is never evidence (no llr).
- `predict()` output must be unchanged when `extra_log_lrs` is empty/None.
- Five bands exactly: `HIGH_CONFIDENCE_SYNTHETIC` (P≥0.995), `LEANING_SYNTHETIC` (0.60<P<0.995), `INCONCLUSIVE` (0.40≤P≤0.60), `LEANING_AUTHENTIC` (0.005<P<0.40), `HIGH_CONFIDENCE_AUTHENTIC` (P≤0.005).
- A failing check must never raise out of the registry (becomes `ERROR`).
- Scans of file bytes are bounded (head/tail windows ≤ 4 MB).
- Do not commit; leave git state for the user. Existing tests must stay green.

## File Structure

| File | Responsibility |
|---|---|
| `core/forensics/__init__.py` | Public exports |
| `core/forensics/schemas.py` | `Severity`, `FindingStatus`, `EvidenceClass`, `Finding`, `DimensionReport` |
| `core/forensics/config.py` | Caps, env var names, default paths |
| `core/forensics/registry.py` | `CheckContext`, `CheckRegistry`, `registry`, `run_checks`, `build_report` |
| `core/forensics/gates.py` | `HardBlockGate`, `GateResult`, `recognize_scientific_format` |
| `core/forensics/ood.py` | `OODGate` |
| `core/bands.py` | `Band`, `classify_band`, `classify_band_from_percent` |
| `image_detector/dimension_checks/_common.py` | Byte windows, sniff, JPEG quant helpers, dHash/pHash |
| `image_detector/dimension_checks/integrity.py` | Format sniff, trailing data, polyglot, SVG, prompt injection |
| `image_detector/dimension_checks/formats.py` | JPEG quant tables, PNG chunks |
| `image_detector/dimension_checks/metadata.py` | EXIF consistency, timestamps, thumbnail, ICC |
| `image_detector/dimension_checks/context.py` | Perceptual hash + optional local index |
| `image_detector/dimension_checks/legal.py` | Rights/privacy/biometric/AI-label flags |
| `image_detector/dimension_checks/lifecycle.py` | Platform re-encode likelihood |
| `image_detector/dimension_checks/reliability.py` | Confidence limiters |
| `image_detector/dimension_checks/__init__.py` | `ImageDimensionAnalysis`, `check_image_gates` |
| `ui/stages.py` | Modality-agnostic stage renderers + pure row helpers |

Modified: `image_detector/detector.py` (extra_log_lrs), `image_detector/attribution.py` (unknown_source), `image_detector/pipeline.py`, `ui/batch_ui.py`, `ui/feedback_ui.py`, `README.md`, the three `GLOBAL_*_REPORT.md`.

---

### Task 1: Core schemas, config, registry
**Files:** Create `core/forensics/{__init__,schemas,config,registry}.py`; Test `core/tests/test_forensics_registry.py`.
**Produces:** `Finding(check_id, dimension, stage, title, status, severity, evidence_class, detail="", data={}, llr=None, llr_cap=0.25)`; `DimensionReport(findings, gates, score_terms, reliability)`; `registry.register(modality, check_id, phase="pre")`; `run_checks(modality, ctx, phase)->List[Finding]`; `build_report(findings, gates=None)->DimensionReport`; `CheckContext(path, modality, profile, provenance, content, ai_result, extra)`.
- [ ] Tests: ineligible class llr forced None; per-finding cap; explicit cap 0.40; total clamp ±0.40 proportional; exception in check → one ERROR finding; list-return supported; phase separation.
- [ ] Implement; run `pytest core/tests/test_forensics_registry.py -q` → PASS.

### Task 2: Bands + OOD gate
**Files:** Create `core/bands.py`, `core/forensics/ood.py`; Test `core/tests/test_bands.py`, `core/tests/test_ood.py`.
**Produces:** `Band` str-enum; `classify_band(p_ai: float)`; `classify_band_from_percent(p: float)`; `OODGate.fit(X)`, `.save(path)`, `OODGate.load(path)`, `.score(vec)->{"status","distance","threshold"}`.
- [ ] Tests: every boundary (0.995, 0.60, 0.40, 0.005 inclusive per spec), percent helper; OOD NOT_CALIBRATED when unfitted/missing file; fit on gaussian cloud → in-dist point IN_DISTRIBUTION, far point OUT_OF_DISTRIBUTION; save/load round trip.
- [ ] Implement; run tests.

### Task 3: Gates
**Files:** Create `core/forensics/gates.py`; Test `core/tests/test_forensics_gates.py`.
**Produces:** `HardBlockGate(blocklist_path=None).check(path)->GateResult(status in {"INACTIVE","CLEAR","HARD_BLOCK_ESCALATE"}, sha256, triggered)`; `recognize_scientific_format(path)->Optional[{"type","description"}]`.
- [ ] Tests: no file → INACTIVE; matching sha → HARD_BLOCK_ESCALATE; comments/blank lines/uppercase ok; FITS/DICOM/EXR/HDF5/NetCDF/AEDAT magic; plain PNG → None.
- [ ] Implement; run tests.

### Task 4: Image integrity checks
**Files:** Create `image_detector/dimension_checks/{_common,integrity}.py`, package `__init__.py` stub; Test `image_detector/tests/test_dimension_checks_integrity.py`.
**Produces:** checks `format_sniff`, `trailing_data`, `polyglot_signatures`, `svg_active_content`, `prompt_injection_text` (class SECURITY, stage `file_integrity`, dimension "O/Q"), registered for modality `"image"` phase `"pre"`.
- [ ] Tests (Pillow fixtures): clean JPEG/PNG PASS; PNG saved as `.jpg` → format mismatch WARN; JPEG+appended ZIP (valid zipfile bytes) → trailing WARN and polyglot FAIL; SVG with `<script>` FAIL, clean SVG PASS, PNG → NOT_APPLICABLE; PNG tEXt "ignore previous instructions…" → WARN; MPF-style double JPEG not flagged polyglot.
- [ ] Implement; run tests.

### Task 5: Format checks
**Files:** Create `image_detector/dimension_checks/formats.py`; Test `image_detector/tests/test_dimension_checks_formats.py`.
**Produces:** `jpeg_quant_tables`, `png_chunks` (METADATA_WEAK, stage `formats`); `_common.estimate_libjpeg_quality(tables)->Optional[int]`.
- [ ] Tests: Pillow-saved JPEG q=90 → standard libjpeg, est quality 90, no llr; custom quant tables via `quantization=` → non-standard, llr −0.10 only if `provenance.has_camera_hardware`; PNG with A1111 `parameters` text ("Steps: 20, Sampler: Euler") → WARN, llr +0.40, cap 0.40; ComfyUI `prompt` JSON with `class_type` → +0.40; plain PNG → INFO, llr None.
- [ ] Implement; run tests.

### Task 6: Metadata checks
**Files:** Create `image_detector/dimension_checks/metadata.py`; Test `image_detector/tests/test_dimension_checks_metadata.py`.
**Produces:** `exif_consistency`, `timestamp_sanity`, `thumbnail_match`, `icc_profile`; helper `_common.extract_exif_thumbnail(path)->Optional[bytes]`.
- [ ] Tests: full coherent camera EXIF + has_camera_hardware → llr −0.25; Apple make with Galaxy-style model → WARN +0.25; absent EXIF → INFO llr None; Digitized earlier than Original → WARN +0.15; matching thumbnail PASS; mismatched thumbnail (hand-built EXIF IFD1) WARN +0.25; no thumbnail NOT_APPLICABLE.
- [ ] Implement; run tests.

### Task 7: Context, legal, lifecycle, reliability + analysis helper
**Files:** Create `image_detector/dimension_checks/{context,legal,lifecycle,reliability}.py`, finalize `__init__.py`; Test `image_detector/tests/test_dimension_checks_post.py`.
**Produces:** `ImageDimensionAnalysis(path, profile, provenance)` with `.run_pre()->Dict[str,float]` (score_terms), `.run_post(content, ai_result, attribution=None)->Dict` (report dict plus `confidence_band`, `ood`, `attribution_open_set`); `check_image_gates(path)->Dict` with keys `hard_block`, `recognition`.
- [ ] Tests: dHash/pHash stable for same image and differ for different; local index hit → WARN CONTEXT; GPS EXIF → location privacy WARN; faces_count>0 → biometric notice; AI-likely (ai_percentage 80) without label → disclosure WARN, with `c2pa_present` → PASS; WhatsApp-like (no EXIF, 1600px, q≈70) → likelihood ≥0.6; tiny image → reliability limiters; band from ai_percentage; OOD NOT_CALIBRATED default; `score_terms` clamped.
- [ ] Implement; run tests.

### Task 8: Detector, attribution, pipeline, UI-path integration
**Files:** Modify `image_detector/detector.py` (predict signature + merge before pooling), `image_detector/attribution.py` (+`unknown_source`, `UNKNOWN_SOURCE_THRESHOLD`), `image_detector/config.py`, `image_detector/pipeline.py`, `ui/batch_ui.py:process_single_image`; Test `image_detector/tests/test_dimension_integration.py`.
- [ ] Tests: `predict(..., extra_log_lrs=None)` equals baseline result; `extra_log_lrs={"x":0.3}` appears in `log_likelihood_ratios` as `dim_x` and shifts P toward AI; pipeline result has `dimension_report`, `confidence_band`, `gate`; hard-block file → `final_status == "HARD_BLOCK_ESCALATE"` without probabilities; `.fits`-magic file → `RECOGNIZED_OUT_OF_SCOPE`.
- [ ] Implement; run integration + full image test file.

### Task 9: UI stages
**Files:** Create `ui/stages.py`; Modify `ui/feedback_ui.py:render_linear_image_pipeline_results`; Test `tests/test_ui_stages_logic.py`.
**Produces:** `findings_to_rows(findings, show_weight)->List[dict]`, `band_badge(band)->(emoji,label)`, `render_gate_banner`, `render_findings_stage`, `render_confidence_and_reliability`.
- [ ] Tests (logic only): rows contain status badge/class/weight text; weight column shows "capped, weak" for METADATA_WEAK with llr; band badges for all five.
- [ ] Implement renderers and insert Stage 1b/3b/4b/6b + gate banner; `python -c "import ui.feedback_ui"` imports cleanly.

### Task 10: Docs and full verification
**Files:** Modify `README.md`; append "Implementation Status" appendix to the three `GLOBAL_*_REPORT.md` (+ revision row).
- [ ] Write appendices (implemented / partial / spec-only per dimension), README section.
- [ ] Run full `pytest -q`; report results honestly.
