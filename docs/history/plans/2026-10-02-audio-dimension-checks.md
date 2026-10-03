# Audio Dimension Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Executed inline, test-first.

**Goal:** Wire the audio modality into the `core/forensics` foundation (checks, gates, bands, OOD, open-set, UI stages).
**Architecture / Tech stack / Global constraints:** identical to `2026-10-02-dimension-checks-foundation-image.md` (no new deps; caps 0.25/0.40/±0.40 base-10; absence never scored; checks never raise; bounded scans; no commits). Python is `.venv/Scripts/python.exe` (3.10).

## Tasks

1. **Shared bytescan + symbolic recognition.** Create `core/forensics/bytescan.py` (`read_windows`, `scan_signatures`, `INJECTION_PATTERNS`, `find_injection`); add `recognize_symbolic_music` to `core/forensics/gates.py`; refactor `image_detector/dimension_checks/integrity.py` to use bytescan. Tests: `core/tests/test_bytescan.py`, extend `test_forensics_gates.py`; image integrity tests stay green.
2. **Audio helpers + integrity checks.** `audio_detector/dimension_checks/_common.py` (sniff, RIFF walker, ID3v2 parser, Ogg page parser + CRC, FLAC STREAMINFO, MP3 frame header, native PCM loader) and `integrity.py` (`format_sniff`, `trailing_data`, `polyglot_signatures`, `prompt_injection_text`). Tests: `audio_detector/tests/test_dimension_checks_integrity.py`.
3. **Container checks.** `container.py` (`id3_tags`, `riff_structure`, `flac_md5`, `mp3_encoder_tag`, `ogg_structure`). Tests: `test_dimension_checks_container.py`.
4. **Signal checks.** `signal.py` (`enf_trace`, `fake_hires`, `telephony_channel`, `loudness_dynamics`); checks read `ctx.extra["samples"]=(x, sr, dur)`. Tests: `test_dimension_checks_signal.py`.
5. **Post checks + analysis helper.** `context.py`, `legal.py`, `lifecycle.py`, `reliability.py`, `fit_ood.py`, `__init__.py` (`AudioDimensionAnalysis`, `check_audio_gates`, `gate_short_circuit_result`, `summarize_for_evidence_trail`, `normalize_provenance`). Tests: `test_dimension_checks_post.py`, `test_fit_ood.py`.
6. **Detector + pipeline + UI integration.** `AudioAIDetector.analyze_audio_file(extra_log_lrs=...)`; `AudioForensicPipeline.analyze`; `ui/batch_ui.process_single_audio`; `ui/feedback_ui.render_linear_audio_pipeline_results` stages. Tests: `test_dimension_integration.py`; AppTest render.
7. **Docs + full verification.** Audio report Appendix F + README + full `pytest`.
