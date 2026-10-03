# Video Dimension Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Executed inline, test-first.

**Goal:** Wire the video modality into the `core/forensics` foundation (checks, gates, bands, OOD, open-set, UI stages).
**Architecture / Tech stack / Global constraints:** identical to the image and audio plans (no new deps; caps 0.25/0.40/±0.40 base-10; absence never scored; checks never raise; bounded scans; no commits). Python is `.venv/Scripts/python.exe` (3.10). Avoid shell heredocs with escapes: write scripts with the Write tool.

## Tasks

1. **ISOBMFF/EBML helpers + integrity checks.** `video_detector/dimension_checks/_common.py` (box walker, moov parser with stts/stsz/stss/stco/mdhd/tkhd/hdlr/stsd/fiel, text-item extraction, EBML info, SEI string finder) and `integrity.py`. Tests: `video_detector/tests/video_fixtures.py`, `test_dimension_checks_integrity.py`.
2. **Container checks.** `container.py` (`isobmff_boxes`, `isobmff_tables`, `mp4_metadata`, `telemetry_tracks`, `encoder_sei`, `ebml_info`). Tests: `test_dimension_checks_container.py`.
3. **Signal checks.** `signal.py` (`interlacing`, `temporal_cadence`) over `ctx.extra["frames"]`. Tests: `test_dimension_checks_signal.py`.
4. **Post checks + analysis helper.** `context.py`, `legal.py`, `lifecycle.py`, `reliability.py`, `fit_ood.py`, `__init__.py`. Tests: `test_dimension_checks_post.py`, `test_fit_ood.py`.
5. **Detector + pipeline + UI integration.** `VideoAIDetector.analyze_video(extra_log_lrs=...)`; `VideoForensicPipeline.analyze`; `ui/batch_ui.process_single_video`; `ui/feedback_ui.render_linear_video_pipeline_results`. Tests: `test_dimension_integration.py`; AppTest render.
6. **Docs + full verification.** Video report Appendix F + README + full `pytest`.
