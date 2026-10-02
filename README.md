# project-content-validation

A multi-modal (image / video / audio) content-authenticity forensics engine with a Streamlit UI. It inspects an uploaded or URL-fetched file and produces a probability-weighted authenticity verdict — real camera capture, conventionally edited, AI-enhanced/composite, fully AI-generated, or a screenshot (with its own authentic/AI-enhanced/AI-generated sub-states) — backed by a written evidence trail, a 9-dimension forensic dossier, and a generative-model attribution guess.

**Read this before trusting it in production:** the detection logic is a hand-tuned heuristic/statistical scoring engine, not a trained multi-class classifier. See [What this project actually is (and isn't)](#what-this-project-actually-is-and-isnt) before relying on its verdicts.

---

## Table of contents

- [Quick start](#quick-start)
- [Top-level repository tree](#top-level-repository-tree)
- [Architecture](#architecture)
- [How a request actually flows](#how-a-request-actually-flows)
- [`image_detector/` workspace](#image_detector-workspace)
- [`audio_detector/` workspace](#audio_detector-workspace)
- [`video_detector/` workspace](#video_detector-workspace)
- [`core/` — shared orchestration layer](#core--shared-orchestration-layer)
- [`ui/` — Streamlit presentation layer](#ui--streamlit-presentation-layer)
- [`app.py` — the Streamlit entry point](#apppy--the-streamlit-entry-point)
- [Testing](#testing)
- [Data, models, and what's actually on disk vs. git](#data-models-and-whats-actually-on-disk-vs-git)
- [What this project actually is (and isn't)](#what-this-project-actually-is-and-isnt)
- [`scripts/` (local, gitignored, not part of the product)](#scripts-local-gitignored-not-part-of-the-product)

---

## Quick start

```bash
pip install -r requirements.txt
```

**You also need `ffmpeg` installed separately and on `PATH`.** It is not a pip package. `audio_detector/validator.py` and `video_detector/extractor.py` both shell out to the `ffmpeg` CLI for format transcoding / audio-track extraction; if it's missing, both modules log a one-time warning and fall back to reduced functionality (audio falls back to Python's native `wave` module, which only handles WAV files; video's audio-track extraction is skipped). Get it from https://ffmpeg.org/download.html.

Run the app:

```bash
streamlit run app.py
```

This opens a 5-tab UI: Image Validation, Video Validation, Audio Validation, Social URL Checker, and Continuous Learning & Memory. Each media tab accepts either direct file upload or a URL (fetched through an SSRF-hardened downloader), analyzes one or many files, and renders a full forensic report per file plus a batch comparison table when multiple files are analyzed together.

Run the test suite:

```bash
pytest
```

(`pytest.ini` points at `core/tests`, `image_detector/tests`, `audio_detector/tests`, `video_detector/tests`, and `tests/`. The root-level `test_pipeline.py` is **not** collected by this configuration — run it directly with `python test_pipeline.py` if needed.)

---

## Top-level repository tree

```
project-content-validation/
├── app.py                      # Streamlit entry point — the only thing you launch
├── core/                       # Shared, presentation-independent orchestration layer
│   ├── atomic_io.py            #   crash-safe JSON read/write with per-path locking
│   ├── decision.py             #   cross-modal Bayesian fusion + the one normalize_percentages()
│   ├── security.py             #   anti-SSRF / DNS-rebinding-safe URL fetching, decompression-bomb guard
│   ├── forensic_service.py     #   lazy singleton facade wrapping all three detector packages
│   └── tests/test_enterprise_hardening.py
├── image_detector/              # Full image forensic pipeline (see dedicated section below)
├── audio_detector/               # Full audio forensic pipeline (see dedicated section below)
├── video_detector/               # Full video forensic pipeline (see dedicated section below)
├── ui/                          # Streamlit rendering + per-modality orchestration glue
│   ├── batch_ui.py              #   process_single_image/video/audio + batch dashboard/table/selector
│   ├── feedback_ui.py           #   every "render_*" results page + the feedback/rating widgets
│   └── validators.py            #   file/C2PA/EXIF validation, URL platform detection, media fetch
├── data/session_cache/           # App-owned scratch space for uploaded-file copies (created at runtime)
├── tests/test_unified_suite.py   # Cross-package integration tests (core.decision, SSRF delegation, etc.)
├── test_pipeline.py              # Root-level smoke test (NOT in pytest.ini's testpaths)
├── requirements.txt
├── pytest.ini
├── .gitignore
├── GLOBAL_IMAGE_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md   # Governing taxonomy/physics research doc
├── image_detector_update_implementation_plan.md             # File-by-file upgrade plan (gitignored)
└── scripts/                      # Personal local dev/eval scripts (gitignored, not part of the product)
```

Each of `image_detector/`, `audio_detector/`, `video_detector/` follows the same internal shape (file names are intentionally mirrored across all three so the same mental model applies everywhere — see the per-workspace sections for the real behavioral differences hiding behind that symmetry):

```
<modality>_detector/
├── __init__.py          # package facade — re-exports the public API
├── config.py            # paths, thresholds, weight defaults — zero outside deps
├── schemas.py           # dataclasses for every result/feedback/benchmark shape
├── validator.py         # file-integrity / format / size / decodability gate (runs first)
├── profiler.py          # low-level technical signal extraction (hashes, noise, geometry, ...)
├── provenance.py        # C2PA manifest scan + EXIF/container metadata signature detection
├── features.py / temporal.py + extractor.py (video only)   # forensic signal computation
├── detector.py          # the actual AI-vs-real scoring engine for this modality
├── content.py (+ face.py)   # scene/object/human/face intelligence
├── attribution.py        # "which specific generator made this" guesser
├── scoring.py            # evidence pooling + the taxonomy/decision logic
├── explain.py            # 9-dimension dossier + plain-English narrative generator
├── pipeline.py           # orchestrates all of the above into one analyze() call
├── batch.py              # process_files()/process_directory() over the pipeline
├── benchmarks.py         # accuracy/precision/recall/F1/ROC-AUC against a labeled dataset
├── downloader.py         # thin wrapper around core.security.SecureUrlFetcher
├── learner.py            # feedback-driven scalar-constant recalibration (NOT model training)
├── trainer.py            # the actual PyTorch training harness for the neural component
├── models/backbone.py    # the neural network architecture definition
├── data/                 # calibration.json + feedback/memory.json (gitignored)
├── dataset/              # train/val {ai_generated,real} folders for trainer.py / benchmarks.py
└── tests/test_*_detector.py
```

---

## Architecture

Four layers, strict dependency direction (lower layers never import upward):

1. **`core/`** — zero dependency on any detector package. Provides the shared SSRF-safe URL fetcher, atomic JSON persistence, and the single cross-modal decision-fusion function (`core.decision.generate_final_decision`) plus the one canonical `normalize_percentages()` that all three detector packages' `scoring.py` modules import rather than reimplementing.
2. **`image_detector/` / `audio_detector/` / `video_detector/`** — each a fully self-contained forensic pipeline for its modality. Each only imports from `core/` for three things: `core.security` (safe downloading), `core.atomic_io` (calibration persistence), and `core.decision.normalize_percentages` (shared math). They never import each other directly.
3. **`ui/`** — Streamlit rendering code plus the `process_single_image/video/audio` orchestration functions that call into a specific detector package's pipeline, then into `core.decision.generate_final_decision` to produce the final cross-modal verdict dict the UI actually renders.
4. **`app.py`** — the Streamlit launch target. Builds cached detector/service instances, wires the sidebar sensitivity selector, and lays out the 5 tabs, each of which calls down into `ui.batch_ui`.

`core/forensic_service.py`'s `ForensicService` is a separate, parallel orchestration facade (a lazy-loading singleton exposing `analyze_image/audio/video`, `record_feedback`, `health_check`) intended for headless/non-Streamlit use. `app.py` builds one via `get_forensic_service()` but, per the verified call graph, the Streamlit tabs actually route through `ui.batch_ui`'s `process_single_*` functions rather than through `ForensicService`'s own `analyze_*` methods — both paths reach the same underlying pipelines, just via two different facades that currently coexist rather than one superseding the other.

---

## How a request actually flows

For an image (video/audio follow the same shape — see their sections for the exact differences):

```
User uploads/URL-fetches a file in app.py
  → ui.validators.validate_file / fetch_media_from_url   (format check / SSRF-safe download)
  → ui.batch_ui.process_single_image(...)
      → ui.validators.validate_file               (again, inside the orchestration function)
      → ui.feedback_ui.profile_media(..., modality="image")   → image_detector.profiler.ImageProfiler
      → ui.validators.analyze_provenance                       → each package's own provenance scan
      → image_detector.detector.ImageAIDetector.predict(...)   ← the core scoring engine (see below)
      → image_detector.content.ImageContentAnalyzer.analyze_image_content(...)
      → image_detector.attribution.ImageModelAttributionEngine.attribute_media(...)
      → image_detector.validator.analyze_image(...)            (quality/validation result)
      → core.decision.generate_final_decision(...)             ← cross-modal fusion / final verdict
      → image_detector.explain.build_nine_dimensions_dossier(...)
      → image_detector.explain.generate_newbie_explanation(...)
  → ui.feedback_ui.render_linear_image_pipeline_results(...)    (renders everything above)
```

Inside `ImageAIDetector.predict()` itself (the single most important function in the repo — see the [image_detector section](#image_detector-workspace) for the exact ordered list of every signal it computes and every log-odds term it contributes), roughly: decode the image → extract ~14 forensic signals (sensor noise, surface smoothness, FFT spectral decay, ELA, EXIF/C2PA metadata, watermark, background cutout, scanned-print, face-swap, digital-art, screenshot, inpainting, screen-recapture-Moiré, spectral modality) → load the current feedback-calibrated weights → combine every signal into a Bayesian log-odds posterior → convert to AI%/Real%/Undecided% → run the result through a 7-branch taxonomy decision tree (`image_detector/scoring.py`) that assigns one of 10 defined taxonomy states.

---

## `image_detector/` workspace

**203 real sample images on disk** (`dataset/train/{real: 119, ai_generated: 41}`, `dataset/val/{real: 33, ai_generated: 10}`), and a trained checkpoint (`models/ai_detector.pt`, ~42.8 MB ResNet18) actually exists and loads at runtime — this is the only one of the three modalities where the neural-network code path is live rather than dormant.

### File-by-file

| File | Role |
|---|---|
| `config.py` | Paths, `IMAGE_SIZE=224`, `MIN_RESOLUTION=64`, noise/smoothness Gaussian baselines (`REAL_NOISE_MU/SIGMA`, `AI_NOISE_MU/SIGMA`, etc.), `CANONICAL_RESOLUTIONS` (known AI-generator output sizes), `CANONICAL_SCREEN_RESOLUTIONS` (real device-screen table for screenshot detection), `KNOWN_AI_SOFTWARE_SIGNATURES` (single source of truth, imported by `features.py` and `provenance.py`), `SENSITIVITY_PRIORS` (`balanced: 0.00`, `high: 0.20`, `aggressive: 0.40`). |
| `schemas.py` | `ImageTaxonomyState` — the 10 taxonomy-state constants (`AUTHENTIC_REAL_PHOTOGRAPH`, `AUTHENTIC_EDITED`, `AUTHENTIC_RECAPTURED_SCREEN`, `AI_ENHANCED_COMPOSITE`, `FULLY_AI_GENERATED`, `PROCEDURAL_CGI_SYNTHETIC`, `ADVERSARIAL_SPOOF_SYNTHETIC`, `AUTHENTIC_SCREENSHOT`, `AI_ENHANCED_SCREENSHOT`, `AI_GENERATED_SCREENSHOT`) plus `ImageForensicResult`, `ImageValidationResult`, `ImageFeedbackRecord`, `ImageBenchmarkMetrics` dataclasses. `ADVERSARIAL_SPOOF_SYNTHETIC` has a label/description defined here but no branch in `scoring.py` ever returns it — it's an unreachable taxonomy state in the current decision tree. |
| `validator.py` | `ImageValidator.validate()` — file exists → extension supported → size ≤ 100MB → Pillow can open it → dimensions ≥ 64px → OpenCV can decode it → blur/exposure/contrast quality scoring. Runs first in the pipeline; any failure short-circuits everything downstream. |
| `profiler.py` | `ImageProfiler.profile_image()` — hashes, DPI/ICC/EXIF (including GPS, exposure, lens), geometry, per-channel stats, Shannon entropy, dominant-color palette, and its own independent PRNU/smoothness/FFT-decay computation (duplicated from, not shared with, `features.py`'s versions). Runs before any AI prediction, as "Stage 1." |
| `provenance.py` | C2PA JUMBF byte-signature scan (head + tail, 512KB each) plus EXIF/XMP signature classification, producing one of 7 `provenance_status` values. Its output is passed into `detector.predict()` so the C2PA/camera-hardware signal actually reaches the Bayesian posterior, not just the UI evidence trail. |
| `features.py` (1119 lines — the largest module) | Every pixel-level forensic function: `calculate_sensor_noise_profile`, `calculate_surface_smoothness`, `analyze_fft_radial_power_spectrum`, `compute_ela`, `detect_ai_watermark` (Gemini-sparkle contour geometry), `detect_background_cutout`, `detect_scanned_photo`, `detect_face_swap_artifacts` (filename-signature only), `detect_digital_art_and_painting`, `detect_screenshot` (multi-signal: resolution table, filename, software tag, aspect-ratio device-type inference, UI-structure edge-density analysis), `detect_inpainting_and_manipulation`, `detect_screen_rephotography_moire`, `detect_spectral_modality`, `generate_spatial_manipulation_heatmap`, `extract_image_metadata`. |
| `detector.py` | `ImageAIDetector` — the scoring engine. `predict()` runs every `features.py` function, pulls the current feedback-calibrated weights from `learner.py`, assembles a dict of log-likelihood-ratio terms (camera hardware, C2PA, watermark, scanned-print, face-swap, inpainting, digital-art, PRNU noise, surface smoothness, FFT decay, canonical dimensions, and — if the checkpoint loaded — the neural backbone's own log-odds), pools them via `scoring.pool_bayesian_log_odds`, converts to percentages via `core.decision.normalize_percentages`, and classifies the result via `scoring.evaluate_taxonomy_classification`. |
| `content.py` / `face.py` | `ImageContentAnalyzer` (SSDLite-MobileNetV3 object detection, lighting/tone/environment heuristics, text-region detection, genre inference) and `FaceDeepfakeDetector` (YCrCb skin-chrominance face localization + bilateral-filter texture/noise deepfake-risk scoring — no trained face-detection model is used). |
| `attribution.py` | `ImageModelAttributionEngine` — scores an image against 21 known generator profiles (Midjourney, DALL-E 3/GPT Image 1, Flux.1, Google Imagen/Gemini/Nano Banana, Stable Diffusion, Adobe Firefly, Topaz Photo AI, Ideogram, Recraft, Magnific, Canva, neural face-swap pipelines, Leonardo.Ai, Grok Imagine, ByteDance Seedream, Tencent Hunyuan Image, Alibaba Qwen-Image, Kuaishou Kolors, Remini) via watermark/metadata/filename/canonical-resolution/spectral-slope signals, returning the best match with a region-of-origin guess and up to 3 alternate candidates. |
| `scoring.py` | `pool_bayesian_log_odds` (base-10 additive log-likelihood-ratio Bayesian pooling, with a correlation discount when both PRNU-noise and surface-smoothness fire together) and `evaluate_taxonomy_classification` — the full 7-branch decision tree (screenshot categorization → fully-AI-explicit-signature → AI-enhanced/composite-explicit-signature → AI-enhanced/composite-score-fallback → high-confidence-synthesis-by-score → screen-recapture → authentic-edited → default-authentic-real). |
| `explain.py` | `build_nine_dimensions_dossier()` (hardware/provenance, photometric/geometry, PRNU noise, surface smoothness, FFT decay, genre/subject, visual medium, sensor spectrum, generative attribution) and `generate_newbie_explanation()` (a plain-English narrative branching on screenshot/synthetic/edited/authentic). |
| `pipeline.py` | `ImageForensicPipeline.analyze()` — chains validator → profiler → provenance → detector (receiving the provenance result) → content analyzer → attribution → dossier/narrative generation into one call, returning one large result dict. |
| `batch.py` | `ImageBatchProcessor.process_files()`/`process_directory()` — runs the pipeline over many files, tallying `ai_generated`/`composite`/`real`/`undecided`/`errors` counts. |
| `benchmarks.py` | `ImageBenchmarkSuite.evaluate_dataset()` — accuracy/precision/recall/F1/ROC-AUC/confusion-matrix against a labeled `ai_generated/`+`real/` folder; this is a **binary** evaluation only (it cannot measure accuracy on the other 8 taxonomy states). |
| `learner.py` | `ImageSelfImprover` — persists feedback records and nudges 5 scalar feature-weights + sensitivity offsets by small fixed deltas per feedback event. **Does not retrain or touch the neural network's weights.** |
| `trainer.py` | `ImageDetectorTrainer` — the actual PyTorch training harness (ResNet18/50/MobileNetV3, `CrossEntropyLoss` + `AdamW`) that produces `models/ai_detector.pt`. Binary classifier only: `{ai_generated: 0, real: 1}`. |
| `downloader.py` | `ImageDownloader` — thin wrapper delegating to `core.security.SecureUrlFetcher`. |

### What it can and can't actually distinguish

The CNN checkpoint is a **binary** real-vs-ai_generated classifier (203 training images, both classes). Everything beyond that binary signal — edited/graphic-design, AI-enhanced/composite, every screenshot sub-state — is produced entirely by the hand-written decision tree in `scoring.py` operating on heuristic pixel/metadata signals, not by anything the network was trained to recognize. There is currently no training or test data in this repo for edited/composite/screenshot categories; `tests/test_image_detector.py`'s `TestImageDetectorRealSamples` class (added as part of this project's own code-review remediation) runs real end-to-end regression tests against the real/ai_generated dataset but explicitly documents that gap rather than papering over it.

---

## `audio_detector/` workspace

**No training data and no trained checkpoint exist for this modality** (`dataset/` is present but empty; `models/` holds only the architecture file). The neural-inference branch in `detector.py` is therefore dead code in this repo's current state — every analysis runs in pure statistical/heuristic mode.

### File-by-file

| File | Role |
|---|---|
| `config.py` | `TARGET_SAMPLE_RATE=16000`, `MAX_FILE_SIZE_MB=200`, `MAX_DURATION_SECONDS=3600`, vocoder cutoff bands (6500–8200 Hz / 15000–16500 Hz), flatness/silence thresholds, evidence weights (vocoder 0.40 / flatness 0.30 / silence 0.20 / HF-ratio 0.10). |
| `schemas.py` | `AudioForensicResult`, `AudioModalityScore`, `AudioTemporalSegment`, `AudioValidationResult`, `AudioFeedbackRecord`, `AudioBenchmarkMetrics`. No enum class for taxonomy — labels are plain strings (`"LIKELY AI-GENERATED"`, `"LIKELY REAL"`, `"UNDECIDED"`, `"NO_AUDIO"`, `"ERROR"`) since this modality's taxonomy is far simpler than image's. |
| `validator.py` | `AudioValidator.extract_pcm_samples()` — tries `ffmpeg` subprocess transcoding to 16kHz mono WAV first; if `ffmpeg` is missing, falls back to Python's native `wave` module (WAV only), with manual stereo→mono averaging and resampling. `validate()` chains existence → extension → size → decode → duration → clipping/silence checks. |
| `profiler.py` | `AudioProfiler.profile_audio()` — hashes, peak/RMS/crest-factor/dynamic-range, clipping/silence flags. |
| `provenance.py` | C2PA byte-signature scan (first 512KB) plus a raw first-4096-byte scan for known encoder-name substrings (`lame`, `ffmpeg`, `elevenlabs`, etc.). |
| `features.py` | `compute_spectral_features()` — manual-framing STFT, vocoder-cutoff frequency detection (98.5% cumulative energy point), Wiener spectral flatness, digital-silence ratio, high-frequency energy ratio; combines these into a standalone heuristic `p_audio_ai` score used independently by `segment_audio_temporal()`'s per-window timeline labels (this per-segment scorer does **not** consult the calibration weights that `scoring.pool_acoustic_evidence` does — see the Known Limitations note below). `generate_spectrogram_image()` renders a viridis-colormapped STFT image. |
| `detector.py` | `AudioAIDetector.analyze_audio_file()` (aliased `predict`/`predict_audio`) — decode PCM → load calibration → compute spectral features + temporal segments → (if a checkpoint existed) neural inference → `pool_acoustic_evidence()` → epistemic uncertainty → `normalize_percentages()` → `evaluate_audio_decision()` → assemble forensic cues + result. |
| `content.py` | `AudioContentAnalyzer.analyze_audio_scene()` — RMS/crest-factor-based delivery-style classification (Dynamic/Compressed/Natural), speech-vs-music-vs-ambient modality/setting inference from speech-band vs high-band energy ratios. |
| `attribution.py` | `AudioModelAttributionEngine` — scores against 13 known voice/music generators (ElevenLabs, CosyVoice, ChatTTS, Suno, Udio, OpenAI Voice/Realtime API, Gemini/Lyria, Meta AudioCraft, Hume AI, Cartesia, PlayHT, Resemble AI, Coqui XTTS-v2). Only the first 6 have real spectral/vocoder-cutoff calibration (documented explicitly in the module docstring); the rest are metadata-signature-match only. |
| `scoring.py` | `pool_acoustic_evidence()` — a **linear weighted average** of four heuristic 0/1-ish scores (vocoder/flatness/silence/HF-roll), blended 55/45 with the neural probability when available; explicitly documented as not Bayesian and not comparable to `image_detector`'s or `video_detector`'s own pooling math. `evaluate_audio_decision()` — simple threshold cutoff (50.0 high/aggressive, 55.0 balanced) against `REAL_THRESHOLD=55.0`. |
| `explain.py` | `build_audio_nine_dimensions_dossier()` / `generate_audio_newbie_explanation()` — same 2-function pattern as image's `explain.py`. |
| `pipeline.py` | `AudioForensicPipeline.analyze()` — validator → profiler → provenance → detector (reusing already-extracted PCM samples, not re-decoding) → content analyzer → attribution → dossier/narrative. |
| `batch.py` / `benchmarks.py` / `learner.py` / `trainer.py` / `downloader.py` | Same role as their `image_detector` counterparts, scaled to audio. `learner.py`'s module docstring explicitly states it does not retrain anything and no checkpoint exists in this repo. |

---

## `video_detector/` workspace

Also **no trained checkpoint and an empty `dataset/`**, same as audio — the neural temporal-transition model never loads in this repo's current state.

### File-by-file

| File | Role |
|---|---|
| `config.py` | `DEFAULT_MAX_FRAMES=30`, `MAX_FILE_SIZE_MB=500`, motion-variance thresholds (`MOTION_VAR_HIGH_WARPING=140.0`, `MOTION_VAR_SUSPICIOUS_FLICKER=75.0`, `MOTION_VAR_UNNATURAL_FREEZE=0.8`), flicker thresholds, temporal pooling weights. |
| `schemas.py` | `VideoForensicResult`, `VideoModalityScore`, `VideoTemporalSegment` (frame-count granularity, not per-window scored like audio's), `VideoValidationResult`, `VideoFeedbackRecord`, `VideoBenchmarkMetrics`. Labels are plain strings including warping-risk states (`HIGH_WARPING_DETECTED`/`SUSPICIOUS_FLICKER`/`UNNATURAL_FREEZE`/`LOW`) and face-risk states. |
| `validator.py` | `VideoValidator.validate()` — existence/extension/size/metadata-readable/duration, then a stream-readability check reading a first and middle frame via OpenCV. |
| `extractor.py` | `VideoFrameExtractor` — `get_video_metadata()`, `extract_sampled_frames()` (uniform sampling via frame-index seeking, falling back to sequential read if the container doesn't report frame count), `extract_keyframe()` (defaults to 15% into the timeline, to skip black title frames), `extract_audio_track()` (ffmpeg subprocess demux to WAV; warns once and continues if ffmpeg is missing). |
| `profiler.py` | `VideoProfiler.profile_video()` — hashes, fps/frame-count/resolution/codec via OpenCV. |
| `provenance.py` | C2PA scan (first 1MB + last 128KB) plus generic MP4 atom detection (`KNOWN_VIDEO_ATOMS`) kept **separate** from `KNOWN_VIDEO_GENERATOR_SIGNATURES` vendor-name scanning (the module explicitly documents that atom type-names like `ftyp`/`moov` can never carry vendor identity — `attribution.py` reads `vendor_signatures_found`, not `container_atoms`). |
| `temporal.py` | `compute_interframe_motion_variance()` — classifies `HIGH_WARPING_DETECTED` / `SUSPICIOUS_FLICKER` / `UNNATURAL_FREEZE` / `LOW` from inter-frame grayscale-difference variance, using calibration-supplied thresholds when available. `detect_diffusion_flickering()` — luminance sign-change-ratio based flicker scoring. `group_temporal_segments()` — merges consecutive same-label frames into timeline segments. |
| `face.py` | `VideoFaceDeepfakeDetector` — same YCrCb skin-segmentation + bilateral-texture/noise deepfake-risk approach as `image_detector/face.py`, extended with a `person_boxes`-gated mode and a whole-video aggregation method (`analyze_video_frames`). |
| `content.py` | `VideoContentAnalyzer.analyze_video_frames()` — delegates face/human counting to `VideoFaceDeepfakeDetector`, picks the middle sampled frame for lighting/environment classification. |
| `cross_modal.py` | `CrossModalConsistencyEngine.evaluate_consistency()` — compares a video-forensics result against an **externally supplied** audio-forensics result (video_detector never calls audio_detector itself; some caller outside both packages is expected to run the audio pipeline separately and pass its result in). Flags modality asymmetry (≥40-point AI% gap between video and audio), scene-plausibility mismatches (outdoor scene + suspiciously silent audio), and classifies `CROSS_MODAL_COHERENT`/`SUSPICIOUS_ASYMMETRY`/`CROSS_MODAL_INCONSISTENT`. |
| `detector.py` | `VideoAIDetector.analyze_video()` (aliased `predict`/`predict_video`) — load → extract metadata → sample frames → load calibration → `compute_interframe_motion_variance` → `detect_diffusion_flickering` → per-frame scoring (via an optional pluggable external `frame_detector`, or an internal noise/smoothness heuristic fallback) → optional neural transition inference (dead without a checkpoint) → cache keyframes for the pipeline's content/face analysis (avoiding a second video decode) → `pool_video_temporal_score` → `normalize_percentages` → `evaluate_video_decision`. |
| `attribution.py` | `VideoModelAttributionEngine` — scores against 15 known video generators (ByteDance Seedance, Kuaishou Kling, OpenAI Sora, Runway Gen-2/3, Google Veo, Luma Dream Machine, Pika, MiniMax Hailuo, Alibaba Wan, Tencent Hunyuan Video, Vidu, Pixverse, Meta Movie Gen, Adobe Firefly Video, Amazon Nova Reel) via vendor-metadata-signature matching, C2PA presence, and motion/flicker heuristics. Same "first 8 calibrated, rest metadata-only" caveat as audio's attribution list. |
| `scoring.py` | `pool_video_temporal_score()` — linear weighted average (frame-AI-ratio / motion-warping / diffusion-flicker / optional neural-temporal), explicitly documented as intentionally distinct math from both `image_detector`'s Bayesian pooling and `audio_detector`'s own pooling. `evaluate_video_decision()` — same threshold pattern as audio's. |
| `explain.py` | Same 2-function dossier/narrative pattern. Two of its nine dimension fields (`mean_frame_noise`, `flicker_variance`) read dict keys that `detector.py`/`temporal.py` never actually populate under those exact names, so those two dossier fields always render their hardcoded fallback values in practice — a known, documented gap, not a crash. |
| `pipeline.py` | `VideoForensicPipeline.analyze()` — validator → profiler → provenance → detector (reuses cached keyframes rather than re-decoding) → content analyzer → attribution → cross-modal consistency (consuming an optional externally-supplied `audio_forensics` argument) → dossier/narrative. |
| `batch.py` / `benchmarks.py` / `learner.py` / `trainer.py` / `downloader.py` | Same role pattern as the other two packages. `benchmarks.py` computes F1 via `2·tp/(2·tp+fp+fn)` rather than from separately-computed precision/recall (equivalent result, different code path than audio's/image's benchmarking). |

---

## `core/` — shared orchestration layer

| File | Role |
|---|---|
| `security.py` | `validate_secure_url()` — rejects non-http(s) schemes, rejects localhost/metadata-service hostnames by name, resolves every A/AAAA record and rejects if *any* resolved IP falls in a private/loopback/link-local/cloud-metadata/multicast/reserved range (19 hardcoded CIDR blocks, IPv4 + IPv6). `_PinnedResolver` closes the DNS-rebinding TOCTOU gap by pinning `socket.getaddrinfo` to only the already-validated IPs for the duration of one request. `SecureUrlFetcher` — streams the download with a byte-count ceiling, rejects `text/html` responses, manually follows up to 3 redirects while re-validating (and re-pinning) every hop. `sanitize_filename()`, `generate_secure_cache_name()`. Also sets `PIL.Image.MAX_IMAGE_PIXELS` at import time as a decompression-bomb guard — importing this module anywhere in the process changes that global Pillow setting. |
| `decision.py` | `normalize_percentages()` — the single canonical implementation all three detector packages' `scoring.py` modules import. Two modes: scale-all-three-together when an `undecided_val` is supplied (what every per-modality detector uses, calibrated against this exact algorithm — don't change it without re-validating every taxonomy threshold), or derive `undecided` from the AI/Real score gap when it's omitted (used internally below). `generate_final_decision()` — the cross-modal Bayesian fusion function: pools image/video/audio AI-probabilities plus provenance/cross-modal/attribution evidence as weighted log-odds terms, derives a `final_status` and (if the per-modality result didn't already carry one) a fallback taxonomy state, and sanitizes attribution for authentic verdicts (forces "None (Authentic Capture)" so an authentic photo never shows a spurious generator attribution). |
| `atomic_io.py` | `atomic_write_json()`/`atomic_read_json()`/`atomic_update_json()` — write-to-temp-then-`os.replace` atomic persistence with per-resolved-path `RLock`s and Windows-specific retry-on-`PermissionError` handling. Used by all three packages' `learner.py` for calibration/feedback persistence. |
| `forensic_service.py` | `ForensicService` — a thread-safe lazy-loading singleton wrapping every detector/pipeline/improver object behind `analyze_image/audio/video()`, `record_feedback()`, and `health_check()`. Note: `health_check()` always reports `HEALTHY` — it is not a live liveness probe, just a static capability-description dict. |

---

## `ui/` — Streamlit presentation layer

| File | Role |
|---|---|
| `validators.py` | `validate_file()` (dispatches to each package's own validator), `scan_c2pa_markers()` + `extract_exif_metadata()` → `analyze_provenance()` (a UI-layer provenance check, distinct from each detector package's own `provenance.py` — both exist and are both exercised), `validate_expected_platform()` / `detect_platform()` (Instagram/YouTube/Facebook/TikTok/X domain matching), `fetch_media_from_url()` (wraps `core.security.SecureUrlFetcher`). |
| `batch_ui.py` | `process_single_image/video/audio()` — the actual per-file orchestration functions that chain each detector package's validator → profiler/provenance → detector → content analyzer → attribution → `core.decision.generate_final_decision()` → dossier/narrative. `run_batch_pipeline()` dispatches to these by modality string. `render_batch_summary_dashboard/overview_table/file_selector()` — the multi-file comparison UI with CSV/JSON export. |
| `feedback_ui.py` (1711 lines, the largest UI file) | `profile_media()`, the full stack of `render_*` functions for every pipeline stage (pre-analysis specs, 9-dimension breakdown, type/category, quantified inventory, scene intelligence, newbie narrative), the rating/feedback widgets that call each package's `*SelfImprover.record_feedback()`, `render_learning_dashboard()` (shows current calibration state across all three modalities), and the three top-level `render_linear_image/video/audio_pipeline_results()` functions that `app.py` calls once per analyzed file. |

---

## `app.py` — the Streamlit entry point

Sets `Image.MAX_IMAGE_PIXELS` and `ImageFile.LOAD_TRUNCATED_IMAGES=False` at import time (decompression-bomb hardening before any UI code runs). Builds a `ForensicService` and five `@st.cache_resource`-wrapped detector/content-analyzer/attribution-engine accessors. Renders a sidebar sensitivity selector (`balanced` default / `high` / `aggressive` — see the note on sensitivity priors below) and 5 tabs:

1. **Image Validation** / 2. **Video Validation** / 3. **Audio Validation** — identical shape: upload-or-URL → session-state-cached batch run via `ui.batch_ui.run_batch_pipeline` → (if multiple files) summary dashboard + overview table + file selector, otherwise the single result directly → the selected result rendered via the matching `ui.feedback_ui.render_linear_*_pipeline_results`.
4. **Social URL Checker** — standalone; just calls `ui.validators.validate_expected_platform()` and dumps the JSON. Touches no detector package.
5. **Continuous Learning & Memory** — a single call to `ui.feedback_ui.render_learning_dashboard()`, showing the current calibration state for all three modalities.

---

## Testing

```
pytest.ini testpaths:
  core/tests/              — security/atomic-IO/forensic-service-facade hardening tests
  image_detector/tests/    — unit tests + TestImageDetectorRealSamples (real-sample end-to-end regression)
  audio_detector/tests/    — unit tests against one synthetic 440Hz sine-wave WAV
  video_detector/tests/    — unit tests against one synthetic 15-frame random-noise MP4
  tests/                   — cross-package integration (core.decision fusion, SSRF delegation
                              across all three downloaders, atomic persistence, forensic-service facade)
```

**Honest coverage gap:** only `image_detector` has real-sample end-to-end tests (against the checked-in `dataset/` images) — `audio_detector` and `video_detector` have no real sample media checked into this repo, so their test suites exercise the code paths with synthetic sine-wave/noise fixtures rather than validating actual detection accuracy against real recordings. This is a known, documented limitation, not an oversight hidden from users of this README.

`test_pipeline.py` at the repository root is a smoke test but is **not** listed in `pytest.ini`'s `testpaths`, so a plain `pytest` invocation will not run it — invoke it directly with `python test_pipeline.py` if you need it.

---

## Data, models, and what's actually on disk vs. git

- `image_detector/models/ai_detector.pt` (~42.8MB trained ResNet18 checkpoint) exists on disk and is loaded by `ImageAIDetector.load()`, but `.gitignore` excludes `*.pt` files — it is **not tracked in git**. If you clone this repo fresh, this file will be missing and the image detector will silently fall back to pure heuristic/statistical mode (the `load()` method catches the exception and logs a warning rather than failing).
- `audio_detector/models/` and `video_detector/models/` contain only `backbone.py`/`__init__.py` — no checkpoint exists for either modality in this repo at all, trained or otherwise.
- `image_detector/dataset/{train,val}/{real,ai_generated}/` contains 203 real images used by `trainer.py` and `benchmarks.py`. `audio_detector/dataset/` and `video_detector/dataset/` exist as empty directories (created lazily by each package's `ensure_directories()`) — nothing populates them in this repo.
- Each package's `data/` directory holds its own `*_calibration.json` (feedback-adjusted scoring weights) and `*_feedback.json`/`*_memory.json` (raw feedback records) — all gitignored.
- `data/session_cache/` at the repository root is `app.py`'s own scratch space for uploaded-file copies across all three modalities — deliberately kept separate from each detector package's own `data/` directory so the orchestration layer doesn't write into a package's persistent calibration store.
- `.gitignore` also excludes every image/video/audio file extension project-wide (explicitly commented "prevent personal photo/video/audio uploads"), and the entire `scripts/` directory plus `image_detector_update_implementation_plan.md` as personal/local-only files.

---

## What this project actually is (and isn't)

Read this before trusting a verdict from this tool:

- **The AI-vs-real distinction is a hand-tuned statistical/heuristic scoring system**, not a trained multi-class model, for everything except image's binary real/ai_generated backbone. Every other taxonomy distinction (edited, composite, screenshot sub-states, generator attribution) is produced by threshold logic over pixel/metadata signals written by hand — not learned from labeled examples of those categories, because no such labeled data exists anywhere in this repo.
- **"Self-improving" / "feedback-driven learning" does not mean model training.** Every package's `learner.py` module docstring says this explicitly: `record_feedback()` only nudges a handful of scalar scoring-weight constants by small fixed deltas per feedback event, persisted to a JSON calibration file. It never touches a neural network's weights and never retrains anything. The actual PyTorch training code lives separately in each package's `trainer.py` and must be run manually.
- **Only `image_detector` has a trained neural checkpoint on disk in this repo, and it's a binary classifier** (real vs. ai_generated, 203 training images). `audio_detector` and `video_detector` have no checkpoint at all — every "neural inference" code path in their `detector.py` files is unreachable until someone runs `trainer.py` against real labeled data and the resulting checkpoint is placed under `models/`.
- **Generator-attribution lists are honestly marked for calibration confidence.** Each package's `attribution.py` module docstring states exactly which generator entries have real spectral/vendor-signature calibration versus which are metadata-signature-match-only placeholders for 2025/2026-era tools (added without real sample data to calibrate against).
- **Sensitivity defaults to `balanced`** (a neutral 50/50 prior before any evidence is evaluated) rather than `high`, specifically because a `high`/`aggressive` prior biases ambiguous low-signal real images toward a false "AI" verdict by design. You can still choose `high`/`aggressive` explicitly in the sidebar if you want more aggressive scrutiny at the cost of more false positives.
- **`video_detector`'s cross-modal consistency check requires an externally-supplied audio result** — `video_detector` never calls `audio_detector` itself; whatever caller wants cross-modal checking has to run both pipelines and pass the audio result into `VideoForensicPipeline.analyze(audio_forensics=...)`.

---

## `scripts/` (local, gitignored, not part of the product)

A directory of personal diagnostic/evaluation scripts the project author used during development — hardcoded personal file paths, not part of the shipped application, and excluded from git via `.gitignore`. They exist on the local disk this README was written from but will not be present in a fresh clone. They include dataset-folder accuracy spot-checkers (`analyze_android_dataset.py`, `compare_profiles.py`, `diagnose_camera_pic.py`, `test_all_domains.py`, `test_android_photos.py`, `test_sony_camera.py`, `evaluate_family_images.py`, `evaluate_wedding_ai.py`) and manual training-data ingestion/fine-tuning scripts (`train_and_learn_wedding.py`, `train_on_android_dataset.py`, `train_on_camera_dataset.py`). All of them import directly from `image_detector` and `ui.batch_ui`, exercising the same production code paths as `app.py` itself, just headlessly against personal local datasets outside the repository.
