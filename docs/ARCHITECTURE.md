# Architecture and reference

How the engine is organised, how one request flows through it, and what each module does. For installation and daily use see the [README](../README.md) and the [user guide](USER_GUIDE.md).

## Layers

Dependencies point downwards only; lower layers never import upwards.

1. **`core/`**: shared, detector-independent code: SSRF-safe fetching (`security.py`), crash-safe JSON storage (`atomic_io.py`), the decision fusion (`decision.py`), probability bands, the dimension-check foundation (`forensics/`), the media library and retraining engine.
2. **`image_detector/`, `audio_detector/`, `video_detector/`**: one self-contained forensic pipeline per modality. They import from `core/` only and never from each other.
3. **`services/`**: `ForensicService` (headless facade over all three pipelines), `calibration_cli` (measure real accuracy) and `system_status` (what the sidebar shows).
4. **`ui/` and `app.py`**: the Streamlit presentation layer.

```
app.py                 page layout: sidebar + Image / Video / Audio / Link check / Learning tabs
ui/shell.py            sidebar (sensitivity, system status, clear files)
ui/media_tab.py        one implementation of "upload or link -> analyse -> show result", driven by a MediaTabSpec
ui/adapters.py         process_single_image/video/audio + run_batch_pipeline (thin wrappers over the pipelines)
ui/batch_views.py      comparison table and file picker for multi-file runs
ui/results/            result pages: summary (verdict card), checks (evidence), media previews, explanation,
                       feedback, learning dashboard, details sections, pages.py (composes them)
ui/stages.py           modality-agnostic finding/band/gate components
ui/validators.py       file validation, provenance dispatch, URL platform detection, media fetch
```

## How one request flows

```
upload / link
  -> ui.validators (format check, SSRF-safe download)
  -> ui.adapters.process_single_*            thin adapter
       -> <modality>_detector.pipeline.*ForensicPipeline.run()      the single analysis sequence:
            gates -> profile -> provenance -> dimension checks -> detector -> content -> attribution
            -> post checks -> core.decision.generate_final_decision -> dossier -> narrative
  -> ui.results.pages.render_*_result        verdict card + Overview | Evidence | Details | Feedback
```

`pipeline.analyze()` turns the same `run()` output into the headless report used by `ForensicService`; `tests/test_ui_flow_apptest.py` asserts both paths give the same verdict.

Inside `ImageAIDetector.predict()` the image is decoded, about 14 forensic signals are computed (sensor noise, surface smoothness, FFT decay, ELA, EXIF/C2PA, watermark, cut-out background, scanned print, face swap, digital art, screenshot, inpainting, screen recapture, spectral modality), combined as a Bayesian log-odds posterior, converted to AI / Real / Undecided percentages and passed through a seven-branch taxonomy decision tree (`image_detector/scoring.py`) that assigns one of ten states.

## Dimension checks

The taxonomy reports (`GLOBAL_*_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md`) describe dimensions beyond pixel/waveform physics: file security, metadata and container forensics, legal flags, context re-use, lifecycle laundering, detector reliability, probability bands, out-of-distribution (OOD) detection and open-set attribution. A modality-agnostic foundation in `core/` implements them, and the **image**, **audio** and **video** pipelines are all wired to it (see each report's *Implementation Status* appendix for exactly what is implemented, partial, recognition-only or spec-only).

```
core/forensics/   schemas.py (Finding/DimensionReport) · registry.py (isolated checks + hybrid caps)
                  gates.py (hard-block hash list, scientific-format recognition) · ood.py (Mahalanobis OOD gate)
core/bands.py     the five calibrated probability bands
image_detector/dimension_checks/   integrity · formats · metadata · context · legal · lifecycle · reliability
audio_detector/dimension_checks/   integrity · container · signal · context · legal · lifecycle · reliability
core/forensics/bytescan.py         shared bounded byte scans (signatures, prompt-injection text)
video_detector/dimension_checks/   integrity · container · signal · context · legal · lifecycle · reliability
ui/stages.py      modality-agnostic Streamlit stage components (shared by image, audio and video)
```

**What affects the verdict.** Only metadata/physical-consistency findings (`PHYSICAL_SIGNAL`, `METADATA_WEAK`) may add log-odds, as small capped terms (per finding +/-0.25, explicit generator-parameter PNG chunk 0.40, total +/-0.40, base-10), passed to `ImageAIDetector.predict(extra_log_lrs=...)`. Absence of metadata is never scored. Security, legal, context, lifecycle and reliability findings, bands, OOD status and `UNKNOWN_SOURCE` are advisory and never change P(AI). A failing check becomes an `ERROR` finding and never breaks the pipeline.

**Operator-supplied local files** (all gitignored; nothing is bundled):

| Purpose | How to enable |
|---|---|
| Hard-block gate (SHA-256 list of known-prohibited files; no classifier is built in) | one hex SHA-256 per line in `core/data/hardblock_sha256.txt`, or set `OMNI_HARDBLOCK_SHA256_FILE`. A match stops analysis with `HARD_BLOCK_ESCALATE`. |
| Context re-use index | JSON lines `{"phash": "<16 hex>", "label": "...", "source": "..."}` in `image_detector/data/context_hash_index.jsonl`, or set `OMNI_CONTEXT_HASH_INDEX`. |
| OOD gate | register labeled media in the library, then `python -m image_detector.dimension_checks.fit_ood` (writes `image_detector/data/ood_stats.npz`). Until fitted the gate reports `NOT_CALIBRATED` rather than inventing a threshold. |

**Audio** adds container, signal, context, legal, lifecycle and reliability checks (ID3 / RIFF-bext / FLAC MD5 / MP3-LAME / Ogg structure; ENF mains-hum trace and splice detection; fake hi-res and bit-depth padding; telephony band-limit; loudness). Audio pools by weighted average, so the capped terms (explicit generator string in tags +0.40, ENF continuous -0.15, ENF splice +0.15) shift the pooled probability in log-odds space; with no terms the result is unchanged. A missing ENF trace is never evidence of synthesis. Extra operator files: `OMNI_AUDIO_FP_INDEX` (JSON-lines acoustic fingerprint index; build entries with `audio_detector.dimension_checks.context.fingerprint_hex`) and `python -m audio_detector.dimension_checks.fit_ood`.

**Video** adds container, signal, context, legal, lifecycle and reliability checks (MP4/MOV box layout, sample-table consistency, creation times and tool strings, telemetry-track presence, encoder strings, interlace/combing and frame cadence). Video pools by weighted average, so the one scoring term (an explicit generative-tool name in container software/comment fields, +0.40) shifts the pooled probability in log-odds space; with no terms the result is unchanged. Interlace and cadence findings are informational. Extra operator files: `OMNI_VIDEO_FP_INDEX` (JSON-lines frame-hash index; build entries with `video_detector.dimension_checks.context.video_fingerprint_hashes`) and `python -m video_detector.dimension_checks.fit_ood`.

Recognized-but-unscored scientific formats (FITS, DICOM, GeoTIFF, OpenEXR, HDF5, NetCDF, AEDAT event files) return `RECOGNIZED_OUT_OF_SCOPE` instead of a real-vs-AI verdict.

### Verdict semantics and known limits

- **Probabilities are heuristic and uncalibrated.** Every dossier carries `calibration_status: "UNCALIBRATED_HEURISTIC"` and the UI says so. Thresholds (ENF jump, combing ratio, bits-per-pixel, band cutoffs) were tuned on synthetic fixtures only; no labeled dataset is checked in, so real accuracy / false-positive rates are **unmeasured**.
- **C2PA is presence-only.** `ui.validators.scan_c2pa_markers` and `image_detector.provenance` look for byte markers; no certificate chain or hash binding is validated. Presence is reported but never scored as protective evidence, and the wording is never "verified".
- **Camera EXIF is unauthenticated.** Coherent EXIF earns no credit in the dimension checks. In `ImageAIDetector.predict`, camera EXIF is *contradicted* (hardware credit shrinks from `EXIF_TRUSTED_CREDIT` to `EXIF_UNTRUSTED_CREDIT`, physical-signal discount withheld) when the raw noise+smoothness evidence leans synthetic by at least `EXIF_CONTRADICTION_LR`. A forger who also produces natural-looking pixel noise is still not detected.
- **`core.decision.generate_final_decision` has two explicit modes** (`decision_mode`): `image_authoritative` (a single image result with a taxonomy state is final) and `fused` (video / audio / cross-modal / declared-AI provenance are pooled). Video verdicts come from the video detector's own result (`video_result=`), not from a keyframe image. Attribution is explanation only and is never double-counted as evidence.
- **One orchestration path.** Each `*_detector.pipeline.*ForensicPipeline.run()` is the single analysis sequence (profile → provenance → dimension checks → detector → content → attribution → post checks → `generate_final_decision` → dossier → narrative) and returns every intermediate (`ImageRun` / `AudioRun` / `VideoRun`). `analyze()` turns that into the headless report; `ui.adapters.process_single_*` are thin adapters that add only UI concerns (gate short-circuit shape, display profile blocks, keyframe file). `tests/test_ui_flow_apptest.py` asserts both paths produce the same verdict.
- **Measure, don't trust.** `python -m services.calibration_cli --modality image|audio|video` scores the held-out validation split of your media library and reports accuracy, false-positive rate, Brier score, ECE and band occupancy (`core/calibration_report.py`). With no labeled library it says so and reports nothing.
- **Test isolation.** The root `conftest.py` fingerprints `*/models/*.pt`, `*/data/*.json|npz` and `core/data/*` before and after a run and fails the session if any test changed them.

## Image detector (`image_detector/`)

No dataset and no trained checkpoint ship (the project starts blank). The neural-network path in `detector.py` stays inactive until you train a checkpoint (see the [user guide](USER_GUIDE.md#train-on-your-own-media)); until then every analysis uses the statistical signals alone.

### File-by-file

| File | Role |
|---|---|
| `config.py` | Paths, `IMAGE_SIZE=224`, `MIN_RESOLUTION=64`, noise/smoothness Gaussian baselines (`REAL_NOISE_MU/SIGMA`, `AI_NOISE_MU/SIGMA`, etc.), `CANONICAL_RESOLUTIONS` (known AI-generator output sizes), `CANONICAL_SCREEN_RESOLUTIONS` (real device-screen table for screenshot detection), `KNOWN_AI_SOFTWARE_SIGNATURES` (single source of truth, imported by `features.py` and `provenance.py`), `SENSITIVITY_PRIORS` (`balanced: 0.00`, `high: 0.20`, `aggressive: 0.40`). |
| `schemas.py` | `ImageTaxonomyState` — the 10 taxonomy-state constants (`AUTHENTIC_REAL_PHOTOGRAPH`, `AUTHENTIC_EDITED`, `AUTHENTIC_RECAPTURED_SCREEN`, `AI_ENHANCED_COMPOSITE`, `FULLY_AI_GENERATED`, `PROCEDURAL_CGI_SYNTHETIC`, `ADVERSARIAL_SPOOF_SYNTHETIC`, `AUTHENTIC_SCREENSHOT`, `AI_ENHANCED_SCREENSHOT`, `AI_GENERATED_SCREENSHOT`) plus `ImageForensicResult`, `ImageValidationResult`, `ImageFeedbackRecord`, `ImageBenchmarkMetrics` dataclasses. `ADVERSARIAL_SPOOF_SYNTHETIC` has a label/description defined here but no branch in `scoring.py` ever returns it — it's an unreachable taxonomy state in the current decision tree. |
| `validator.py` | `ImageValidator.validate()` — file exists → extension supported → size ≤ 100MB → Pillow can open it → dimensions ≥ 64px → OpenCV can decode it → blur/exposure/contrast quality scoring. Runs first in the pipeline; any failure short-circuits everything downstream. |
| `profiler.py` | `ImageProfiler.profile_image()` — hashes, DPI/ICC/EXIF (including GPS, exposure, lens), geometry, per-channel stats, Shannon entropy, dominant-color palette, and its own independent PRNU/smoothness/FFT-decay computation (duplicated from, not shared with, `features.py`'s versions). Runs before any AI prediction. |
| `provenance.py` | C2PA JUMBF byte-signature scan (head + tail, 512KB each) plus EXIF/XMP signature classification, producing one of 7 `provenance_status` values. Its output is passed into `detector.predict()` so the C2PA/camera-hardware signal actually reaches the Bayesian posterior, not just the UI evidence trail. |
| `features.py` (1119 lines — the largest module) | Every pixel-level forensic function: `calculate_sensor_noise_profile`, `calculate_surface_smoothness`, `analyze_fft_radial_power_spectrum`, `compute_ela`, `detect_ai_watermark` (Gemini-sparkle contour geometry), `detect_background_cutout`, `detect_scanned_photo`, `detect_face_swap_artifacts` (filename-signature only), `detect_digital_art_and_painting`, `detect_screenshot` (multi-signal: resolution table, filename, software tag, aspect-ratio device-type inference, UI-structure edge-density analysis), `detect_inpainting_and_manipulation`, `detect_screen_rephotography_moire`, `detect_spectral_modality`, `generate_spatial_manipulation_heatmap`, `extract_image_metadata`. |
| `feature_store.py` | Optional rebuildable feature cache: `FeatureStore` (512-dim embedding + 12-dim forensic vector per image, keyed by content hash + backbone fingerprint, no file names, never deletes sources), `FeatureBankDataset`, and `FeatureClassifierHead`. |
| `detector.py` | `ImageAIDetector` — the scoring engine. `predict()` accepts file paths, raw bytes, BytesIO, or numpy arrays (pure in-memory ingestion). Runs every `features.py` function, pulls the current feedback-calibrated weights from `learner.py`, assembles a dict of log-likelihood-ratio terms (camera hardware, C2PA, watermark, scanned-print, face-swap, inpainting, digital-art, PRNU noise, surface smoothness, FFT decay, canonical dimensions, and — if the checkpoint loaded — the neural backbone's own log-odds), pools them via `scoring.pool_bayesian_log_odds`, converts to percentages via `core.decision.normalize_percentages`, and classifies the result via `scoring.evaluate_taxonomy_classification`. |
| `content.py` / `face.py` | `ImageContentAnalyzer` (SSDLite-MobileNetV3 object detection, lighting/tone/environment heuristics, text-region detection, genre inference) and `FaceDeepfakeDetector` (YCrCb skin-chrominance face localization + bilateral-filter texture/noise deepfake-risk scoring — no trained face-detection model is used). |
| `attribution.py` | `ImageModelAttributionEngine` — scores an image against 21 known generator profiles (Midjourney, DALL-E 3/GPT Image 1, Flux.1, Google Imagen/Gemini/Nano Banana, Stable Diffusion, Adobe Firefly, Topaz Photo AI, Ideogram, Recraft, Magnific, Canva, neural face-swap pipelines, Leonardo.Ai, Grok Imagine, ByteDance Seedream, Tencent Hunyuan Image, Alibaba Qwen-Image, Kuaishou Kolors, Remini) via watermark/metadata/filename/canonical-resolution/spectral-slope signals, returning the best match with a region-of-origin guess and up to 3 alternate candidates. |
| `scoring.py` | `pool_bayesian_log_odds` (base-10 additive log-likelihood-ratio Bayesian pooling, with a correlation discount when both PRNU-noise and surface-smoothness fire together) and `evaluate_taxonomy_classification` — the full 7-branch decision tree (screenshot categorization → fully-AI-explicit-signature → AI-enhanced/composite-explicit-signature → AI-enhanced/composite-score-fallback → high-confidence-synthesis-by-score → screen-recapture → authentic-edited → default-authentic-real). |
| `explain.py` | `build_nine_dimensions_dossier()` (hardware/provenance, photometric/geometry, PRNU noise, surface smoothness, FFT decay, genre/subject, visual medium, sensor spectrum, generative attribution) and `generate_newbie_explanation()` (a plain-English narrative branching on screenshot/synthetic/edited/authentic). |
| `pipeline.py` | `ImageForensicPipeline.analyze()` — chains validator → profiler → provenance → detector (receiving the provenance result) → content analyzer → attribution → dossier/narrative generation into one call, returning one large result dict. |
| `batch.py` | `ImageBatchProcessor.process_files()`/`process_directory()` — runs the pipeline over many files, tallying `ai_generated`/`composite`/`real`/`undecided`/`errors` counts. |
| `benchmarks.py` | `ImageBenchmarkSuite.evaluate_dataset()` — accuracy/precision/recall/F1/ROC-AUC/confusion-matrix against a labeled `ai_generated/`+`real/` folder; this is a **binary** evaluation only (it cannot measure accuracy on the other 8 taxonomy states). |
| `learner.py` | `ImageSelfImprover` — persists feedback records and nudges 5 scalar feature-weights + sensitivity offsets by small fixed deltas per feedback event. **Does not retrain or touch the neural network's weights.** |
| `trainer.py` | `ImageDetectorTrainer` — the actual PyTorch training harness (ResNet18/50/MobileNetV3, `CrossEntropyLoss` + `AdamW`) that produces `models/ai_detector.pt`. Supports directory loading, in-place sample loaders (`loaders_from_samples`), and training from `.npz` feature caches (`prepare_feature_bank`, `train_from_feature_bank`). Binary classifier only: `{ai_generated: 0, real: 1}`. |
| `downloader.py` | `ImageDownloader` — thin wrapper delegating to `core.security.SecureUrlFetcher`. |

### What it can and can't actually distinguish

The optional CNN checkpoint (`models/ai_detector.pt`, not shipped) is a **binary** real-vs-ai_generated classifier once you train one. Everything beyond that binary signal — edited/graphic-design, AI-enhanced/composite, every screenshot sub-state — is produced entirely by the hand-written decision tree in `scoring.py` operating on heuristic pixel/metadata signals, not by anything a network was trained to recognize. With no checkpoint present the detector runs in pure statistical/heuristic mode (the default for a fresh project).

## Audio detector (`audio_detector/`)

**No training data and no trained checkpoint ship for this modality.** The neural-inference branch in `detector.py` stays inactive until you train one, so every analysis runs in pure statistical/heuristic mode.

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

## Video detector (`video_detector/`)

Also **no trained checkpoint ships**, same as audio — the neural temporal-transition model does not load until you train one.

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

## `core/` modules

| File | Role |
|---|---|
| `security.py` | `validate_secure_url()` — rejects non-http(s) schemes, rejects localhost/metadata-service hostnames by name, resolves every A/AAAA record and rejects if *any* resolved IP falls in a private/loopback/link-local/cloud-metadata/multicast/reserved range (19 hardcoded CIDR blocks, IPv4 + IPv6). `_PinnedResolver` closes the DNS-rebinding TOCTOU gap by pinning `socket.getaddrinfo` to only the already-validated IPs for the duration of one request. `SecureUrlFetcher` — streams the download with a byte-count ceiling, rejects `text/html` responses, manually follows up to 3 redirects while re-validating (and re-pinning) every hop. `sanitize_filename()`, `generate_secure_cache_name()`. Also sets `PIL.Image.MAX_IMAGE_PIXELS` at import time as a decompression-bomb guard — importing this module anywhere in the process changes that global Pillow setting. |
| `decision.py` | `normalize_percentages()` — the single canonical implementation all three detector packages' `scoring.py` modules import. Two modes: scale-all-three-together when an `undecided_val` is supplied (what every per-modality detector uses, calibrated against this exact algorithm — don't change it without re-validating every taxonomy threshold), or derive `undecided` from the AI/Real score gap when it's omitted (used internally below). `generate_final_decision()` — the cross-modal Bayesian fusion function: pools image/video/audio AI-probabilities plus provenance/cross-modal/attribution evidence as weighted log-odds terms, derives a `final_status` and (if the per-modality result didn't already carry one) a fallback taxonomy state, and sanitizes attribution for authentic verdicts (forces "None (Authentic Capture)" so an authentic photo never shows a spurious generator attribution). |
| `atomic_io.py` | `atomic_write_json()`/`atomic_read_json()`/`atomic_update_json()` — write-to-temp-then-`os.replace` atomic persistence with per-resolved-path `RLock`s and Windows-specific retry-on-`PermissionError` handling. Used by all three packages' `learner.py` for calibration/feedback persistence. |
| `services/forensic_service.py` | `ForensicService`: a thread-safe lazy singleton exposing `analyze_image/audio/video()`, `record_feedback()` and `health_check()` (builds every component and reports `HEALTHY` or `DEGRADED`). Lives outside `core/` because it depends on the detectors. |

## `ui/` and `app.py`

See the layer overview above. Two details worth knowing:

- `app.py` builds cached detector accessors (`@st.cache_resource`) and one `MediaTabSpec` per modality; `ui.media_tab.render_media_tab` does the rest, so adding a modality means adding a spec, not copying a tab.
- Uploads go to a per-session scratch directory in the OS temp folder (`core.atomic_io.get_session_cache_dir`), never into the repository. "Clear uploaded files" empties only the current session's directory. Results are cached per session by content hash, so re-uploading identical bytes does not re-run the analysis.
