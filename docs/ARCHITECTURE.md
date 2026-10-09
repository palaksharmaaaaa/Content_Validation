# Architecture

How the engine is organised, how one request flows through it, and what each module does. For installation and daily use see the [README](../README.md) and the [user guide](USER_GUIDE.md); for the list of checks see [CHECKS.md](CHECKS.md).

## Layers

Dependencies point downwards only; lower layers never import upwards.

1. **`core/`**: shared, detector-independent code. SSRF-safe fetching, crash-safe JSON storage, decision fusion, probability bands, the dimension-check foundation (`forensics/`), the perception models (`perception/`), the media library and the retraining engine.
2. **`image_detector/`, `audio_detector/`, `video_detector/`**: one self-contained forensic pipeline per modality. They import from `core/` only, never from each other.
3. **`services/`**: `ForensicService` (headless facade over the three pipelines), `calibration_cli` (measure real accuracy), `system_status` (what the sidebar shows), `fetch_models` (download the large pretrained models), and the evaluation tools `age_eval` and `blind_test`.
4. **`ui/` and `app.py`**: the Streamlit presentation layer.

```
app.py            page layout: sidebar + Image / Video / Audio / Link check / Learning tabs
ui/shell.py       sidebar (sensitivity, system status, clear files)
ui/media_tab.py   one implementation of "upload or link -> analyse -> show result", driven by a MediaTabSpec
ui/adapters.py    process_single_image/video/audio and run_batch_pipeline: thin wrappers over the pipelines
ui/validators.py  file validation, URL platform detection, media fetch
ui/results/       result pages: summary (verdict card), checks (evidence), media previews, explanation, feedback,
                  learning dashboard, details, the nine-dimension breakdown (dimensions.py), pages.py composes them
ui/stages.py      modality-agnostic finding / band / gate components
```

## How one request flows

```
upload / link
  -> ui.validators                       format check, SSRF-safe download
  -> ui.adapters.process_single_*        thin adapter
       -> <modality>_detector.pipeline.*ForensicPipeline.run()      the single analysis sequence:
            gates -> profile -> provenance -> dimension checks (pre) -> detector -> content -> attribution
            -> dimension checks (post) -> core.decision.generate_final_decision -> dossier -> explanation
  -> ui.results.pages.render_*_result    verdict card + Overview | Evidence | Details | Feedback
```

`pipeline.analyze()` turns the same `run()` output into the headless report used by `ForensicService`; `tests/test_ui_flow_apptest.py` asserts both paths give the same verdict.

## Verdict semantics

- **Probabilities are heuristic and uncalibrated.** Every dossier carries `calibration_status: "UNCALIBRATED_HEURISTIC"` and the UI says so. Thresholds were tuned on synthetic fixtures; real error rates are measured only if you run `services.calibration_cli` on your own labelled media.
- **Evidence is capped.** Only physical, learned and weak-metadata findings may move the probability, each by a small capped amount (see [CHECKS.md](CHECKS.md#what-may-change-the-probability)). Security, legal, context and reliability findings, the bands, the out-of-distribution status and `UNKNOWN_SOURCE` are advisory.
- **C2PA is presence-only.** Byte markers are looked for; no certificate chain or hash binding is validated, so it is never scored as protective and never called "verified".
- **Camera EXIF is unauthenticated.** In `ImageAIDetector.predict`, EXIF earns hardware credit, but the credit shrinks and the physical-signal discount is withheld when the raw noise and smoothness evidence leans synthetic by at least `EXIF_CONTRADICTION_LR`.
- **Two decision modes** (`decision_mode` from `core.decision.generate_final_decision`): `image_authoritative` (a single image result with a taxonomy state is final) and `fused` (video, audio, cross-modal and declared-AI provenance are pooled as weighted log-odds). A video verdict comes from the video detector's own result, not from a keyframe image. Attribution is explanation only and is never counted as evidence.
- **A verdict never rests on a missing measurement.** When a value could not be measured (clip too short, no content analysis, no recorded DPI) the dossier says "not measured" or "not recorded"; no default stands in for it. The nine-dimension page shows such fields in a neutral state, not as a pass.
- **Out-of-distribution gate.** `core/forensics/ood.py` measures a Mahalanobis distance on embeddings. Until fitted from your media library it reports `NOT_CALIBRATED`; it never invents a threshold.

## Perception models (`core/perception/`)

Pretrained models that describe an image's content. All are Apache-2.0 and run on CPU. `python -m services.fetch_models` downloads the two large ones at pinned revisions; the rest ship in the repository.

| Job | Model | Where |
|---|---|---|
| Face detection | YuNet (0.2 MB) | `core/models/` |
| Object detection (people, animals, vehicles, objects) | RF-DETR Small (about 130 MB) | Hugging Face cache, revision pinned in `detector.py` |
| Place, species, vehicle type, genre, time of day | SigLIP 2 Base, zero-shot against `vocab.py` (1.4 GB on disk; its text tower is released after load, see below) | Hugging Face cache, revision pinned in `recognizer.py` |
| Facial expression | OpenCV zoo MobileFaceNet (4.8 MB) | `core/models/` |
| Same-person matching | OpenCV zoo SFace (39 MB) | `core/models/` |
| Apparent age | MiVOLO v2 (115 MB, Git LFS), network code vendored in `mivolo_vendor/` | `core/models/mivolo_v2/` |
| Dominant colours | CIELAB clustering plus a colour-name table (no model) | `colors.py` |

The text side of SigLIP 2 is only needed to embed the fixed prompts in `vocab.py`. Those 182 embeddings ship in `core/perception/vocab_embeddings.npz` (fingerprinted by the pinned revision and every prompt; rebuild with `python -m services.build_vocab_embeddings` after editing a vocabulary, a test fails until you do), and the recognizer frees the text tower after loading, saving about 1 GB of memory. A missing large model makes its part of the report empty or `UNAVAILABLE`; it never stops the analysis. Modules: `age.py` (ages every person, recall-first), `age_video.py` and `video_sampling.py` (minor screening across a video with adaptive sampling), `face_scan.py` (small-face finder), `face_texture.py` (the facial texture cue shared by the image and video detectors), `face_attributes.py`, `enrich.py` (adds recognition details to detections), `hub.py` (offline-friendly model loading, pinned revisions, Git LFS pointer detection).

## Face authenticity

`core/face_detection.py` finds faces. `image_detector/face_authenticity.py` crops each one the same way for training and inference and classifies it with a small ResNet-18 trained by `image_detector/face_training.py`; the checkpoint `image_detector/models/face_authenticity.pt` ships with the repository (Git LFS). `image_detector/dimension_checks/faces.py` turns the result into one `LEARNED_SIGNAL` finding: positive-only, capped at +0.40 log-odds, and it blocks a "likely real" verdict via `core.decision`.

## The three pipelines

Each package follows the same shape.

| File | Role |
|---|---|
| `config.py` | paths, size and duration limits, thresholds, calibration defaults |
| `schemas.py` | typed results (and for images the nine `ImageTaxonomyState` states) |
| `validator.py` | existence, extension, size, decodability, quality |
| `profiler.py` | hashes, geometry or audio/stream facts, statistics |
| `provenance.py` | C2PA byte-marker scan and metadata signatures |
| `features.py` (image, audio), `temporal.py` and `extractor.py` (video) | the measured signals |
| `detector.py` | the scoring engine; loads an optional trained checkpoint |
| `scoring.py` | pooling and thresholds (image: Bayesian log-odds plus the decision tree; audio and video: weighted averages) |
| `content.py`, `attribution.py` | what the media contains; which known generator it resembles |
| `explain.py` | the nine-dimension dossier and the plain-English explanation |
| `pipeline.py` | `run()` (the single analysis sequence) and `analyze()` (headless report) |
| `learner.py`, `retrain.py`, `trainer.py` | feedback records, the fine-tuning entry point, the PyTorch trainer |
| `batch.py`, `benchmarks.py` | thin subclasses of `core.batch` / `core.benchmark` for the package (URLs are fetched by `core.security`; the app's own batches are parallel, see Operations) |
| `dimension_checks/` | the isolated checks (see [CHECKS.md](CHECKS.md)) |

Package-specific points:

- **Image**: `feature_store.py` is an optional, rebuildable feature cache keyed by content hash. `face.py` counts faces and scores a texture heuristic for waxy skin (not a trained deepfake detector); `face_authenticity.py` and `face_training.py` are the trained face check. `learner.py` nudges five scalar weights per feedback event and does not retrain a network.
- **Audio**: `validator.py` uses `ffmpeg` to decode to 16 kHz mono; if `ffmpeg` is not installed it reads WAV files natively and nothing else. Limits: 200 MB, one hour.
- **Video**: `extractor.py` samples frames uniformly (frames below 64 px on the short side are rejected). The representative keyframe is chosen from the sampled frames, and the audio track is decoded by the audio package through `ffmpeg` (skipped, with a warning, if it is missing). `cross_modal.py` compares the video result with an audio result supplied by the caller. Frames are scored by the shared noise and smoothness scorer (`core/frame_scorer.py`, also used by the image detector's `predict_frame`) unless an external frame detector is attached; a frame that cannot be scored is recorded as such and never counted. Too few frames for a motion or flicker reading gives no reading (`None`), and that cue drops out of the pooled score. Limit: 500 MB. Minor screening reads the file itself with adaptive sampling (`core/perception/age_video.py`).

## `core/` modules

| File | Role |
|---|---|
| `security.py` | `validate_secure_url` rejects non-http(s) schemes, local hostnames and any host resolving to a private, loopback, link-local, metadata, multicast or reserved address; `_PinnedResolver` pins DNS to the validated addresses for one request (closing the rebinding gap); `SecureUrlFetcher` streams with a size ceiling, rejects HTML, and re-validates every redirect (at most three). `sanitize_filename` treats `/` and `\` as separators on every OS. Importing it sets Pillow's decompression-bomb limit for the whole process. |
| `decision.py` | `normalize_percentages` (the one implementation all three scoring modules use) and `generate_final_decision` (the fusion described above). |
| `atomic_io.py` | write-to-temp-then-replace JSON persistence with per-path locks and Windows retry on `PermissionError`; per-session scratch folders in the OS temp directory. |
| `media_library.py`, `retrain_engine.py` | labelled media by reference (hash and path hint, never a copy); fine-tuning with a hash-based validation split, promoting a new checkpoint only if its validation accuracy does not drop. |
| `checkpoint_log.py` | a plain-text, Git-tracked provenance log for trained checkpoints. |
| `bands.py`, `calibration_report.py` | the five probability bands; accuracy, false-positive rate, Brier score and ECE from labelled pairs. |
| `forensics/` | `schemas.py` (Finding, DimensionReport), `registry.py` (isolated checks), `gates.py`, `ood.py`, `bytescan.py`, `jsonl_index.py` (re-use indexes read once per change), `reporting.py`, `config.py`. |
| `hashing.py`, `imageio.py` | one streaming file-digest implementation; image reading that works for every file name and bit depth. |
| `frame_scorer.py`, `ffmpeg.py`, `c2pa.py` | the one hardened way to call ffmpeg/ffprobe (file and pipe protocols only, no stdin, bounded decode) and the one PCM decoder; the one C2PA presence scan used by all three packages. |
| `batch.py`, `benchmark.py` | the sequential headless batch runner (one failing file never stops the run; results bucketed by exact status) and the labelled-folder scorer (rank-based ROC-AUC, abstentions and failures reported). |
| `shared_results.py`, `provenance_view.py`, `metrics_util.py` | result shapes and label rules shared by the three packages; the single nested provenance shape used by the decision layer and the UI; conversion of metric values to JSON-safe scalars. |
| `filecache.py`, `lazy.py`, `logging_filters.py` | a memoiser for expensive per-file parses that invalidates when the file changes; lazy package re-exports so light imports do not load torch; a filter for one benign Windows asyncio log line. |

## Operations

- **Parallel batches.** `ui.adapters.run_batch_pipeline` analyses several files at once (at most half the CPU cores; 4 for images and audio, 2 for video; `OMNI_BATCH_WORKERS` overrides, 1 turns it off). Results keep input order, a failing file becomes an error result, and the progress callback is only called from the calling thread because Streamlit forbids calls from workers. Every shared model takes its own lock, which is why concurrent use gives results identical to sequential use.
- **Model warm-up.** `services/warmup.py` loads the perception models in one background thread when the app starts, so the first analysis does not pay for model loading (`OMNI_WARMUP=0` disables it).
- **Memoised Fourier analysis.** The image detector and the moire check ask for the same spectrum within one analysis; the last result is reused when the pixels are identical.

- Uploads go to a per-session scratch directory (`core.atomic_io.get_session_cache_dir`), never into the repository. "Clear uploaded files" empties only the current session's directory. Results are cached per session by content hash.
- The root `conftest.py` fingerprints `*/models/*.pt`, `*/data/*.json|npz` and `core/data/*` before and after a run and fails the session if any test changed them.
- Optional operator files (hard-block list, reuse indexes, OOD statistics) are all git-ignored; see the [user guide](USER_GUIDE.md#optional-local-files).
