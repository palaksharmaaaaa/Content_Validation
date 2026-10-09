# OmniForensics

A local tool that checks whether an **image, video or audio file** is likely AI-generated or edited, and shows the evidence behind every verdict. It also describes what is in an image (people, animals, vehicles, place, light) and screens for minors. Built with Streamlit; it runs on your machine, and your files are never copied into the repository.

> **Read this first.** The AI-versus-real decision is a hand-tuned statistical scoring engine, not a trained classifier, and its overall accuracy has **not been measured**: no labelled dataset ships with the project. Treat a verdict as a lead to investigate, never as proof. [Why, and how to measure it yourself](docs/USER_GUIDE.md#how-much-to-trust-a-verdict). The parts that *were* measured (age screening, false alarms on real photographs) are reported in [Limitations](docs/LIMITATIONS.md).

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Models](#models)
- [Data](#data)
- [Install](#install)
- [Run](#run)
- [Command-line tools](#command-line-tools)
- [Configuration](#configuration)
- [Project tree](#project-tree)
- [Where things are written](#where-things-are-written)
- [Development](#development)
- [Security](#security)
- [Documentation](#documentation)

## What it does

| You give it | You get |
|---|---|
| An image (`.jpg .jpeg .jfif .png .webp .bmp .tif .tiff`, up to 100 MB) | A verdict card, evidence from 16 checks, a scene description, face and age analysis, a nine-dimension breakdown, a plain-English explanation and a JSON report |
| A video (`.mp4 .mov .avi .mkv .webm`, up to 500 MB or 1 hour) | The same, from container checks, per-frame signals, motion and flicker analysis, adaptive minor screening and the audio track |
| Audio (`.wav .mp3 .aac .flac .ogg .m4a`, up to 200 MB or 1 hour) | The same, from container checks, spectral signals, a per-window timeline and generator attribution |
| Several files, or direct links | A comparison table (CSV or JSON download) and a picker; links go through a hardened downloader |

Also: a feedback loop (tell it when it was wrong and, after enough corrections, fine-tune on your own media), a face check (a trained real-versus-generated classifier ships with the repository), and tools to measure accuracy on media you label yourself.

What is checked, exactly: [docs/CHECKS.md](docs/CHECKS.md).

## How it works

```
upload / link
  -> format check, SSRF-safe download
  -> gates            (your SHA-256 block list; scientific formats the engine cannot score)
  -> profile + provenance   (hashes, geometry, metadata, C2PA marker scan)
  -> dimension checks "pre" (integrity, container, metadata, signal checks, face authenticity)
  -> detector         (pixel / spectral / temporal signals -> probability)
  -> content + attribution  (what is in the file; which generator the file itself declares)
  -> dimension checks "post" (reuse fingerprint, rights, re-encoding, confidence limiters)
  -> decision         (one verdict, five confidence bands, nine-dimension dossier, explanation)
```

Only physical, learned and weak-metadata findings can move the probability, each by a small capped amount; everything else is advisory. A value that could not be measured is shown as "not measured", never replaced by a default. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Models

| Model | Job | Size | Where it lives | How you get it |
|---|---|---|---|---|
| MiVOLO v2 (network code vendored) | apparent age of every person (face + body) | 110 MB | `core/models/mivolo_v2/model.safetensors` | in the repository (Git LFS), pinned to revision `5339352` |
| Face-authenticity ResNet-18 | real versus generated face | 43 MB | `image_detector/models/face_authenticity.pt` | in the repository (Git LFS); retrain with `image_detector.face_training` |
| YuNet | face detection | 0.2 MB | `core/models/` | in the repository |
| OpenCV zoo MobileFaceNet | facial expression | 4.6 MB | `core/models/` | in the repository |
| OpenCV zoo SFace | same-person matching | 37 MB | `core/models/` | in the repository |
| RF-DETR Small | people, animals, vehicles, objects | about 130 MB | Hugging Face cache | `python -m services.fetch_models`, pinned to revision `3bdc465` |
| SigLIP 2 Base (zero-shot) | place, species, vehicle type, genre, time of day, age-group opinion | about 1.4 GB on disk, about 0.7 GB in memory | Hugging Face cache; its 182 prompt embeddings ship in `core/perception/vocab_embeddings.npz` | `python -m services.fetch_models`, pinned to revision `75de2d5` |

Pinned revisions mean every machine loads identical weights. **Not included:** a trained AI-versus-real backbone for any modality, calibration history, feedback and labelled datasets; until you supply and train on them, the AI-versus-real score comes from the statistical signals alone and every probability is labelled `UNCALIBRATED_HEURISTIC`. An optional image/audio/video backbone checkpoint (`<modality>_detector/models/*.pt`) is picked up automatically if you train one.

## Data

**The repository contains no photograph, audio or video.** A test (`tests/test_repo_hygiene.py`) fails if any media file is tracked, if a required model file is missing, or if any tracked file contains a machine-specific path. Training and evaluation data stay where you keep them: the media library stores each file's SHA-256, label and a path hint, never a copy.

Public datasets used only to *evaluate* the age screening (UTKFace, LAGENDA, FairFace) are described in [docs/AGE_DATASETS.md](docs/AGE_DATASETS.md); what to collect to train an AI-versus-real detector is in [docs/DATASET_SPEC.md](docs/DATASET_SPEC.md). The test suite uses only synthetic inputs (tones, noise, drawn shapes, generated clips).

## Install

**Requirements:** Python 3.10 (the version CI and development use), [Git](https://git-scm.com) with [Git LFS](https://git-lfs.com), about 2 GB of free disk (1.6 GB of pretrained models plus about 200 MB of checkpoints and models), and [`ffmpeg`](https://ffmpeg.org/download.html) on your `PATH` for compressed audio and for the audio track of videos. Without `ffmpeg`, WAV and video frames still work. A GPU is optional and untested; everything runs on CPU.

**1. Clone with Git LFS.** The checkpoints are Git LFS objects; a clone made without LFS contains small text stubs instead and the age screening and face check will report themselves unavailable.

```bash
git lfs install                 # once per machine, before cloning
git clone <repository-url>
cd <repository-folder>
```

If you already cloned without LFS: `git lfs install && git lfs pull`.

**2. Create the environment and install.**

```bash
python -m venv .venv
```

Activate it with the line for your shell, then install:

| Shell | Activate |
|---|---|
| macOS, Linux (bash, zsh) | `source .venv/bin/activate` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Git Bash | `source .venv/Scripts/activate` |

```bash
pip install -r requirements.txt
```

`requirements.txt` states minimum versions; `requirements.lock.txt` pins the exact versions the project is developed and tested against (use it for a reproducible environment). Headless servers can use `opencv-python-headless` instead of `opencv-python`.

**3. Download the two large pretrained models** (one-off, about 1.6 GB, needs internet; afterwards the app works offline):

```bash
python -m services.fetch_models           # download what is missing
python -m services.fetch_models --check   # only report
```

**4. Install ffmpeg** (optional but recommended): `winget install ffmpeg` or download from ffmpeg.org on Windows, `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Debian and Ubuntu.

## Run

```bash
streamlit run app.py
```

Open the address Streamlit prints (usually `http://localhost:8501`), choose **Image**, **Video** or **Audio**, and drop a file or paste direct links. The sidebar's **System status** shows what is active: each model, calibration state, the age-screening model, `ffmpeg` and the compute device. Models load on first use, so the first analysis is slower.

Headless use (no UI), for scripts and tests:

```python
from services.forensic_service import ForensicService

report = ForensicService.get_instance().image_pipeline.analyze("photo.jpg")   # also audio_pipeline, video_pipeline
```

## Performance

Measured on a Windows laptop CPU with 14 threads, no GPU, Python 3.10 (your numbers will differ; the ratios are what matter):

| What | Time |
|---|---|
| Importing the service layer (torch dominates) | about 6 s |
| First image analysis in a fresh process, models loading on demand | 7.5 s |
| First image analysis when the background warm-up has finished | 1.9 s |
| Image analysis once warm (1200 x 1600 photograph, full chain) | about 1.2-1.7 s |
| Audio analysis (20 s clip) | about 0.3 s |
| Video analysis (10 s clip, 250 frames, minor screening included) | about 4 s |
| Batch of 12 images, 1 worker versus 4 workers | 16.6 s versus 8.2 s |
| Process memory after an image analysis (all image models loaded) | 1.2 GB (it was 2.4 GB before SigLIP 2's text tower was released); about 1.3 GB while six analyses run at once |
| Memory growth over 40 further analyses | none measurable |

Hostile input is cheap: truncated, empty, mislabelled and decompression-bomb files are rejected in well under a second with a clear error.

## Command-line tools

All are run from the repository root with `python -m <module>`.

| Module | Purpose |
|---|---|
| `services.fetch_models` | download (or `--check`) the pretrained models |
| `services.build_vocab_embeddings` | rebuild the shipped SigLIP prompt embeddings after editing `core/perception/vocab.py` |
| `core.media_library <image\|audio\|video> add --label real\|ai_generated <paths>` / `stats` / `rescan` | register labelled media by reference |
| `services.calibration_cli --modality image\|audio\|video` | accuracy, false-positive rate, Brier score, ECE and band occupancy on your held-out files |
| `image_detector.face_training --real <dirs> --ai <dirs>` | train the face-authenticity checkpoint |
| `<modality>_detector.dimension_checks.fit_ood` | fit the out-of-distribution gate from your library |
| `<modality>_detector.trainer` | train a backbone |
| `<modality>_detector.benchmarks --dataset <dir> [--output report.json]` | score a labelled folder (`real/`, `ai_generated/`): accuracy, precision, recall, F1, ROC-AUC, latency; abstentions and failures are counted, never scored as "real" |
| `services.age_eval run\|wild\|fairface` | measure the age screening on public datasets |
| `services.blind_test run\|summary` | run the image pipeline over folders without looking at labels |
| `manual_pipeline_smoke.py` | a manual end-to-end smoke script (not part of the test suite) |

## Configuration

Nothing needs configuring to start. Optional operator files and variables (all git-ignored; none ship):

| Purpose | Default location | Environment variable |
|---|---|---|
| Hard-block list (SHA-256, one per line; a match stops analysis) | `core/data/hardblock_sha256.txt` | `OMNI_HARDBLOCK_SHA256_FILE` |
| Image re-use index (JSON lines with `phash`) | `image_detector/data/context_hash_index.jsonl` | `OMNI_CONTEXT_HASH_INDEX` |
| Audio fingerprint index | `audio_detector/data/audio_fingerprint_index.jsonl` | `OMNI_AUDIO_FP_INDEX` |
| Video fingerprint index | `video_detector/data/video_fingerprint_index.jsonl` | `OMNI_VIDEO_FP_INDEX` |
| Out-of-distribution statistics | `<modality>_detector/data/ood_stats.npz` | written by `fit_ood` |
| Load the models in the background at start (default on; `0` turns it off) | none | `OMNI_WARMUP` |
| Files analysed at once in a batch (default: up to half the cores; `1` turns it off) | none | `OMNI_BATCH_WORKERS` |

Sensitivity (Balanced, High, Aggressive) is set in the sidebar. Limits and thresholds live in each package's `config.py`.

## Project tree

```
.
├── app.py                     Streamlit entry point (sidebar + Image / Video / Audio / Link check / Learning tabs)
├── requirements.txt           minimum versions          requirements.lock.txt   exact tested versions
├── pytest.ini  conftest.py    test discovery and the isolation guard
├── manual_pipeline_smoke.py   manual smoke script
├── .github/workflows/         tests.yml: CI (Linux, Python 3.10, Git LFS, pinned models)
├── core/                      shared, detector-independent code
│   ├── security.py            SSRF-hardened fetching, filename sanitising
│   ├── decision.py            fusion of image / video / audio evidence into one verdict
│   ├── atomic_io.py           crash-safe JSON, per-session scratch folders
│   ├── media_library.py  retrain_engine.py  checkpoint_log.py   labelled media by reference, fine-tuning
│   ├── bands.py  calibration_report.py   probability bands, accuracy measurement
│   ├── forensics/             Finding schema, check registry, gates, OOD gate, byte scans
│   ├── perception/            age (MiVOLO), small-face finder, video sampling, object + scene recognition
│   └── models/                YuNet, SFace, expression ONNX files; mivolo_v2/ weights
├── image_detector/            image pipeline (detector, features, scoring, attribution, explain, face authenticity)
│   ├── dimension_checks/      16 image checks     models/   face_authenticity.pt
├── audio_detector/            audio pipeline      └── dimension_checks/
├── video_detector/            video pipeline      └── dimension_checks/
├── services/                  ForensicService, system_status, fetch_models, calibration_cli, age_eval, blind_test
├── ui/                        presentation layer (tabs, result pages, sidebar, batch views)
├── tests/                     cross-package tests and golden digests (tests/data)
├── docs/                      USER_GUIDE, CHECKS, ARCHITECTURE, TESTING, LIMITATIONS, DATASET_SPEC, AGE_DATASETS
└── <package>/tests/           per-package tests
```

About 300 tracked files and 34,000 lines of Python, including tests.

## Where things are written

| What | Where | In git? |
|---|---|---|
| Uploaded and downloaded files | per-session folder under the OS temp directory (`omni_forensics_ephemeral_cache`), removed by **Clear uploaded files** | no |
| Media library | `<modality>_detector/data/library.json` | no |
| Feedback memory and calibration | `<modality>_detector/data/*_memory.json`, `*_calibration.json` | no |
| OOD statistics, reuse indexes | `<modality>_detector/data/` | no |
| Candidate checkpoints during retraining | `<modality>_detector/models/*.candidate.pt` | no |
| Evaluation output, logs | `reports/` | no |
| Trained checkpoints you promote | `<modality>_detector/models/*.pt` and `CHECKPOINT_LOG.md` | `.pt` files yes (Git LFS) |

The project starts with none of the "no" entries; they appear as you use it.

## Development

```bash
pytest                      # whole suite, about three minutes on a laptop CPU
pytest tests/test_x.py -q   # one file
```

- **Golden files** (`tests/data/*.digests.json`) pin outputs; regenerate deliberately with `UPDATE_GOLDEN=1 pytest <file>` and review the diff. They are written to hold on every OS.
- **Guards:** `conftest.py` fails the run if a test changes a model, data or calibration file; `tests/test_repo_hygiene.py` blocks committed media and machine-specific paths.
- **CI:** `.github/workflows/tests.yml` runs the suite on every push and pull request on Ubuntu with Python 3.10. Windows is tested locally; macOS has never been run.
- **Adding a check:** implement it as an isolated function in the modality's `dimension_checks/` package and register it; see [docs/TESTING.md](docs/TESTING.md).
- **Never commit media or trained data.** `.gitignore` blocks image, video and audio extensions and the `data/` folders; checkpoints are the only binaries tracked, through Git LFS.

## Security

- Link downloads go through an SSRF-hardened fetcher: ports 80/443 only, DNS pinned to validated addresses, redirects re-validated, size and content-type limits.
- Uploads live in a per-session temporary folder and are removed by **Clear uploaded files**.
- There is **no authentication**. The app is meant for personal, local use; do not expose it to the internet as is.
- No licence file is included in the repository; ask the owner before reusing the code.

## Documentation

| Document | For |
|---|---|
| [User guide](docs/USER_GUIDE.md) | Using the app, reading a verdict, minor screening, giving feedback, training on your media, measuring accuracy |
| [What is checked](docs/CHECKS.md) | Every check, per media type, and what may change the probability |
| [Architecture](docs/ARCHITECTURE.md) | Layers, request flow, every module |
| [Testing](docs/TESTING.md) | Running the tests, golden files, adding a check, CI |
| [Limitations](docs/LIMITATIONS.md) | What is unverified or missing, with the measurements that exist |
| [Dataset specification](docs/DATASET_SPEC.md) | What to collect to train and test an AI-versus-real detector |
| [Age datasets](docs/AGE_DATASETS.md) | Public datasets for checking the age and minor screening |
