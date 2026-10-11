# Testing

```bash
pytest                                   # whole suite (about three minutes on a laptop CPU)
pytest tests/test_system_status.py -q    # one file
```

`pytest.ini` collects `core/tests`, `image_detector/tests`, `audio_detector/tests`, `video_detector/tests` and `tests`. The root `manual_pipeline_smoke.py` is a manual script, not part of the suite (`python manual_pipeline_smoke.py`).

## What is covered

- **Unit tests** for each module (security, atomic I/O, scoring, dimension checks, media library, retraining engine).
- **End-to-end UI tests** drive the real `app.py` through Streamlit's `AppTest`: upload an image, a recording and a video, and assert the result page renders without exceptions. Another test checks that the session cache is keyed by file content.
- **Golden regression tests** pin the output of decision fusion, `predict`, taxonomy, attribution, dossiers, narratives, metadata, faces, screenshots, digital art, audio/video analysis, headless pipelines and the rendered result pages.
- **Regression tests** (`tests/test_audit_regressions.py` and the per-package suites) keep every bug that was found and fixed from coming back, including the ones found by running the suite on a fresh clone and on Linux.
- **Repository hygiene** (`tests/test_repo_hygiene.py`): fails if any image, audio or video file is tracked, if a required model file is not tracked, or if any tracked file contains a machine-specific path.
- **Isolation guard**: the root `conftest.py` fingerprints model, data and calibration files before and after the run and fails the session if any test changed them.

All inputs are synthetic (tones, noise, drawn shapes, generated clips). The suite proves the code behaves as designed; it does **not** measure detection accuracy. See [Limitations](LIMITATIONS.md).

## Golden files

Goldens live in `tests/data/<name>.digests.json` as per-case SHA-256 digests, so a change shows up as a short diff.

When a behaviour change is intended:

```bash
UPDATE_GOLDEN=1 pytest path/to/test_file.py     # PowerShell: $env:UPDATE_GOLDEN=1; pytest ...
git diff tests/data                              # review: only the cases you expected should change
```

Goldens are written to be identical on every OS: values that depend on how the test's own files were encoded (hashes, byte sizes, the ffmpeg version) are blanked by `scrub_encoding` in `tests/golden_support.py`.

Goldens that depend on learned state (calibration files, a checkpoint) are recorded on a blank project and skip themselves, with a message, when such state exists. Run the suite on a clean checkout to exercise them.

## Adding a check

1. Implement it as an isolated function in the modality's `dimension_checks/` package and register it. A failing check must return an `ERROR` finding, never raise.
2. Add a unit test with a minimal synthetic file that triggers it and one that does not.
3. Decide whether it may add log-odds. Only `PHYSICAL_SIGNAL`, `METADATA_WEAK` and `LEARNED_SIGNAL` findings may, and only as small capped terms; everything else is advisory.
4. Regenerate the affected goldens as above and review the diff.

## Continuous integration

`.github/workflows/tests.yml` runs on every push and pull request: Ubuntu, Python 3.14, Git LFS objects (restored from a cache so runs do not spend the LFS bandwidth quota), CPU PyTorch plus `requirements.lock.txt`, `ffmpeg`, the two pinned pretrained models (cached), then `pytest`. It sets `GOLDEN_DUMP=1`, so a golden mismatch prints the differing values. Windows is tested locally; macOS is not tested.


## Coverage

Measured on 2026-10-12 on Python 3.14 with `coverage run --source=core,image_detector,audio_detector,video_detector,services,ui -m pytest` (821 tests): **87 % of statements** (14,459 statements, 1,834 not run; line coverage, not branch coverage; test files excluded). The command-line tools `services/blind_test.py`, `services/calibration_cli.py` and `services/build_vocab_embeddings.py` are now covered. The weakest parts are the tools that need real data or a long training run: `services/age_eval.py` (34 %), the three `fit_ood.py` scripts (37-52 %), `services/fetch_models.py` (46 %), `image_detector/face.py` (47 %) and `image_detector/face_training.py` (47 %). Coverage says which lines ran, not that their results are right: the golden tests pin outputs, and real accuracy on real AI media is unmeasured (see [LIMITATIONS.md](LIMITATIONS.md)).

## Lint, fuzzing and optional quality tools

`pip install -r requirements.txt` includes the tools below; CI installs them from `requirements.lock.txt`.

- **`ruff check .`** (rules in `pyproject.toml`: pycodestyle, pyflakes, isort, pyupgrade for Python 3.14, bugbear, simplify) and **bandit** (security lint, same file) run before the tests in CI and are clean today.
- **`tests/test_property_fuzzing.py`** feeds random bytes to the C2PA scanner, the RIFF and signature parsers and the image profiler with Hypothesis, and asserts they never crash. It skips when Hypothesis is not installed.
- **Evidently** (`core/retrain_engine.py`): before a retrained model is promoted, a drift report compares the validation and training samples. The only feature it compares is file size, so it flags unusually differently sized files, not a change in what the media show; it only adds a warning to the result message. With fewer than 10 samples on either side, or without Evidently, the report is skipped.
- **Cleanlab** (`ui/results/feedback.py`): a correction that contradicts a detector result that is at least 80 % sure the other way, with a label quality score under 0.20, is held back until the reviewer ticks a confirmation box, so one mistaken click does not move the thresholds. For one example the score is the detector's own probability for the label chosen (Cleanlab's default "self confidence"); it is computed directly when Cleanlab is not installed, so the guard never switches off.
- **`tests/test_video_memory_profile.py`** bounds Python-side memory during frame extraction with `tracemalloc`; it cannot see leaks in OpenCV's native buffers.
