# User guide

## Contents

- [Check a file](#check-a-file)
- [Read a result](#read-a-result)
- [How much to trust a verdict](#how-much-to-trust-a-verdict)
- [Sensitivity](#sensitivity)
- [Give feedback](#give-feedback)
- [Train on your own media](#train-on-your-own-media)
- [Measure real accuracy](#measure-real-accuracy)
- [Optional local files](#optional-local-files)
- [Troubleshooting](#troubleshooting)

## Check a file

1. Start the app with `streamlit run app.py`.
2. Choose **Image**, **Video** or **Audio**.
3. Drop one or more files on the uploader, or open **Or analyse files from links** and paste direct links (one per line).
4. With several files you get a comparison table (downloadable as CSV or JSON) and a picker to open any file.

The **Link check** tab only validates that a social-media link is well formed and matches the platform you expect; it does not open the link.

## Read a result

Every result starts with a **verdict card**: the most likely category, an AI-likelihood score, how strong the evidence is, the main reasons, and a note on how far to trust it. Below it:

| Tab | What it shows |
|---|---|
| Overview | A preview (image, video frame, audio player and timeline) and a plain-English explanation |
| Evidence | Every check grouped by topic, each marked pass / warning / fail / info, with its detail |
| Details | The full file profile, category, detections and the nine-dimension breakdown |
| Feedback | Say whether the result was right, and download the full JSON report |

Status words: **Pass** nothing unusual; **Warning** worth a look; **Fail** a strong anomaly; **Info** context only; **N/A** the check does not apply to this file; **Not calibrated** the check needs data you have not supplied yet.

A blocked-file banner means the file matched your hard-block list or is a recognised format the tool does not score (for example DICOM); no AI verdict is given in that case.

### Minor screening

For images and videos, a notice appears directly under the verdict card:

- **Likely minor / Possible minor: send to a human reviewer.** At least one person looks young enough that the tool will not clear them. It gives the youngest *apparent* age (an estimate from appearance, never anyone's real age). For a video it also says in how many of the examined frames this happened, and at which times.
- **Nobody flagged.** Nothing was flagged among the people found. This is not a guarantee: small, hidden or turned-away faces can be missed.
- **Did not run.** The age model could not be loaded or the file could not be read, so nobody was checked.

The screening is deliberately cautious: it sends many adults to review as well (see [Limitations](LIMITATIONS.md#age-and-minor-screening) for the measured rates). It reports; a person decides.

## How much to trust a verdict

- Scores are **heuristic and uncalibrated**. Thresholds were tuned on synthetic test files only. The real error rate is unknown until you measure it ([below](#measure-real-accuracy)).
- A **confidence band** (very likely AI / leaning AI / inconclusive / leaning real / very likely real) is shown instead of a bare number. Prefer the band.
- **C2PA** is detected by presence only. The signature is not verified, so it is never counted as proof of authenticity.
- **Camera metadata (EXIF)** can be forged and earns no trust on its own.
- **Generator attribution** names a generator only when the file itself declares one (a watermark, or a metadata or software tag). Without that it says "Unknown". It is an explanation aid and never changes the verdict.
- Heavily compressed reposts, screenshots and low-resolution files weaken every signal.
- Faces are counted with a trained detector (YuNet), but the facial deepfake-risk score is a texture heuristic. The separate face-authenticity check is a trained classifier whose generality is limited (see [Limitations](LIMITATIONS.md)).
- A field the tool could not measure is shown as "not measured" or "not recorded", never as a normal reading.
- A file with nothing to analyse (one flat colour, silence, blank frames) gets "No usable content" instead of a verdict.

More detail: [Limitations](LIMITATIONS.md).

## Sensitivity

Set in the sidebar.

| Mode | Use it when | Cost |
|---|---|---|
| Balanced (default) | General use. No lean toward AI or real before evidence | None |
| High sensitivity | You expect modern generators and accept more false alarms | Ambiguous real media is flagged more often |
| Aggressive | Heavily compressed reposts | Highest false-alarm risk |

## Give feedback

On the **Feedback** tab choose what the file really is (*AI-generated*, *Real / authentic*, *Edited or partly AI*) and save. The app keeps a copy of the file in its own data folder on your machine (`<type>_detector/data/feedback_media/`, named by the file's content; nothing is uploaded anywhere) so a later retrain can still read it, records its hash, and nudges its scoring constants slightly. Saving the same file with the same answer again changes nothing, and an answer other than AI or real records nothing. This is not model training; training is a separate step.

## See which models were used

Every result has a line under the verdict naming what decided it and how many files this project trained models on, and a **Models, algorithms and training data** panel listing every step with its model, variant, whether it is pretrained, fine-tuned here or hand-written, the data it was trained on, the number of training files, and its licence. The same table is in the downloaded JSON report.

## Size limits

One image or audio file may be up to 10 MB and one video up to 100 MB; one batch may hold up to 500 MB per type. Edit `limits.toml` (or set `OMNIFORENSICS_MAX_<IMAGE|VIDEO|AUDIO>_<FILE|BATCH>_MB`) and restart the app to change them. A file or batch over the limit is left out with the reason shown.

## Train on your own media

Your files stay where they are. A local registry (`<modality>_detector/data/library.json`, git-ignored) stores each file's SHA-256, its label and a path hint.

```bash
# register labelled media by reference (files or folders, recursive)
python -m core.media_library image add --label real "<folder of real photos>"
python -m core.media_library image add --label ai_generated "<folder of AI-generated images>"
python -m core.media_library image stats

# after moving folders, re-link files by content
python -m core.media_library image rescan "<moved folder>"
```

Then fine-tune from Python (same API in `audio_detector` and `video_detector`):

```python
from image_detector.retrain import count_pending_corrections, run_retrain
print(count_pending_corrections())
result = run_retrain(min_new=15, epochs=5)
print(result.promoted, result.new_accuracy, result.message)
```

Rules the trainer follows:

- About 20 % of files (chosen by hash, always the same ones) are held out for validation and never trained on.
- The new checkpoint replaces the live one only if its validation accuracy is at least the last promoted value; otherwise it is discarded and your labels stay queued.
- Identity is exact bytes. A re-saved or resized copy counts as a different file and could land on the other side of the split.
- To share results between machines, share the checkpoint (`*.pt`, stored with Git LFS: run `git lfs install` once), not your data.

## Train the face model

The Evidence tab's **Faces** check uses a classifier that judges whether each face is a real photograph or AI-generated. A trained checkpoint ships with the repository; retrain it only if you have face images of your own to add. Training takes two folders of face images:

```bash
python -m image_detector.face_training --real "<dir of real face photos>" --ai "<dir of AI-generated face images>" --epochs 6
```

It reads the files in place, holds out about 20 % by file hash, and writes `image_detector/models/face_authenticity.pt` (about 43 MB, stored with Git LFS). Training takes roughly 25 minutes on a laptop CPU. The report prints accuracy on held-out images, on held-out images degraded the same way for both classes, and through the exact inference path.

How it affects results: a face scored 90 % or more AI-like adds a small, capped amount of evidence toward AI and stops the verdict from saying "likely real"; when a single face is the main subject (at least 15 % of the picture) and is scored 95 % or more AI-like, the face classifier leads the AI score (65 % face, 35 % pixel detector). A face that looks real adds nothing, because a generator the model has never seen would also look real to it.

## Measure real accuracy

After registering labelled media:

```bash
python -m services.calibration_cli --modality image --out image_report.json
```

It scores the held-out validation files and reports accuracy, false-positive rate, Brier score, expected calibration error and band occupancy. With no labelled library it says so and reports nothing. Use at least a few hundred files per class before believing the numbers.

## Optional local files

Nothing here ships; add them only if you want the matching check to become active. All are git-ignored.

| Purpose | How |
|---|---|
| Hard-block list (SHA-256 of files you never want processed) | One hash per line in `core/data/hardblock_sha256.txt`, or set `OMNI_HARDBLOCK_SHA256_FILE`. A match stops analysis. |
| Image re-use index | JSON lines `{"phash": "<16 hex>", "label": "...", "source": "..."}` in `image_detector/data/context_hash_index.jsonl`, or set `OMNI_CONTEXT_HASH_INDEX`. |
| Audio / video fingerprint index | Set `OMNI_AUDIO_FP_INDEX` / `OMNI_VIDEO_FP_INDEX` (JSON lines; see `dimension_checks/context.py` in each package). |
| Out-of-distribution gate | After registering labelled media: `python -m image_detector.dimension_checks.fit_ood` (same for `audio_detector`, `video_detector`). Until fitted, the gate reports "not calibrated" instead of guessing. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Video or non-WAV audio is skipped or limited | Install `ffmpeg` and make sure it is on `PATH`. The sidebar System status shows whether it is found. |
| First analysis is slow | Models load on first use and are cached afterwards. |
| A link will not download | Only direct `http`/`https` links on ports 80/443 to public addresses are allowed, and HTML pages are rejected. |
| Everything says "uncalibrated" | Expected on a fresh project. See [Measure real accuracy](#measure-real-accuracy). |
| "... is a Git LFS pointer, not the model" in the log, or the face check / age screening is unavailable | The repository was cloned without Git LFS. Run `git lfs install` then `git lfs pull` in the repository. |
| "Minor screening did not run" | The age weights could not be loaded (see the row above) or `torch`/`timm` is missing; the sidebar System status row "Minor screening (age model)" says which. |
| Windows log shows `WinError 10054` | Harmless browser disconnect; it is filtered out of the log. |
