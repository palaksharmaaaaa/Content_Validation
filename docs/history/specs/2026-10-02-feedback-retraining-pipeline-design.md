> **SUPERSEDED (2026-10-02):** this describes the earlier copy-into-a-queue design. The implemented design reads media in place by content hash; see `core/media_library.py`, `core/retrain_engine.py`, `<modality>_detector/retrain.py` and the README section "Train on your own media, in place".

# Feedback-driven retraining pipeline

**Status:** Approved — proceeding to implementation.
**Date:** 2026-10-02

## Context

The product lets a user upload or URL-fetch images/video/audio, runs a forensic AI-detection pipeline, and collects user feedback (a "this verdict was wrong" correction) through `ui/feedback_ui.py`'s rating widgets, which call each modality's `*SelfImprover.record_feedback()`.

Today, `record_feedback()` only nudges a handful of hand-tuned scoring-weight constants in a calibration JSON file (see each package's `learner.py`, and the project README's "What this project actually is (and isn't)" section, which documents this honestly). It never touches the neural network's weights. The user wants the model itself to actually learn from corrected mistakes — a real retraining loop — while satisfying two hard constraints:

1. **Personal media (the images/audio/video the user uploads, including AI-generated and downloaded files) must never be committed to git.**
2. **Anyone who clones the repository and runs it must get the exact same detection results the user gets**, for any input file, regardless of what local data trained the model.

These two constraints are reconciled by a standard ML-engineering separation: raw training data stays local and gitignored; only the *trained artifact* (the checkpoint) and its *provenance metadata* (version history, accuracy, sample counts — never filenames) are shared, via Git LFS.

The project starts with no checkpoint and no data for any modality; neural-inference code paths stay inactive until a first checkpoint is trained. This pipeline is built once, identically, for all three; for audio/video it will initially behave as "build the first-ever checkpoint from scratch" rather than "fine-tune," which is a direct and accepted consequence of their current zero-data state, not a special case in the code.

## Decisions made during brainstorming

- **Feedback queues, doesn't retrain immediately.** Corrections accumulate in a local dataset; retraining is a distinct, deliberate step.
- **Retraining is auto-suggested, manually confirmed.** The UI shows "N corrections queued — Retrain now?" once a threshold is hit; nothing retrains without an explicit click (or CLI invocation).
- **Checkpoints are distributed via Git LFS.** `models/*.pt` moves from gitignored to LFS-tracked, so a clone + `git lfs pull` reproduces the user's exact results.
- **Retraining stays binary** (real vs. ai_generated), matching the existing model's capacity. The rule-based taxonomy (edited/composite/screenshot states) is untouched — out of scope for this work.
- **New checkpoints are validated before promotion, with automatic rollback.** A retrain only replaces the live checkpoint if its held-out validation accuracy is `>=` the current checkpoint's (or no checkpoint currently exists). Otherwise the candidate is discarded and the pending corrections stay queued.
- **Default auto-suggest threshold: 15 queued corrections per modality**, configurable via each package's `config.py`.

## Architecture

Three new pieces, built identically for `image_detector/`, `audio_detector/`, and `video_detector/`:

### 1. Corrections dataset (gitignored)

```
<modality>_detector/data/corrections/
├── pending_manifest.jsonl       # gitignored — lists {sha256, label, timestamp, original_ext}
├── pending/
│   ├── ai_generated/<sha256>.<ext>
│   └── real/<sha256>.<ext>
└── archive/
    ├── ai_generated/<sha256>.<ext>
    └── real/<sha256>.<ext>
```

`pending/` holds corrections not yet folded into a promoted checkpoint. `archive/` holds corrections that *have* been folded in (kept for reproducibility/debugging of a specific checkpoint version, and so the same file is never re-added to a future training run). Deduplication is by SHA-256 of file bytes — if the same file is corrected twice, the manifest entry is updated in place rather than duplicated.

### 2. `<modality>_detector/retrain.py` (new module)

Public functions:

- `count_pending() -> Dict[str, int]` — reads `pending_manifest.jsonl`, returns counts per label (`{"ai_generated": N, "real": M}`) plus the total. Used by both the UI banner and the CLI.
- `run_retrain(min_new: int = 1) -> RetrainResult` — the actual job. Raises/returns early if `count_pending()["total"] < min_new`. Steps:
  1. Load the current checkpoint if one exists (via the existing `*AIDetector.load()` / `build_*_classifier` path); if none exists, start from the architecture's pretrained-ImageNet initialization (same as `trainer.py` does today).
  2. Build the training set: existing `dataset/train` + `dataset/val` images, plus every file currently in `data/corrections/pending/`, stratified-split 80/20 by label (reusing each package's existing `*DetectorTrainer.prepare_data`-equivalent loader, extended to also read from `pending/`).
  3. Fine-tune for a small, fixed number of epochs (reuses the existing `*DetectorTrainer.train()` loop; default epoch count matches `trainer.py`'s current default for that modality) continuing from the loaded weights — not from scratch, unless step 1 found no checkpoint.
  4. Evaluate the resulting candidate checkpoint via the existing `*BenchmarkSuite.evaluate_dataset()` against the held-out validation split (existing `dataset/val` + the new 20% held out from `pending/`).
  5. Compare candidate accuracy to the current checkpoint's last-recorded accuracy (read from `CHECKPOINT_LOG.md`, or treated as 0.0 if no checkpoint/log exists yet).
     - **Candidate `>=` current:** promote — write the candidate to `models/<modality>_detector.pt` (atomic replace, reusing `core.atomic_io` patterns), move every folded-in file from `pending/` to `archive/`, update `pending_manifest.jsonl` (remove promoted entries), append one line to `CHECKPOINT_LOG.md`.
     - **Candidate `<` current:** discard the candidate checkpoint file, leave `pending/` and the manifest untouched, return a result indicating rollback with both accuracy numbers so the caller can show *why*.
  6. Returns a `RetrainResult` dataclass: `promoted: bool`, `old_accuracy: Optional[float]`, `new_accuracy: float`, `samples_folded_in: int`, `cumulative_samples: int`, `message: str`.
- CLI entrypoint (`if __name__ == "__main__"`): `python -m image_detector.retrain` / `python -m audio_detector.retrain` / `python -m video_detector.retrain`, printing a human-readable summary of the `RetrainResult`.

### 3. `models/CHECKPOINT_LOG.md` (git-tracked, per modality)

A plain Markdown table, append-only, committed normally (small, text-only, no personal data):

```markdown
| Version | Date | Train Acc | Val Acc | New Samples | Cumulative Samples |
|---|---|---|---|---|---|
| 1 | (date) | (train acc) | (val acc) | (n) | (n) |
| 2 | (date) | (train acc) | (val acc) | (n) | (cumulative) |
```

"Version" is a simple incrementing integer per modality, stored alongside the log. No filenames, no paths, no image content — just the numbers needed to audit how the live checkpoint got to where it is.

### 4. Feedback-capture extension (modifies existing `learner.py` files)

`*SelfImprover.record_feedback(...)` gains two new side effects, added *after* its existing calibration-nudge logic (which is unchanged):

1. Compute SHA-256 of the submitted file's bytes; copy it into `data/corrections/pending/<ai_generated|real>/<sha256>.<ext>` (skip if that hash already exists in `pending/` or `archive/` — already queued or already trained on).
2. Append/update `{sha256, label, timestamp, original_ext}` in `pending_manifest.jsonl` via `core.atomic_io` (same atomic-write pattern already used for calibration persistence).

The existing `record_feedback` signature and return value are unchanged — this is purely additive inside the function body. Binary label mapping: for image_detector, a correction's `user_label` (`"AI"`/`"REAL"`) maps directly to `ai_generated`/`real`; corrections submitted against richer taxonomy states (e.g., "this is AI_ENHANCED_COMPOSITE") collapse to `ai_generated` for training purposes, consistent with the "stay binary" decision.

### 5. UI integration (`ui/feedback_ui.py`)

`render_learning_dashboard()` (the existing "Continuous Learning & Memory" tab) gains, per modality sub-tab: a call to `<modality>_detector.retrain.count_pending()`, and — only when the total is `>= RETRAIN_SUGGEST_THRESHOLD` (new `config.py` constant, default 15) — a `st.info` banner with a "Retrain Now" button that calls `run_retrain()` and renders the `RetrainResult` (promoted/rolled-back, before/after accuracy, samples folded in).

### 6. Git / distribution changes

- `.gitattributes` (new, repo root): `*.pt filter=lfs diff=lfs merge=lfs -text`.
- `.gitignore`: remove the blanket model-weight-extension exclusion for `models/*.pt` specifically (keep it for any *other* stray weight files elsewhere); add `**/data/corrections/pending/` and `**/data/corrections/archive/` and `**/data/corrections/pending_manifest.jsonl` to the ignore list explicitly (the existing broad `**/data/` media-extension rules likely already cover the media files themselves, but the manifest `.jsonl` needs its own rule since `.jsonl` isn't covered by the existing image/video/audio extension blocklist).
- `README.md`: new subsection under "Data, models, and what's actually on disk vs. git" documenting `git lfs install && git lfs pull`, and a new subsection describing the retraining workflow end-to-end (supersedes the "What this project actually is" paragraph that currently says feedback never touches model weights — that paragraph gets corrected to describe the new real mechanism once it exists, while still being honest that calibration-only nudging is what happens *between* retrains).

## Error handling

- `run_retrain()` with no corrections queued and no existing checkpoint: returns a `RetrainResult` with `promoted=False` and a clear message ("Nothing to train on") rather than crashing or silently training on an empty set.
- A corrupted/unreadable file in `pending/` (e.g., truncated download) is skipped with a logged warning during dataset loading, not a hard failure of the whole retrain — reuses each package's existing PIL/decode-error handling pattern from `trainer.py`.
- If writing the new checkpoint fails partway (disk full, permission error), the live checkpoint is never touched until the full candidate file is written and validated — no window where `models/*.pt` is a partial/corrupt file. Same atomic-replace discipline already used by `core.atomic_io`.
- If Git LFS isn't installed/configured when someone clones, `git lfs pull` fails loudly with LFS's own standard error — the README calls this out explicitly as a prerequisite, and `*AIDetector.load()`'s existing graceful fallback (log a warning, run in pure-heuristic mode) means a missing checkpoint degrades rather than crashes the app.

## Testing

- `test_retrain.py` per modality (new, under `<modality>_detector/tests/`): using tiny synthetic images/audio/video and `epochs=1`, verifies: `count_pending()` reflects a manually-seeded `pending/` + manifest; a retrain with a deliberately-worse candidate (e.g. corrupting validation labels) correctly rolls back and leaves `pending/` untouched; a retrain with a clearly-better/only candidate correctly promotes, moves files to `archive/`, and appends to `CHECKPOINT_LOG.md`.
- `test_learner_corrections.py` (extends each existing `test_*_detector.py`'s `test_learner` case): verifies `record_feedback()` writes the expected file into `pending/<label>/` and updates the manifest, and that submitting the same file twice does not duplicate it.
- No existing test is modified in a way that changes its assertions — this is additive test coverage alongside the existing calibration-nudge tests.

## Explicitly out of scope

- Multi-class taxonomy retraining (stays binary, per decision above).
- Automatic (non-confirmed) retraining — always requires an explicit click or CLI invocation.
- Any change to the rule-based `scoring.py` taxonomy decision trees in any package.
- Building the actual initial audio/video datasets — this spec builds the *mechanism*; populating `dataset/` or `data/corrections/` with real audio/video samples is the user's own subsequent data-collection work, same as it is for continuing to grow image's dataset.
