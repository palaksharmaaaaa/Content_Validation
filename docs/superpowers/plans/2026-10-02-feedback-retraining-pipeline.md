> **SUPERSEDED (2026-10-02):** this describes the earlier copy-into-a-queue design. The implemented design reads media in place by content hash; see `core/media_library.py`, `core/retrain_engine.py`, `<modality>_detector/retrain.py` and the README section "Train on your own media, in place".

# Feedback-Driven Retraining Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the loop between user feedback and the actual neural checkpoint — today `record_feedback()` only nudges heuristic scoring constants; this plan makes it queue corrections, and adds a `retrain.py` per modality that fine-tunes, validates, and safely promotes (or rolls back) a new checkpoint, with Git LFS distribution so a fresh clone reproduces the same results without ever receiving personal media.

**Architecture:** A new shared `core/corrections_store.py` module provides modality-agnostic file-bookkeeping (queue/dedupe/promote corrections, read/write a `CHECKPOINT_LOG.md`). Each of `image_detector/`, `audio_detector/`, `video_detector/` gets: a small extension to its existing `trainer.py` (accept extra in-memory samples alongside the on-disk dataset), a small extension to its existing `learner.py` (`record_feedback()` also queues the file), and a new `retrain.py` (fine-tune → validate → promote-or-rollback). `models/*.pt` moves from gitignored to Git-LFS-tracked.

**Tech Stack:** Python, PyTorch/torchvision (existing), `core.atomic_io` (existing, reused for the manifest), stdlib `hashlib`/`pathlib`/`dataclasses`, Git LFS.

## Global Constraints

- Retraining stays **binary** (`ai_generated`=0 / `real`=1) — matches every existing checkpoint's capacity exactly. No multi-class taxonomy work in this plan.
- Personal media must never be written anywhere git-tracked. `data/corrections/{pending,archive}/` is already covered by the repo's existing blanket `data/` gitignore rule (verified: `git check-ignore -v image_detector/data/corrections/pending_manifest.jsonl` → matches `.gitignore:93:data/`) — no new gitignore entries are needed for it.
- `models/*.pt` is currently blanket-ignored by `.gitignore`'s `*.pt` rule (`.gitignore:48`) — this plan adds a negation so it's tracked (via Git LFS) instead.
- A retrain only replaces the live checkpoint if candidate validation accuracy `>=` the current checkpoint's last-logged accuracy, or unconditionally if no checkpoint has ever been promoted (`CHECKPOINT_LOG.md` has no rows yet). On rollback: the candidate file is deleted, the pending queue is left untouched, nothing about the live checkpoint changes.
- Every new/modified function with a default parameter must preserve prior behavior exactly when called the old way (e.g. `prepare_data(dataset_dir)` without `extra_samples` must behave identically to before this plan).
- Validation accuracy is read directly from each trainer's own `history` return value (`val_acc`/`val_accuracy`, whichever that package already uses) — never fabricated, never computed from an unrelated quantity like training loss.
- Full test suite (`pytest`) must pass after every task, not just at the end.

---

### Task 1: `core/corrections_store.py` — shared feedback-queue and checkpoint-log bookkeeping

**Files:**
- Create: `core/corrections_store.py`
- Test: `core/tests/test_corrections_store.py`

**Interfaces:**
- Consumes: `core.atomic_io.atomic_read_json`, `core.atomic_io.atomic_write_json` (existing, signatures: `atomic_read_json(file_path, default=None) -> Any`, `atomic_write_json(file_path, data, indent=2) -> None`).
- Produces (used by every later task):
  - `compute_sha256(data: bytes) -> str`
  - `queue_correction(corrections_dir: str | Path, file_bytes: bytes, label: str, original_ext: str) -> bool`
  - `count_pending(corrections_dir: str | Path) -> Dict[str, int]` (keys: `"ai_generated"`, `"real"`, `"total"`)
  - `list_pending(corrections_dir: str | Path, label: str) -> List[Path]`
  - `promote_pending_to_archive(corrections_dir: str | Path, shas: List[str]) -> int`
  - `append_checkpoint_log(log_path, version: int, date: str, train_loss: float, val_acc: float, new_samples: int, cumulative_samples: int) -> None`
  - `read_last_checkpoint_accuracy(log_path: str | Path) -> Optional[float]`
  - `next_checkpoint_version(log_path: str | Path) -> int`
  - `read_cumulative_samples(log_path: str | Path) -> int`
  - `@dataclass RetrainResult`: fields `promoted: bool`, `new_accuracy: float`, `samples_folded_in: int`, `cumulative_samples: int`, `message: str`, `old_accuracy: Optional[float] = None`

- [ ] **Step 1: Write the failing test**

Create `core/tests/test_corrections_store.py`:

```python
"""Tests for core.corrections_store -- the shared feedback-queue/checkpoint-log bookkeeping."""
import tempfile
import unittest
from pathlib import Path

from core.corrections_store import (
    append_checkpoint_log,
    compute_sha256,
    count_pending,
    list_pending,
    next_checkpoint_version,
    promote_pending_to_archive,
    queue_correction,
    read_cumulative_samples,
    read_last_checkpoint_accuracy,
)


class TestCorrectionsStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corrections_dir = Path(self.tmp.name) / "corrections"

    def tearDown(self):
        self.tmp.cleanup()

    def test_queue_correction_writes_file_and_manifest(self):
        changed = queue_correction(self.corrections_dir, b"fake image bytes", "ai_generated", ".jpg")
        self.assertTrue(changed)
        counts = count_pending(self.corrections_dir)
        self.assertEqual(counts, {"ai_generated": 1, "real": 0, "total": 1})
        pending_files = list_pending(self.corrections_dir, "ai_generated")
        self.assertEqual(len(pending_files), 1)
        self.assertEqual(pending_files[0].read_bytes(), b"fake image bytes")

    def test_duplicate_submission_is_a_noop(self):
        queue_correction(self.corrections_dir, b"same bytes", "real", ".png")
        changed = queue_correction(self.corrections_dir, b"same bytes", "real", ".png")
        self.assertFalse(changed)
        self.assertEqual(count_pending(self.corrections_dir)["total"], 1)

    def test_relabeling_moves_file_between_label_folders(self):
        queue_correction(self.corrections_dir, b"flip-flop bytes", "real", ".png")
        changed = queue_correction(self.corrections_dir, b"flip-flop bytes", "ai_generated", ".png")
        self.assertTrue(changed)
        counts = count_pending(self.corrections_dir)
        self.assertEqual(counts, {"ai_generated": 1, "real": 0, "total": 1})
        sha = compute_sha256(b"flip-flop bytes")
        self.assertFalse((self.corrections_dir / "pending" / "real" / f"{sha}.png").exists())
        self.assertTrue((self.corrections_dir / "pending" / "ai_generated" / f"{sha}.png").exists())

    def test_invalid_label_raises(self):
        with self.assertRaises(ValueError):
            queue_correction(self.corrections_dir, b"x", "not_a_label", ".jpg")

    def test_promote_pending_to_archive_moves_files(self):
        queue_correction(self.corrections_dir, b"sample a", "ai_generated", ".jpg")
        queue_correction(self.corrections_dir, b"sample b", "real", ".jpg")
        sha_a = compute_sha256(b"sample a")
        sha_b = compute_sha256(b"sample b")
        moved = promote_pending_to_archive(self.corrections_dir, [sha_a, sha_b])
        self.assertEqual(moved, 2)
        self.assertEqual(count_pending(self.corrections_dir)["total"], 0)
        self.assertTrue((self.corrections_dir / "archive" / "ai_generated" / f"{sha_a}.jpg").is_file())
        self.assertTrue((self.corrections_dir / "archive" / "real" / f"{sha_b}.jpg").is_file())

    def test_checkpoint_log_round_trip(self):
        log_path = Path(self.tmp.name) / "CHECKPOINT_LOG.md"
        self.assertIsNone(read_last_checkpoint_accuracy(log_path))
        self.assertEqual(next_checkpoint_version(log_path), 1)
        self.assertEqual(read_cumulative_samples(log_path), 0)

        append_checkpoint_log(log_path, version=1, date="2026-10-02", train_loss=0.3, val_acc=0.94, new_samples=203, cumulative_samples=203)
        self.assertAlmostEqual(read_last_checkpoint_accuracy(log_path), 0.94)
        self.assertEqual(next_checkpoint_version(log_path), 2)
        self.assertEqual(read_cumulative_samples(log_path), 203)

        append_checkpoint_log(log_path, version=2, date="2026-10-15", train_loss=0.2, val_acc=0.96, new_samples=18, cumulative_samples=221)
        self.assertAlmostEqual(read_last_checkpoint_accuracy(log_path), 0.96)
        self.assertEqual(next_checkpoint_version(log_path), 3)
        self.assertEqual(read_cumulative_samples(log_path), 221)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest core/tests/test_corrections_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.corrections_store'`

- [ ] **Step 3: Write the implementation**

Create `core/corrections_store.py`:

```python
"""
core.corrections_store: Shared, modality-agnostic file bookkeeping for the
feedback -> retrain pipeline (queue/dedupe/promote user-corrected media, and
read/write a checkpoint provenance log). Pure path/file logic -- no imports
from image_detector/audio_detector/video_detector, preserving core/'s
one-directional dependency (core never imports from a detector package).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.atomic_io import atomic_read_json, atomic_write_json

_VALID_LABELS = ("ai_generated", "real")
_LOG_HEADER = "| Version | Date | Train Loss | Val Accuracy | New Samples | Cumulative Samples |\n|---|---|---|---|---|---|\n"


@dataclass
class RetrainResult:
    """Outcome of one retrain.py run_retrain() call."""
    promoted: bool
    new_accuracy: float
    samples_folded_in: int
    cumulative_samples: int
    message: str
    old_accuracy: Optional[float] = None


def compute_sha256(data: bytes) -> str:
    """Returns the hex SHA-256 digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def _manifest_path(corrections_dir: Path) -> Path:
    return corrections_dir / "pending_manifest.json"


def _read_manifest(corrections_dir: Path) -> List[Dict[str, Any]]:
    loaded = atomic_read_json(_manifest_path(corrections_dir), default=[])
    return loaded if isinstance(loaded, list) else []


def _write_manifest(corrections_dir: Path, entries: List[Dict[str, Any]]) -> None:
    atomic_write_json(_manifest_path(corrections_dir), entries, indent=2)


def queue_correction(
    corrections_dir: str | Path,
    file_bytes: bytes,
    label: str,
    original_ext: str,
) -> bool:
    """
    Queues a user-corrected file for the next retrain. `label` must be
    "ai_generated" or "real". Deduplicates by SHA-256 of file content:
    re-submitting the same bytes with the same label is a no-op; re-submitting
    with a DIFFERENT label moves the queued entry between label folders rather
    than creating a duplicate. Returns True if the pending queue changed, False
    if this was a no-op (identical resubmission).
    """
    if label not in _VALID_LABELS:
        raise ValueError(f"label must be one of {_VALID_LABELS}, got {label!r}")

    corrections_dir = Path(corrections_dir)
    sha = compute_sha256(file_bytes)
    ext = original_ext if original_ext.startswith(".") else f".{original_ext}"

    entries = _read_manifest(corrections_dir)
    existing = next((e for e in entries if e.get("sha256") == sha), None)

    if existing is not None and existing.get("status") == "pending" and existing.get("label") == label:
        return False  # identical resubmission, nothing to do

    if existing is not None and existing.get("status") == "pending" and existing.get("label") != label:
        old_path = corrections_dir / "pending" / existing["label"] / f"{sha}{existing['original_ext']}"
        old_path.unlink(missing_ok=True)

    target_dir = corrections_dir / "pending" / label
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{sha}{ext}").write_bytes(file_bytes)

    new_entry = {
        "sha256": sha,
        "label": label,
        "original_ext": ext,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
    }
    entries = [e for e in entries if e.get("sha256") != sha] + [new_entry]
    _write_manifest(corrections_dir, entries)
    return True


def count_pending(corrections_dir: str | Path) -> Dict[str, int]:
    """Returns {"ai_generated": N, "real": M, "total": N+M} currently queued."""
    entries = _read_manifest(Path(corrections_dir))
    pending = [e for e in entries if e.get("status") == "pending"]
    ai = sum(1 for e in pending if e.get("label") == "ai_generated")
    real = sum(1 for e in pending if e.get("label") == "real")
    return {"ai_generated": ai, "real": real, "total": ai + real}


def list_pending(corrections_dir: str | Path, label: str) -> List[Path]:
    """Returns the on-disk paths of every pending correction for the given label."""
    corrections_dir = Path(corrections_dir)
    entries = _read_manifest(corrections_dir)
    paths: List[Path] = []
    for e in entries:
        if e.get("status") == "pending" and e.get("label") == label:
            p = corrections_dir / "pending" / label / f"{e['sha256']}{e['original_ext']}"
            if p.is_file():
                paths.append(p)
    return paths


def promote_pending_to_archive(corrections_dir: str | Path, shas: List[str]) -> int:
    """
    Moves the given sha256 hashes from pending/ to archive/ and marks them
    archived in the manifest. Entries not currently pending (unknown hash, or
    already archived) are silently skipped. Returns the number actually moved.
    """
    corrections_dir = Path(corrections_dir)
    entries = _read_manifest(corrections_dir)
    by_sha = {e.get("sha256"): e for e in entries}
    moved = 0
    for sha in shas:
        e = by_sha.get(sha)
        if e is None or e.get("status") != "pending":
            continue
        label = e["label"]
        ext = e["original_ext"]
        src = corrections_dir / "pending" / label / f"{sha}{ext}"
        if src.is_file():
            dst_dir = corrections_dir / "archive" / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            src.replace(dst_dir / f"{sha}{ext}")
            moved += 1
        e["status"] = "archived"
    _write_manifest(corrections_dir, entries)
    return moved


def _log_rows(log_path: Path) -> List[List[str]]:
    if not log_path.is_file():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or "Version" in line:
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def read_last_checkpoint_accuracy(log_path: str | Path) -> Optional[float]:
    """Returns the Val Accuracy of the last row in the checkpoint log, or None if
    the log doesn't exist or has no rows yet (no checkpoint has ever been promoted)."""
    rows = _log_rows(Path(log_path))
    if not rows:
        return None
    try:
        return float(rows[-1][3])
    except (IndexError, ValueError):
        return None


def next_checkpoint_version(log_path: str | Path) -> int:
    """Returns the version number the NEXT promoted checkpoint should use."""
    rows = _log_rows(Path(log_path))
    if not rows:
        return 1
    try:
        return int(rows[-1][0]) + 1
    except (IndexError, ValueError):
        return len(rows) + 1


def read_cumulative_samples(log_path: str | Path) -> int:
    """Returns the sum of the "New Samples" column across every logged checkpoint."""
    total = 0
    for row in _log_rows(Path(log_path)):
        try:
            total += int(row[4])
        except (IndexError, ValueError):
            continue
    return total


def append_checkpoint_log(
    log_path: str | Path,
    version: int,
    date: str,
    train_loss: float,
    val_acc: float,
    new_samples: int,
    cumulative_samples: int,
) -> None:
    """
    Appends one row to the (git-tracked, plain-text, no personal data) checkpoint
    provenance log: version, date, final training loss, held-out validation
    accuracy, and sample counts. Creates the file with a header if it doesn't exist.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.is_file():
        log_path.write_text(_LOG_HEADER, encoding="utf-8")
    row = f"| {version} | {date} | {train_loss:.4f} | {val_acc:.4f} | {new_samples} | {cumulative_samples} |\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(row)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest core/tests/test_corrections_store.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Run the full suite to confirm no regressions, then commit**

Run: `pytest`
Expected: all previously-passing tests still PASS, plus the 7 new ones.

```bash
git add core/corrections_store.py core/tests/test_corrections_store.py
git commit -m "feat(core): add shared corrections-queue and checkpoint-log bookkeeping"
```

---

### Task 2: `image_detector` — config additions + `trainer.py` extra-samples support

**Files:**
- Modify: `image_detector/config.py:15-17`
- Modify: `image_detector/trainer.py:70-134` (the `prepare_data` method)
- Test: `image_detector/tests/test_image_detector.py` (new test method)

**Interfaces:**
- Consumes: nothing new.
- Produces: `image_detector.config.CORRECTIONS_DIR`, `image_detector.config.CHECKPOINT_LOG_FILE`, `image_detector.config.RETRAIN_SUGGEST_THRESHOLD`; `ImageDetectorTrainer.prepare_data(..., extra_samples: Optional[List[Tuple[Path, int]]] = None)`.

- [ ] **Step 1: Add config constants**

In `image_detector/config.py`, after line 17 (`CALIBRATION_FILE = DATA_DIR / "image_calibration.json"`), add:

```python
CORRECTIONS_DIR = DATA_DIR / "corrections"
CHECKPOINT_LOG_FILE = MODELS_DIR / "CHECKPOINT_LOG.md"
RETRAIN_SUGGEST_THRESHOLD = 15
```

- [ ] **Step 2: Write the failing test for `prepare_data`'s new parameter**

Add to `image_detector/tests/test_image_detector.py` (inside `TestImageDetector`, near `test_learner`):

```python
    def test_prepare_data_merges_extra_samples(self):
        from image_detector.trainer import ImageDetectorTrainer

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_ai = tmp_path / "train" / "ai_generated"
            train_real = tmp_path / "train" / "real"
            train_ai.mkdir(parents=True)
            train_real.mkdir(parents=True)
            for i in range(2):
                cv2.imwrite(str(train_ai / f"ai_{i}.jpg"), np.full((32, 32, 3), 10, dtype=np.uint8))
                cv2.imwrite(str(train_real / f"real_{i}.jpg"), np.full((32, 32, 3), 240, dtype=np.uint8))

            extra_img = tmp_path / "extra.jpg"
            cv2.imwrite(str(extra_img), np.full((32, 32, 3), 10, dtype=np.uint8))

            trainer = ImageDetectorTrainer()
            train_loader, val_loader = trainer.prepare_data(
                tmp_path, extra_samples=[(extra_img, 0)], batch_size=2,
            )
            total_samples = len(train_loader.dataset) + (len(val_loader.dataset) if val_loader else 0)
            self.assertEqual(total_samples, 5)  # 4 on-disk + 1 extra
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest image_detector/tests/test_image_detector.py::TestImageDetector::test_prepare_data_merges_extra_samples -v`
Expected: FAIL with `TypeError: prepare_data() got an unexpected keyword argument 'extra_samples'`

- [ ] **Step 4: Implement — replace `prepare_data`'s body**

In `image_detector/trainer.py`, replace the entire `prepare_data` method (lines 70-134) with:

```python
    def prepare_data(
        self,
        dataset_dir: Path | str,
        batch_size: int = 16,
        val_split: float = 0.2,
        extra_samples: Optional[List[Tuple[Path, int]]] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Scans dataset directory formatted with 'ai_generated' and 'real' subdirectories,
        or pre-split 'train' and 'val' subdirectories. If `extra_samples` is given (a list
        of (path, label) tuples, label 0=ai_generated/1=real), those are split 80/20 (or
        per `val_split`) with the same seeded shuffle and merged into the resulting
        train/val sample lists -- used by retrain.py to fold in user corrections without
        needing them on disk in the same ai_generated/real directory layout.
        """
        dataset_path = Path(dataset_dir)

        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Check for pre-split structure (train/ and val/)
        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"
        if train_dir.exists() and (train_dir / "ai_generated").exists():
            train_ai = [p for p in (train_dir / "ai_generated").rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
            train_real = [p for p in (train_dir / "real").rglob("*") if (train_dir / "real").exists() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
            train_samples = [(p, 0) for p in train_ai] + [(p, 1) for p in train_real]

            val_samples = []
            if val_dir.exists():
                val_ai = [p for p in (val_dir / "ai_generated").rglob("*") if (val_dir / "ai_generated").exists() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
                val_real = [p for p in (val_dir / "real").rglob("*") if (val_dir / "real").exists() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
                val_samples = [(p, 0) for p in val_ai] + [(p, 1) for p in val_real]

            random.seed(42)
            random.shuffle(train_samples)
            if val_samples:
                random.shuffle(val_samples)
        else:
            ai_dir = dataset_path / "ai_generated"
            real_dir = dataset_path / "real"

            ai_files = [p for p in ai_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if ai_dir.exists() else []
            real_files = [p for p in real_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if real_dir.exists() else []

            all_samples = [(p, 0) for p in ai_files] + [(p, 1) for p in real_files]

            random.seed(42)
            random.shuffle(all_samples)

            split_idx = int(len(all_samples) * (1.0 - val_split))
            train_samples = all_samples[:split_idx]
            val_samples = all_samples[split_idx:]

        if extra_samples:
            extra = list(extra_samples)
            random.seed(42)
            random.shuffle(extra)
            extra_split_idx = int(len(extra) * (1.0 - val_split))
            train_samples = train_samples + extra[:extra_split_idx]
            val_samples = val_samples + extra[extra_split_idx:]
            random.seed(42)
            random.shuffle(train_samples)

        if not train_samples and not val_samples:
            raise ValueError(f"No valid image files found in {dataset_path}/ai_generated, {dataset_path}/real, or extra_samples")

        train_loader = DataLoader(ImageDataset(train_samples, train_transform), batch_size=batch_size, shuffle=True) if train_samples else None
        val_loader = DataLoader(ImageDataset(val_samples, val_transform), batch_size=batch_size, shuffle=False) if val_samples else None

        logger.info(
            "Prepared Image Dataset: %d train, %d val samples (%d extra from corrections).",
            len(train_samples), len(val_samples), len(extra_samples or []),
        )
        return train_loader, val_loader
```

(This removes the old `raise ValueError(...)` that fired immediately after building `all_samples` in the flat-directory branch — it's replaced by the single check at the end that also accounts for `extra_samples`, so a dataset with zero on-disk files but nonzero `extra_samples` no longer raises. Every other line of behavior is unchanged from the original.)

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest image_detector/tests/test_image_detector.py::TestImageDetector::test_prepare_data_merges_extra_samples -v`
Expected: PASS

- [ ] **Step 6: Run the full suite to confirm no regressions, then commit**

Run: `pytest`
Expected: all tests PASS, including the pre-existing `test_pipeline_end_to_end`, `test_batch_processor`, `TestImageDetectorRealSamples` (unaffected — they don't call `prepare_data`).

```bash
git add image_detector/config.py image_detector/trainer.py image_detector/tests/test_image_detector.py
git commit -m "feat(image_detector): support folding extra in-memory samples into prepare_data"
```

---

### Task 3: `image_detector` — `learner.py` queues corrections

**Files:**
- Modify: `image_detector/learner.py:1-50` (imports + `__init__`), and the `record_feedback` body around lines 130-136
- Test: `image_detector/tests/test_image_detector.py` (new test method)

**Interfaces:**
- Consumes: `core.corrections_store.queue_correction(corrections_dir, file_bytes, label, original_ext) -> bool` (Task 1).
- Produces: `ImageSelfImprover(..., corrections_dir: Optional[Path] = None)`; `self.corrections_dir`.

- [ ] **Step 1: Write the failing test**

Add to `image_detector/tests/test_image_detector.py` (inside `TestImageDetector`):

```python
    def test_learner_queues_correction_for_retraining(self):
        from core.corrections_store import count_pending

        corrections_dir = self.temp_path / "corrections2"
        learner = ImageSelfImprover(
            memory_file=self.temp_path / "memory2.json",
            calibration_file=self.temp_path / "calib2.json",
            corrections_dir=corrections_dir,
        )
        learner.record_feedback(str(self.img_file), "AI", {"noise_residual_mean": 1.0})
        self.assertEqual(count_pending(corrections_dir), {"ai_generated": 1, "real": 0, "total": 1})

        # Submitting the exact same file+label again must not duplicate the queue entry.
        learner.record_feedback(str(self.img_file), "AI", {"noise_residual_mean": 1.0})
        self.assertEqual(count_pending(corrections_dir)["total"], 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest image_detector/tests/test_image_detector.py::TestImageDetector::test_learner_queues_correction_for_retraining -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'corrections_dir'`

- [ ] **Step 3: Implement**

In `image_detector/learner.py`, change the imports block (lines 27-29) to:

```python
from image_detector.config import CALIBRATION_FILE, CORRECTIONS_DIR, DATA_DIR, MEMORY_FILE
from image_detector.schemas import ImageFeedbackRecord
from core.atomic_io import atomic_read_json, atomic_write_json
from core.corrections_store import queue_correction
```

Change `__init__` (lines 41-50) to:

```python
    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        memory_file: Optional[Path] = None,
        calibration_file: Optional[Path] = None,
        corrections_dir: Optional[Path] = None,
    ):
        self.memory_dir = memory_dir or DATA_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = Path(memory_file) if memory_file else (self.memory_dir / "image_feedback.json")
        self.calibration_file = Path(calibration_file) if calibration_file else (self.memory_dir / "image_calibration.json")
        self.corrections_dir = Path(corrections_dir) if corrections_dir else CORRECTIONS_DIR
```

In `record_feedback`, find this existing block:

```python
        try:
            atomic_write_json(self.feedback_file, memory, indent=2)
        except Exception as e:
            logger.error("Failed to save image memory atomically: %s", e)

        # Dynamic recalibration based on new feedback
```

Replace it with:

```python
        try:
            atomic_write_json(self.feedback_file, memory, indent=2)
        except Exception as e:
            logger.error("Failed to save image memory atomically: %s", e)

        try:
            img_path_obj = Path(image_path)
            if img_path_obj.is_file():
                correction_label = "ai_generated" if user_label.upper() == "AI" else "real"
                queue_correction(
                    self.corrections_dir,
                    img_path_obj.read_bytes(),
                    correction_label,
                    img_path_obj.suffix or ".bin",
                )
        except Exception as e:
            logger.warning("Could not queue correction for retraining: %s", e)

        # Dynamic recalibration based on new feedback
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest image_detector/tests/test_image_detector.py::TestImageDetector::test_learner_queues_correction_for_retraining -v`
Expected: PASS

- [ ] **Step 5: Run the full suite to confirm no regressions, then commit**

Run: `pytest`
Expected: all tests PASS, including the existing `test_learner` (its call site never passes `corrections_dir`, so it defaults to the real `CORRECTIONS_DIR` and queues one correction as a side effect — this is harmless, since `data/corrections/` is gitignored and the queued file is throwaway test data under the real project path; verify this doesn't fail the test, only adds an untracked file).

```bash
git add image_detector/learner.py image_detector/tests/test_image_detector.py
git commit -m "feat(image_detector): queue user feedback corrections for retraining"
```

---

### Task 4: `image_detector/retrain.py` — fine-tune, validate, promote-or-rollback

**Files:**
- Create: `image_detector/retrain.py`
- Test: `image_detector/tests/test_retrain.py`

**Interfaces:**
- Consumes: `core.corrections_store.{RetrainResult, append_checkpoint_log, count_pending, list_pending, next_checkpoint_version, promote_pending_to_archive, read_cumulative_samples, read_last_checkpoint_accuracy}` (Task 1); `image_detector.trainer.ImageDetectorTrainer.prepare_data(..., extra_samples=...)` (Task 2); `image_detector.config.{CORRECTIONS_DIR, CHECKPOINT_LOG_FILE, DATASET_DIR, DEFAULT_CHECKPOINT}`.
- Produces: `count_pending_corrections() -> Dict[str, int]`; `run_retrain(min_new: int = 1, epochs: int = 5) -> RetrainResult` — used by Task 12 (UI).

- [ ] **Step 1: Write the failing tests**

Create `image_detector/tests/test_retrain.py`:

```python
"""Tests for image_detector.retrain -- the feedback-correction fine-tuning/promotion flow."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import image_detector.config as config
import image_detector.retrain as retrain
from core.corrections_store import append_checkpoint_log, queue_correction


def _make_image(path: Path, value: int) -> None:
    cv2.imwrite(str(path), np.full((64, 64, 3), value, dtype=np.uint8))


class TestImageRetrain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmp.name)

        self._orig = (config.DATASET_DIR, config.DEFAULT_CHECKPOINT, config.CORRECTIONS_DIR, config.CHECKPOINT_LOG_FILE)
        config.DATASET_DIR = tmp_path / "dataset"
        config.DEFAULT_CHECKPOINT = tmp_path / "models" / "ai_detector.pt"
        config.CORRECTIONS_DIR = tmp_path / "corrections"
        config.CHECKPOINT_LOG_FILE = tmp_path / "models" / "CHECKPOINT_LOG.md"
        retrain.DATASET_DIR = config.DATASET_DIR
        retrain.DEFAULT_CHECKPOINT = config.DEFAULT_CHECKPOINT
        retrain.CORRECTIONS_DIR = config.CORRECTIONS_DIR
        retrain.CHECKPOINT_LOG_FILE = config.CHECKPOINT_LOG_FILE

        train_ai = config.DATASET_DIR / "train" / "ai_generated"
        train_real = config.DATASET_DIR / "train" / "real"
        val_ai = config.DATASET_DIR / "val" / "ai_generated"
        val_real = config.DATASET_DIR / "val" / "real"
        for d in (train_ai, train_real, val_ai, val_real):
            d.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            _make_image(train_ai / f"ai_{i}.jpg", 20)
            _make_image(train_real / f"real_{i}.jpg", 220)
        _make_image(val_ai / "ai_val.jpg", 20)
        _make_image(val_real / "real_val.jpg", 220)

    def tearDown(self):
        config.DATASET_DIR, config.DEFAULT_CHECKPOINT, config.CORRECTIONS_DIR, config.CHECKPOINT_LOG_FILE = self._orig
        self.tmp.cleanup()

    def test_run_retrain_with_no_pending_corrections_is_a_noop(self):
        result = retrain.run_retrain(min_new=1)
        self.assertFalse(result.promoted)
        self.assertIn("Nothing to train on", result.message)

    def test_run_retrain_promotes_first_checkpoint(self):
        correction_img = Path(self.tmp.name) / "scratch_correction.jpg"
        _make_image(correction_img, 20)
        queue_correction(config.CORRECTIONS_DIR, correction_img.read_bytes(), "ai_generated", ".jpg")

        result = retrain.run_retrain(min_new=1, epochs=1)

        self.assertTrue(result.promoted)
        self.assertIsNone(result.old_accuracy)
        self.assertEqual(result.samples_folded_in, 1)
        self.assertTrue(config.DEFAULT_CHECKPOINT.is_file())
        self.assertTrue(config.CHECKPOINT_LOG_FILE.is_file())
        self.assertEqual(retrain.count_pending_corrections()["total"], 0)
        archived = list((config.CORRECTIONS_DIR / "archive" / "ai_generated").glob("*.jpg"))
        self.assertEqual(len(archived), 1)

    def test_run_retrain_rolls_back_a_worse_candidate(self):
        append_checkpoint_log(
            config.CHECKPOINT_LOG_FILE, version=1, date="2026-01-01",
            train_loss=0.1, val_acc=1.01, new_samples=1, cumulative_samples=1,
        )
        correction_img = Path(self.tmp.name) / "scratch_correction2.jpg"
        _make_image(correction_img, 220)
        queue_correction(config.CORRECTIONS_DIR, correction_img.read_bytes(), "real", ".jpg")

        result = retrain.run_retrain(min_new=1, epochs=1)

        self.assertFalse(result.promoted)
        self.assertAlmostEqual(result.old_accuracy, 1.01)
        self.assertFalse(config.DEFAULT_CHECKPOINT.is_file())
        self.assertEqual(retrain.count_pending_corrections()["total"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest image_detector/tests/test_retrain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_detector.retrain'`

- [ ] **Step 3: Implement**

Create `image_detector/retrain.py`:

```python
"""
image_detector.retrain: Folds queued user corrections into the training set, fine-tunes
the existing checkpoint (or trains from pretrained ImageNet weights if none exists yet),
and promotes the result to models/ai_detector.pt only if its held-out validation accuracy
is >= the current checkpoint's last-recorded accuracy. On rollback, the candidate is
discarded and every pending correction stays queued, untouched.
See docs/superpowers/specs/2026-10-02-feedback-retraining-pipeline-design.md.
"""
from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from core.corrections_store import (
    RetrainResult,
    append_checkpoint_log,
    count_pending,
    list_pending,
    next_checkpoint_version,
    promote_pending_to_archive,
    read_cumulative_samples,
    read_last_checkpoint_accuracy,
)
from image_detector.config import CHECKPOINT_LOG_FILE, CORRECTIONS_DIR, DATASET_DIR, DEFAULT_CHECKPOINT
from image_detector.trainer import ImageDetectorTrainer

logger = logging.getLogger("image_detector.retrain")


def count_pending_corrections() -> Dict[str, int]:
    """Returns {"ai_generated": N, "real": M, "total": N+M} corrections queued for the next retrain."""
    return count_pending(CORRECTIONS_DIR)


def run_retrain(min_new: int = 1, epochs: int = 5) -> RetrainResult:
    """
    Folds every pending correction into the training set, fine-tunes the current
    checkpoint, and promotes the result only if it validates at least as well as
    what's live. Never leaves a partial state: either the live checkpoint is fully
    replaced and the folded-in corrections move to archive/, or nothing changes.
    """
    pending = count_pending_corrections()
    if pending["total"] < min_new:
        return RetrainResult(
            promoted=False,
            new_accuracy=0.0,
            samples_folded_in=0,
            cumulative_samples=read_cumulative_samples(CHECKPOINT_LOG_FILE),
            message=f"Nothing to train on: {pending['total']} correction(s) queued, need at least {min_new}.",
        )

    extra_samples: List[Tuple[Path, int]] = []
    extra_samples += [(p, 0) for p in list_pending(CORRECTIONS_DIR, "ai_generated")]
    extra_samples += [(p, 1) for p in list_pending(CORRECTIONS_DIR, "real")]

    trainer = ImageDetectorTrainer(checkpoint_path=DEFAULT_CHECKPOINT)
    if DEFAULT_CHECKPOINT.is_file():
        checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.info("Continuing fine-tuning from existing checkpoint at %s", DEFAULT_CHECKPOINT)
    else:
        logger.info("No existing checkpoint found -- training from pretrained ImageNet weights.")

    train_loader, val_loader = trainer.prepare_data(DATASET_DIR, extra_samples=extra_samples)

    candidate_path = DEFAULT_CHECKPOINT.with_name(DEFAULT_CHECKPOINT.stem + ".candidate" + DEFAULT_CHECKPOINT.suffix)
    trainer.checkpoint_path = candidate_path
    history = trainer.train(train_loader, val_loader, epochs=epochs)

    new_accuracy = history["val_acc"][-1] if history.get("val_acc") else 0.0
    train_loss = history["train_loss"][-1] if history.get("train_loss") else 0.0
    old_accuracy = read_last_checkpoint_accuracy(CHECKPOINT_LOG_FILE)

    if old_accuracy is not None and new_accuracy < old_accuracy:
        candidate_path.unlink(missing_ok=True)
        return RetrainResult(
            promoted=False,
            old_accuracy=old_accuracy,
            new_accuracy=new_accuracy,
            samples_folded_in=0,
            cumulative_samples=read_cumulative_samples(CHECKPOINT_LOG_FILE),
            message=(
                f"Candidate checkpoint rolled back: validation accuracy {new_accuracy:.4f} "
                f"is below the current checkpoint's {old_accuracy:.4f}. "
                f"{pending['total']} correction(s) remain queued for the next attempt."
            ),
        )

    candidate_path.replace(DEFAULT_CHECKPOINT)
    promoted_shas = [p.stem for p in list_pending(CORRECTIONS_DIR, "ai_generated")]
    promoted_shas += [p.stem for p in list_pending(CORRECTIONS_DIR, "real")]
    promote_pending_to_archive(CORRECTIONS_DIR, promoted_shas)

    version = next_checkpoint_version(CHECKPOINT_LOG_FILE)
    cumulative = read_cumulative_samples(CHECKPOINT_LOG_FILE) + pending["total"]
    append_checkpoint_log(
        CHECKPOINT_LOG_FILE,
        version=version,
        date=date.today().isoformat(),
        train_loss=train_loss,
        val_acc=new_accuracy,
        new_samples=pending["total"],
        cumulative_samples=cumulative,
    )

    return RetrainResult(
        promoted=True,
        old_accuracy=old_accuracy,
        new_accuracy=new_accuracy,
        samples_folded_in=pending["total"],
        cumulative_samples=cumulative,
        message=(
            f"Promoted checkpoint v{version}: validation accuracy {new_accuracy:.4f} "
            f"({'up from ' + format(old_accuracy, '.4f') if old_accuracy is not None else 'first checkpoint'}). "
            f"{pending['total']} correction(s) folded in."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the image AI detector on queued user corrections")
    parser.add_argument("--min-new", type=int, default=1, help="Minimum queued corrections required to run")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs")
    args = parser.parse_args()

    result = run_retrain(min_new=args.min_new, epochs=args.epochs)
    print(result.message)
    if result.promoted:
        print(f"New checkpoint accuracy: {result.new_accuracy:.4f} (cumulative samples: {result.cumulative_samples})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest image_detector/tests/test_retrain.py -v`
Expected: PASS (3 tests). Note: this test builds a real ResNet18 via `build_image_classifier(pretrained=True)` — if torchvision's ImageNet weights aren't already cached locally, the first run will download them (requires network access once; subsequent runs use the cache). This matches the existing test suite's assumptions (e.g. `test_detector_predict` already constructs a real detector).

- [ ] **Step 5: Run the full suite to confirm no regressions, then commit**

Run: `pytest`
Expected: all tests PASS.

```bash
git add image_detector/retrain.py image_detector/tests/test_retrain.py
git commit -m "feat(image_detector): add retrain.py -- fine-tune, validate, promote-or-rollback"
```

---

### Task 5: `audio_detector` — config additions + `trainer.py` extra-samples support

**Files:**
- Modify: `audio_detector/config.py:15-17`
- Modify: `audio_detector/trainer.py:78-113` (the `prepare_data_from_directory` method)
- Test: `audio_detector/tests/test_audio_detector.py` (new test method)

**Interfaces:**
- Produces: `audio_detector.config.{CORRECTIONS_DIR, CHECKPOINT_LOG_FILE, RETRAIN_SUGGEST_THRESHOLD}`; `AudioDetectorTrainer.prepare_data_from_directory(..., extra_audio_samples: Optional[List[Tuple[Path, int]]] = None)`.

- [ ] **Step 1: Add config constants**

In `audio_detector/config.py`, after line 17 (`CALIBRATION_FILE = DATA_DIR / "audio_calibration.json"`), add:

```python
CORRECTIONS_DIR = DATA_DIR / "corrections"
CHECKPOINT_LOG_FILE = MODELS_DIR / "CHECKPOINT_LOG.md"
RETRAIN_SUGGEST_THRESHOLD = 15
```

- [ ] **Step 2: Write the failing test**

Add to `audio_detector/tests/test_audio_detector.py` (inside the test class, near `test_learner`). First check the file's imports include `wave` and `numpy as np`; if not already imported at the top, this step also needs `import wave` added alongside the existing imports.

```python
    def test_prepare_data_merges_extra_samples(self):
        from audio_detector.trainer import AudioDetectorTrainer

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ai_dir = tmp_path / "ai_generated"
            real_dir = tmp_path / "real"
            ai_dir.mkdir()
            real_dir.mkdir()

            def make_wav(path, freq):
                sr = 16000
                t = np.linspace(0, 1.0, sr, endpoint=False)
                samples = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
                with wave.open(str(path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(samples.tobytes())

            make_wav(ai_dir / "ai_0.wav", 440.0)
            make_wav(real_dir / "real_0.wav", 220.0)
            extra_wav = tmp_path / "extra.wav"
            make_wav(extra_wav, 440.0)

            trainer = AudioDetectorTrainer()
            X, y = trainer.prepare_data_from_directory(tmp_path, extra_audio_samples=[(extra_wav, 0)])
            self.assertEqual(len(X), 3)
            self.assertEqual(len(y), 3)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest audio_detector/tests/test_audio_detector.py::TestAudioDetector::test_prepare_data_merges_extra_samples -v`
Expected: FAIL with `TypeError: prepare_data_from_directory() got an unexpected keyword argument 'extra_audio_samples'`

- [ ] **Step 4: Implement**

In `audio_detector/trainer.py`, replace the `prepare_data_from_directory` method (lines 78-113) with:

```python
    def prepare_data_from_directory(
        self,
        dataset_dir: Path | str,
        extra_audio_samples: Optional[List[Tuple[Path, int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scans dataset with 'ai_generated' and 'real' subdirectories. `extra_audio_samples`
        (a list of (audio_path, label) tuples, label 0=ai_generated/1=real) are additionally
        feature-extracted the same way -- used by retrain.py to fold in user-corrected audio
        without needing it on disk in the ai_generated/real directory layout.
        """
        dataset_path = Path(dataset_dir)
        ai_dir = dataset_path / "ai_generated"
        real_dir = dataset_path / "real"

        ai_files = [p for p in ai_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if ai_dir.exists() else []
        real_files = [p for p in real_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if real_dir.exists() else []

        X: List[np.ndarray] = []
        y: List[int] = []

        for p in ai_files:
            vec = self.extract_features_from_file(p)
            if vec is not None:
                X.append(vec)
                y.append(0)

        for p in real_files:
            vec = self.extract_features_from_file(p)
            if vec is not None:
                X.append(vec)
                y.append(1)

        for p, label in (extra_audio_samples or []):
            vec = self.extract_features_from_file(p)
            if vec is not None:
                X.append(vec)
                y.append(label)

        if not X:
            raise ValueError(f"No valid audio samples found in {dataset_path} or extra_audio_samples")

        logger.info(
            "Extracted acoustic features from %d audio recordings (%d AI, %d Real, %d from corrections).",
            len(X), y.count(0), y.count(1), len(extra_audio_samples or []),
        )
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest audio_detector/tests/test_audio_detector.py::TestAudioDetector::test_prepare_data_merges_extra_samples -v`
Expected: PASS

- [ ] **Step 6: Run the full suite to confirm no regressions, then commit**

Run: `pytest`
Expected: all tests PASS.

```bash
git add audio_detector/config.py audio_detector/trainer.py audio_detector/tests/test_audio_detector.py
git commit -m "feat(audio_detector): support folding extra in-memory samples into prepare_data_from_directory"
```

---

### Task 6: `audio_detector` — `learner.py` queues corrections

**Files:**
- Modify: `audio_detector/learner.py:25-46` (imports + `__init__`), and `record_feedback` around lines 131-136
- Test: `audio_detector/tests/test_audio_detector.py` (new test method)

**Interfaces:**
- Consumes: `core.corrections_store.queue_correction` (Task 1).
- Produces: `AudioSelfImprover(..., corrections_dir: Optional[Path] = None)`; `self.corrections_dir`.

- [ ] **Step 1: Write the failing test**

Add to `audio_detector/tests/test_audio_detector.py`:

```python
    def test_learner_queues_correction_for_retraining(self):
        from core.corrections_store import count_pending

        corrections_dir = self.temp_path / "aud_corrections2"
        learner = AudioSelfImprover(
            memory_file=self.temp_path / "aud_memory2.json",
            calibration_file=self.temp_path / "aud_calib2.json",
        )
        learner.corrections_dir = corrections_dir
        learner.record_feedback(str(self.aud_file), "AI", {"spectral_flatness": 0.001})
        self.assertEqual(count_pending(corrections_dir), {"ai_generated": 1, "real": 0, "total": 1})

        learner.record_feedback(str(self.aud_file), "AI", {"spectral_flatness": 0.001})
        self.assertEqual(count_pending(corrections_dir)["total"], 1)
```

(This test sets `learner.corrections_dir` directly after construction rather than via a constructor kwarg, since `AudioSelfImprover.__init__` doesn't take a `memory_dir`-style directory override the way `ImageSelfImprover` does — see Step 3 below for why the constructor itself only gains a `corrections_dir` parameter, set the same way as `memory_file`/`calibration_file`. Using either the constructor kwarg or direct attribute assignment is equally valid; the constructor kwarg form is preferred in new code and used in Task 4's equivalent image test — using direct assignment here specifically exercises that the attribute is mutable and read at call time, not cached.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest audio_detector/tests/test_audio_detector.py::TestAudioDetector::test_learner_queues_correction_for_retraining -v`
Expected: FAIL with `AttributeError: 'AudioSelfImprover' object has no attribute 'corrections_dir'`

- [ ] **Step 3: Implement**

In `audio_detector/learner.py`, change the imports block (lines 25-27) to:

```python
from audio_detector.config import CALIBRATION_FILE, CORRECTIONS_DIR, DATA_DIR, MEMORY_FILE
from audio_detector.schemas import AudioFeedbackRecord
from core.atomic_io import atomic_read_json, atomic_write_json
from core.corrections_store import queue_correction
```

Change `__init__` (lines 39-46) to:

```python
    def __init__(
        self,
        memory_file: Optional[Path] = None,
        calibration_file: Optional[Path] = None,
        corrections_dir: Optional[Path] = None,
    ):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.memory_file = memory_file or MEMORY_FILE
        self.calibration_file = calibration_file or CALIBRATION_FILE
        self.corrections_dir = Path(corrections_dir) if corrections_dir else CORRECTIONS_DIR
```

In `record_feedback`, find:

```python
        try:
            atomic_write_json(self.memory_file, memory, indent=2)
        except Exception as e:
            logger.error("Failed to save audio memory atomically: %s", e)

        weights = calib.setdefault("acoustic_weights", {})
```

Replace with:

```python
        try:
            atomic_write_json(self.memory_file, memory, indent=2)
        except Exception as e:
            logger.error("Failed to save audio memory atomically: %s", e)

        try:
            aud_path_obj = Path(audio_path)
            if aud_path_obj.is_file():
                correction_label = "ai_generated" if user_label.upper() == "AI" else "real"
                queue_correction(
                    self.corrections_dir,
                    aud_path_obj.read_bytes(),
                    correction_label,
                    aud_path_obj.suffix or ".bin",
                )
        except Exception as e:
            logger.warning("Could not queue correction for retraining: %s", e)

        weights = calib.setdefault("acoustic_weights", {})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest audio_detector/tests/test_audio_detector.py::TestAudioDetector::test_learner_queues_correction_for_retraining -v`
Expected: PASS

- [ ] **Step 5: Run the full suite to confirm no regressions, then commit**

Run: `pytest`
Expected: all tests PASS.

```bash
git add audio_detector/learner.py audio_detector/tests/test_audio_detector.py
git commit -m "feat(audio_detector): queue user feedback corrections for retraining"
```

---

### Task 7: `audio_detector/retrain.py` — fine-tune, validate, promote-or-rollback

**Files:**
- Create: `audio_detector/retrain.py`
- Test: `audio_detector/tests/test_retrain.py`

**Interfaces:**
- Consumes: Task 1's `core.corrections_store` API; Task 5's `AudioDetectorTrainer.prepare_data_from_directory(..., extra_audio_samples=...)`.
- Produces: `count_pending_corrections() -> Dict[str, int]`; `run_retrain(min_new: int = 1, epochs: int = 15) -> RetrainResult`.

- [ ] **Step 1: Write the failing tests**

Create `audio_detector/tests/test_retrain.py`:

```python
"""Tests for audio_detector.retrain -- the feedback-correction fine-tuning/promotion flow."""
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

import audio_detector.config as config
import audio_detector.retrain as retrain
from core.corrections_store import append_checkpoint_log, queue_correction


def _make_wav(path: Path, freq_hz: float, duration_sec: float = 1.0, sr: int = 16000) -> None:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


class TestAudioRetrain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmp.name)

        self._orig = (config.DATASET_DIR, config.DEFAULT_AUDIO_CHECKPOINT, config.CORRECTIONS_DIR, config.CHECKPOINT_LOG_FILE)
        config.DATASET_DIR = tmp_path / "dataset"
        config.DEFAULT_AUDIO_CHECKPOINT = tmp_path / "models" / "audio_detector.pt"
        config.CORRECTIONS_DIR = tmp_path / "corrections"
        config.CHECKPOINT_LOG_FILE = tmp_path / "models" / "CHECKPOINT_LOG.md"
        retrain.DATASET_DIR = config.DATASET_DIR
        retrain.DEFAULT_AUDIO_CHECKPOINT = config.DEFAULT_AUDIO_CHECKPOINT
        retrain.CORRECTIONS_DIR = config.CORRECTIONS_DIR
        retrain.CHECKPOINT_LOG_FILE = config.CHECKPOINT_LOG_FILE

    def tearDown(self):
        config.DATASET_DIR, config.DEFAULT_AUDIO_CHECKPOINT, config.CORRECTIONS_DIR, config.CHECKPOINT_LOG_FILE = self._orig
        self.tmp.cleanup()

    def test_run_retrain_with_no_pending_corrections_is_a_noop(self):
        result = retrain.run_retrain(min_new=1)
        self.assertFalse(result.promoted)
        self.assertIn("Nothing to train on", result.message)

    def test_run_retrain_builds_first_checkpoint_from_corrections_alone(self):
        # audio_detector starts with zero training data on disk -- this proves the
        # bootstrap case: corrections alone are enough to produce a first checkpoint.
        for i in range(3):
            p = Path(self.tmp.name) / f"ai_{i}.wav"
            _make_wav(p, freq_hz=440.0)
            queue_correction(config.CORRECTIONS_DIR, p.read_bytes(), "ai_generated", ".wav")
        for i in range(3):
            p = Path(self.tmp.name) / f"real_{i}.wav"
            _make_wav(p, freq_hz=220.0)
            queue_correction(config.CORRECTIONS_DIR, p.read_bytes(), "real", ".wav")

        result = retrain.run_retrain(min_new=1, epochs=2)

        self.assertTrue(result.promoted)
        self.assertEqual(result.samples_folded_in, 6)
        self.assertTrue(config.DEFAULT_AUDIO_CHECKPOINT.is_file())
        self.assertEqual(retrain.count_pending_corrections()["total"], 0)

    def test_run_retrain_rolls_back_a_worse_candidate(self):
        append_checkpoint_log(
            config.CHECKPOINT_LOG_FILE, version=1, date="2026-01-01",
            train_loss=0.1, val_acc=1.01, new_samples=1, cumulative_samples=1,
        )
        p = Path(self.tmp.name) / "real_rollback.wav"
        _make_wav(p, freq_hz=220.0)
        queue_correction(config.CORRECTIONS_DIR, p.read_bytes(), "real", ".wav")

        result = retrain.run_retrain(min_new=1, epochs=1)

        self.assertFalse(result.promoted)
        self.assertFalse(config.DEFAULT_AUDIO_CHECKPOINT.is_file())
        self.assertEqual(retrain.count_pending_corrections()["total"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest audio_detector/tests/test_retrain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'audio_detector.retrain'`

- [ ] **Step 3: Implement**

Create `audio_detector/retrain.py`:

```python
"""
audio_detector.retrain: Folds queued user corrections into the training set, fine-tunes
(or, since no checkpoint exists yet in this project, trains from scratch) the acoustic
classifier, and promotes the result only if it validates at least as well as the current
checkpoint. See docs/superpowers/specs/2026-10-02-feedback-retraining-pipeline-design.md.
"""
from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from core.corrections_store import (
    RetrainResult,
    append_checkpoint_log,
    count_pending,
    list_pending,
    next_checkpoint_version,
    promote_pending_to_archive,
    read_cumulative_samples,
    read_last_checkpoint_accuracy,
)
from audio_detector.config import CHECKPOINT_LOG_FILE, CORRECTIONS_DIR, DATASET_DIR, DEFAULT_AUDIO_CHECKPOINT
from audio_detector.trainer import AudioDetectorTrainer

logger = logging.getLogger("audio_detector.retrain")


def count_pending_corrections() -> Dict[str, int]:
    """Returns {"ai_generated": N, "real": M, "total": N+M} corrections queued for the next retrain."""
    return count_pending(CORRECTIONS_DIR)


def run_retrain(min_new: int = 1, epochs: int = 15) -> RetrainResult:
    """
    Folds every pending correction into the training set, fine-tunes the current
    checkpoint (or trains a fresh AudioClassifierNet if none exists yet), and
    promotes the result only if it validates at least as well as what's live.
    """
    pending = count_pending_corrections()
    if pending["total"] < min_new:
        return RetrainResult(
            promoted=False,
            new_accuracy=0.0,
            samples_folded_in=0,
            cumulative_samples=read_cumulative_samples(CHECKPOINT_LOG_FILE),
            message=f"Nothing to train on: {pending['total']} correction(s) queued, need at least {min_new}.",
        )

    extra_samples: List[Tuple[Path, int]] = []
    extra_samples += [(p, 0) for p in list_pending(CORRECTIONS_DIR, "ai_generated")]
    extra_samples += [(p, 1) for p in list_pending(CORRECTIONS_DIR, "real")]

    trainer = AudioDetectorTrainer(checkpoint_path=DEFAULT_AUDIO_CHECKPOINT)
    if DEFAULT_AUDIO_CHECKPOINT.is_file():
        checkpoint = torch.load(DEFAULT_AUDIO_CHECKPOINT, map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.info("Continuing fine-tuning from existing checkpoint at %s", DEFAULT_AUDIO_CHECKPOINT)
    else:
        logger.info("No existing checkpoint found -- training a fresh acoustic classifier.")

    X, y = trainer.prepare_data_from_directory(DATASET_DIR, extra_audio_samples=extra_samples)

    candidate_path = DEFAULT_AUDIO_CHECKPOINT.with_name(DEFAULT_AUDIO_CHECKPOINT.stem + ".candidate" + DEFAULT_AUDIO_CHECKPOINT.suffix)
    trainer.checkpoint_path = candidate_path
    history = trainer.train(X, y, epochs=epochs)

    new_accuracy = history["val_accuracy"][-1] if history.get("val_accuracy") else 0.0
    train_loss = history["loss"][-1] if history.get("loss") else 0.0
    old_accuracy = read_last_checkpoint_accuracy(CHECKPOINT_LOG_FILE)

    if old_accuracy is not None and new_accuracy < old_accuracy:
        candidate_path.unlink(missing_ok=True)
        return RetrainResult(
            promoted=False,
            old_accuracy=old_accuracy,
            new_accuracy=new_accuracy,
            samples_folded_in=0,
            cumulative_samples=read_cumulative_samples(CHECKPOINT_LOG_FILE),
            message=(
                f"Candidate checkpoint rolled back: validation accuracy {new_accuracy:.4f} "
                f"is below the current checkpoint's {old_accuracy:.4f}. "
                f"{pending['total']} correction(s) remain queued for the next attempt."
            ),
        )

    candidate_path.replace(DEFAULT_AUDIO_CHECKPOINT)
    promoted_shas = [p.stem for p in list_pending(CORRECTIONS_DIR, "ai_generated")]
    promoted_shas += [p.stem for p in list_pending(CORRECTIONS_DIR, "real")]
    promote_pending_to_archive(CORRECTIONS_DIR, promoted_shas)

    version = next_checkpoint_version(CHECKPOINT_LOG_FILE)
    cumulative = read_cumulative_samples(CHECKPOINT_LOG_FILE) + pending["total"]
    append_checkpoint_log(
        CHECKPOINT_LOG_FILE,
        version=version,
        date=date.today().isoformat(),
        train_loss=train_loss,
        val_acc=new_accuracy,
        new_samples=pending["total"],
        cumulative_samples=cumulative,
    )

    return RetrainResult(
        promoted=True,
        old_accuracy=old_accuracy,
        new_accuracy=new_accuracy,
        samples_folded_in=pending["total"],
        cumulative_samples=cumulative,
        message=(
            f"Promoted checkpoint v{version}: validation accuracy {new_accuracy:.4f} "
            f"({'up from ' + format(old_accuracy, '.4f') if old_accuracy is not None else 'first checkpoint'}). "
            f"{pending['total']} correction(s) folded in."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the audio AI detector on queued user corrections")
    parser.add_argument("--min-new", type=int, default=1, help="Minimum queued corrections required to run")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    args = parser.parse_args()

    result = run_retrain(min_new=args.min_new, epochs=args.epochs)
    print(result.message)
    if result.promoted:
        print(f"New checkpoint accuracy: {result.new_accuracy:.4f} (cumulative samples: {result.cumulative_samples})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest audio_detector/tests/test_retrain.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full suite to confirm no regressions, then commit**

Run: `pytest`
Expected: all tests PASS.

```bash
git add audio_detector/retrain.py audio_detector/tests/test_retrain.py
git commit -m "feat(audio_detector): add retrain.py -- fine-tune, validate, promote-or-rollback"
```

---

### Task 8: `video_detector` — config additions + `trainer.py` extra-samples support + validation split

**Files:**
- Modify: `video_detector/config.py:15-17`
- Modify: `video_detector/trainer.py:75-104` (`prepare_data_from_videos`) and `:106-147` (`train`)
- Test: `video_detector/tests/test_video_detector.py` (new test methods)

**Interfaces:**
- Produces: `video_detector.config.{CORRECTIONS_DIR, CHECKPOINT_LOG_FILE, RETRAIN_SUGGEST_THRESHOLD}`; `VideoDetectorTrainer.prepare_data_from_videos(..., extra_video_samples=...)`; `VideoDetectorTrainer.train(..., val_split: float = 0.0)` now also returns `history["val_accuracy"]`.

Video's trainer currently has **no held-out validation at all** (`train()` trains on 100% of frame pairs with no accuracy tracking) — unlike image and audio, which already split train/val. This task adds that, since the promote/rollback decision in Task 10 needs a real validation accuracy to compare against.

- [ ] **Step 1: Add config constants**

In `video_detector/config.py`, after line 17 (`CALIBRATION_FILE = DATA_DIR / "video_calibration.json"`), add:

```python
CORRECTIONS_DIR = DATA_DIR / "corrections"
CHECKPOINT_LOG_FILE = MODELS_DIR / "CHECKPOINT_LOG.md"
RETRAIN_SUGGEST_THRESHOLD = 15
```

- [ ] **Step 2: Write the failing tests**

Add to `video_detector/tests/test_video_detector.py` (inside the test class):

```python
    def test_prepare_data_merges_extra_video_samples(self):
        from video_detector.trainer import VideoDetectorTrainer

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ai_dir = tmp_path / "ai_generated"
            ai_dir.mkdir()

            def make_video(path):
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
                rng = np.random.RandomState(0)
                for _ in range(5):
                    writer.write(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                writer.release()

            make_video(ai_dir / "ai_0.mp4")
            extra_vid = tmp_path / "extra.mp4"
            make_video(extra_vid)

            trainer = VideoDetectorTrainer()
            pairs = trainer.prepare_data_from_videos(tmp_path, extra_video_samples=[(extra_vid, 1)])
            # 1 on-disk ai video + 1 extra real video, each 5 frames -> 4 pairs each = 8 pairs
            self.assertEqual(len(pairs), 8)
            labels = {label for _, _, label in pairs}
            self.assertEqual(labels, {0, 1})

    def test_train_with_val_split_returns_val_accuracy(self):
        from video_detector.trainer import VideoDetectorTrainer

        rng = np.random.RandomState(0)
        pairs = [(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8), rng.randint(0, 255, (32, 32, 3), dtype=np.uint8), i % 2) for i in range(10)]
        trainer = VideoDetectorTrainer()
        history = trainer.train(pairs, epochs=1, batch_size=2, val_split=0.2)
        self.assertIn("val_accuracy", history)
        self.assertEqual(len(history["val_accuracy"]), 1)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest video_detector/tests/test_video_detector.py::TestVideoDetector::test_prepare_data_merges_extra_video_samples video_detector/tests/test_video_detector.py::TestVideoDetector::test_train_with_val_split_returns_val_accuracy -v`
Expected: both FAIL — first with `TypeError: prepare_data_from_videos() got an unexpected keyword argument 'extra_video_samples'`, second with `KeyError: 'val_accuracy'`

- [ ] **Step 4: Implement — `prepare_data_from_videos`**

In `video_detector/trainer.py`, replace `prepare_data_from_videos` (lines 75-104) with:

```python
    def prepare_data_from_videos(
        self,
        dataset_dir: Path | str,
        max_videos_per_class: int = 50,
        extra_video_samples: Optional[List[Tuple[Path, int]]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Extracts consecutive frame pairs from video datasets with subdirectories
        'ai_generated' and 'real'. `extra_video_samples` (a list of (video_path, label)
        tuples, label 0=ai_generated/1=real) are additionally processed the same way --
        used by retrain.py to fold in user-corrected videos without needing them on disk
        in the ai_generated/real directory layout.
        """
        dataset_path = Path(dataset_dir)
        ai_dir = dataset_path / "ai_generated"
        real_dir = dataset_path / "real"

        ai_vids = [p for p in ai_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS][:max_videos_per_class] if ai_dir.exists() else []
        real_vids = [p for p in real_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS][:max_videos_per_class] if real_dir.exists() else []

        pairs: List[Tuple[np.ndarray, np.ndarray, int]] = []

        for vid in ai_vids:
            frames, _, _ = self.extractor.extract_sampled_frames(vid, max_frames=12)
            for i in range(len(frames) - 1):
                pairs.append((frames[i], frames[i + 1], 0))

        for vid in real_vids:
            frames, _, _ = self.extractor.extract_sampled_frames(vid, max_frames=12)
            for i in range(len(frames) - 1):
                pairs.append((frames[i], frames[i + 1], 1))

        for vid_path, label in (extra_video_samples or []):
            frames, _, _ = self.extractor.extract_sampled_frames(vid_path, max_frames=12)
            for i in range(len(frames) - 1):
                pairs.append((frames[i], frames[i + 1], label))

        random.seed(42)
        random.shuffle(pairs)

        logger.info(
            "Extracted %d frame transition pairs from %d videos (%d extra from corrections).",
            len(pairs), len(ai_vids) + len(real_vids) + len(extra_video_samples or []), len(extra_video_samples or []),
        )
        return pairs
```

- [ ] **Step 5: Implement — `train` with optional validation split**

Replace the `train` method (lines 106-147) with:

```python
    def train(
        self,
        pairs: List[Tuple[np.ndarray, np.ndarray, int]],
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-4,
        val_split: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Trains temporal continuity model. If val_split > 0, a seeded-shuffle fraction of
        `pairs` is held out for per-epoch validation accuracy (history["val_accuracy"]) --
        used by retrain.py to decide whether a candidate checkpoint is safe to promote.
        Default val_split=0.0 preserves prior behavior exactly: train on all pairs, no
        held-out validation.
        """
        if not pairs:
            raise ValueError("No video frame pairs available for training.")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_pairs = pairs
        val_pairs: List[Tuple[np.ndarray, np.ndarray, int]] = []
        if val_split > 0.0:
            shuffled = list(pairs)
            random.seed(42)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * (1.0 - val_split))
            train_pairs = shuffled[:split_idx]
            val_pairs = shuffled[split_idx:]

        dataset = FrameTransitionDataset(train_pairs, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_pairs:
            val_dataset = FrameTransitionDataset(val_pairs, transform=transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        history: Dict[str, Any] = {"loss": [], "val_accuracy": []}
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)

            avg_loss = total_loss / max(1, len(train_pairs))
            history["loss"].append(avg_loss)

            if val_loader is not None:
                self.model.eval()
                correct = 0
                val_total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        val_total += len(labels)
                val_acc = correct / max(1, val_total)
                history["val_accuracy"].append(val_acc)
                logger.info("Video Trainer Epoch [%d/%d] - Loss: %.4f - Val Acc: %.2f%%", epoch + 1, epochs, avg_loss, val_acc * 100.0)
            else:
                logger.info("Video Trainer Epoch [%d/%d] - Loss: %.4f", epoch + 1, epochs, avg_loss)

        self.save_checkpoint()
        return history
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest video_detector/tests/test_video_detector.py::TestVideoDetector::test_prepare_data_merges_extra_video_samples video_detector/tests/test_video_detector.py::TestVideoDetector::test_train_with_val_split_returns_val_accuracy -v`
Expected: both PASS

- [ ] **Step 7: Run the full suite to confirm no regressions, then commit**

Run: `pytest`
Expected: all tests PASS, including anything exercising `VideoDetectorTrainer.train()` at its old default (`val_split=0.0` → `history["val_accuracy"] == []`, `history["loss"]` unchanged in meaning).

```bash
git add video_detector/config.py video_detector/trainer.py video_detector/tests/test_video_detector.py
git commit -m "feat(video_detector): support extra in-memory video samples and an optional held-out validation split"
```

---

### Task 9: `video_detector` — `learner.py` queues corrections

**Files:**
- Modify: `video_detector/learner.py:26-49` (imports + `__init__`), and `record_feedback` around lines 134-139
- Test: `video_detector/tests/test_video_detector.py` (new test method)

**Interfaces:**
- Consumes: `core.corrections_store.queue_correction` (Task 1).
- Produces: `VideoSelfImprover(..., corrections_dir: Optional[Path] = None)`; `self.corrections_dir`.

- [ ] **Step 1: Write the failing test**

Add to `video_detector/tests/test_video_detector.py`:

```python
    def test_learner_queues_correction_for_retraining(self):
        from core.corrections_store import count_pending

        corrections_dir = self.temp_path / "vid_corrections2"
        learner = VideoSelfImprover(
            memory_file=self.temp_path / "vid_memory2.json",
            calibration_file=self.temp_path / "vid_calib2.json",
            corrections_dir=corrections_dir,
        )
        learner.record_feedback(str(self.vid_file), "AI", {"motion_variance": 10.0})
        self.assertEqual(count_pending(corrections_dir), {"ai_generated": 1, "real": 0, "total": 1})

        learner.record_feedback(str(self.vid_file), "AI", {"motion_variance": 10.0})
        self.assertEqual(count_pending(corrections_dir)["total"], 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest video_detector/tests/test_video_detector.py::TestVideoDetector::test_learner_queues_correction_for_retraining -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'corrections_dir'`

- [ ] **Step 3: Implement**

In `video_detector/learner.py`, change the imports block (lines 26-28) to:

```python
from video_detector.config import CALIBRATION_FILE, CORRECTIONS_DIR, DATA_DIR, MEMORY_FILE
from video_detector.schemas import VideoFeedbackRecord
from core.atomic_io import atomic_read_json, atomic_write_json
from core.corrections_store import queue_correction
```

Change `__init__` (lines 40-49) to:

```python
    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        memory_file: Optional[Path] = None,
        calibration_file: Optional[Path] = None,
        corrections_dir: Optional[Path] = None,
    ):
        self.memory_dir = memory_dir or DATA_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = Path(memory_file) if memory_file else (self.memory_dir / "video_feedback.json")
        self.calibration_file = Path(calibration_file) if calibration_file else (self.memory_dir / "video_calibration.json")
        self.corrections_dir = Path(corrections_dir) if corrections_dir else CORRECTIONS_DIR
```

In `record_feedback`, find:

```python
        try:
            atomic_write_json(self.feedback_file, memory, indent=2)
        except Exception as e:
            logger.error("Failed to save video memory atomically: %s", e)

        m_thresh = calib.setdefault("motion_thresholds", {})
```

Replace with:

```python
        try:
            atomic_write_json(self.feedback_file, memory, indent=2)
        except Exception as e:
            logger.error("Failed to save video memory atomically: %s", e)

        try:
            vid_path_obj = Path(video_path)
            if vid_path_obj.is_file():
                correction_label = "ai_generated" if user_label.upper() == "AI" else "real"
                queue_correction(
                    self.corrections_dir,
                    vid_path_obj.read_bytes(),
                    correction_label,
                    vid_path_obj.suffix or ".bin",
                )
        except Exception as e:
            logger.warning("Could not queue correction for retraining: %s", e)

        m_thresh = calib.setdefault("motion_thresholds", {})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest video_detector/tests/test_video_detector.py::TestVideoDetector::test_learner_queues_correction_for_retraining -v`
Expected: PASS

- [ ] **Step 5: Run the full suite to confirm no regressions, then commit**

Run: `pytest`
Expected: all tests PASS.

```bash
git add video_detector/learner.py video_detector/tests/test_video_detector.py
git commit -m "feat(video_detector): queue user feedback corrections for retraining"
```

---

### Task 10: `video_detector/retrain.py` — fine-tune, validate, promote-or-rollback

**Files:**
- Create: `video_detector/retrain.py`
- Test: `video_detector/tests/test_retrain.py`

**Interfaces:**
- Consumes: Task 1's `core.corrections_store` API; Task 8's `VideoDetectorTrainer.prepare_data_from_videos(..., extra_video_samples=...)` and `.train(..., val_split=...)`.
- Produces: `count_pending_corrections() -> Dict[str, int]`; `run_retrain(min_new: int = 1, epochs: int = 5) -> RetrainResult`.

- [ ] **Step 1: Write the failing tests**

Create `video_detector/tests/test_retrain.py`:

```python
"""Tests for video_detector.retrain -- the feedback-correction fine-tuning/promotion flow."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import video_detector.config as config
import video_detector.retrain as retrain
from core.corrections_store import append_checkpoint_log, queue_correction


def _make_video(path: Path, n_frames: int = 15, size: int = 64) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (size, size))
    rng = np.random.RandomState(0)
    for _ in range(n_frames):
        writer.write(rng.randint(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()


class TestVideoRetrain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmp.name)

        self._orig = (config.DATASET_DIR, config.DEFAULT_VIDEO_CHECKPOINT, config.CORRECTIONS_DIR, config.CHECKPOINT_LOG_FILE)
        config.DATASET_DIR = tmp_path / "dataset"
        config.DEFAULT_VIDEO_CHECKPOINT = tmp_path / "models" / "video_detector.pt"
        config.CORRECTIONS_DIR = tmp_path / "corrections"
        config.CHECKPOINT_LOG_FILE = tmp_path / "models" / "CHECKPOINT_LOG.md"
        retrain.DATASET_DIR = config.DATASET_DIR
        retrain.DEFAULT_VIDEO_CHECKPOINT = config.DEFAULT_VIDEO_CHECKPOINT
        retrain.CORRECTIONS_DIR = config.CORRECTIONS_DIR
        retrain.CHECKPOINT_LOG_FILE = config.CHECKPOINT_LOG_FILE

    def tearDown(self):
        config.DATASET_DIR, config.DEFAULT_VIDEO_CHECKPOINT, config.CORRECTIONS_DIR, config.CHECKPOINT_LOG_FILE = self._orig
        self.tmp.cleanup()

    def test_run_retrain_with_no_pending_corrections_is_a_noop(self):
        result = retrain.run_retrain(min_new=1)
        self.assertFalse(result.promoted)
        self.assertIn("Nothing to train on", result.message)

    def test_run_retrain_builds_first_checkpoint_from_corrections_alone(self):
        for i in range(2):
            p = Path(self.tmp.name) / f"ai_{i}.mp4"
            _make_video(p)
            queue_correction(config.CORRECTIONS_DIR, p.read_bytes(), "ai_generated", ".mp4")
        for i in range(2):
            p = Path(self.tmp.name) / f"real_{i}.mp4"
            _make_video(p)
            queue_correction(config.CORRECTIONS_DIR, p.read_bytes(), "real", ".mp4")

        result = retrain.run_retrain(min_new=1, epochs=1)

        self.assertTrue(result.promoted)
        self.assertEqual(result.samples_folded_in, 4)
        self.assertTrue(config.DEFAULT_VIDEO_CHECKPOINT.is_file())
        self.assertEqual(retrain.count_pending_corrections()["total"], 0)

    def test_run_retrain_rolls_back_a_worse_candidate(self):
        append_checkpoint_log(
            config.CHECKPOINT_LOG_FILE, version=1, date="2026-01-01",
            train_loss=0.1, val_acc=1.01, new_samples=1, cumulative_samples=1,
        )
        p = Path(self.tmp.name) / "real_rollback.mp4"
        _make_video(p)
        queue_correction(config.CORRECTIONS_DIR, p.read_bytes(), "real", ".mp4")

        result = retrain.run_retrain(min_new=1, epochs=1)

        self.assertFalse(result.promoted)
        self.assertFalse(config.DEFAULT_VIDEO_CHECKPOINT.is_file())
        self.assertEqual(retrain.count_pending_corrections()["total"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest video_detector/tests/test_retrain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'video_detector.retrain'`

- [ ] **Step 3: Implement**

Create `video_detector/retrain.py`:

```python
"""
video_detector.retrain: Folds queued user corrections into the training set, fine-tunes
(or, since no checkpoint exists yet in this project, trains from scratch) the temporal
transition model, and promotes the result only if it validates at least as well as the
current checkpoint. See docs/superpowers/specs/2026-10-02-feedback-retraining-pipeline-design.md.
"""
from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from core.corrections_store import (
    RetrainResult,
    append_checkpoint_log,
    count_pending,
    list_pending,
    next_checkpoint_version,
    promote_pending_to_archive,
    read_cumulative_samples,
    read_last_checkpoint_accuracy,
)
from video_detector.config import CHECKPOINT_LOG_FILE, CORRECTIONS_DIR, DATASET_DIR, DEFAULT_VIDEO_CHECKPOINT
from video_detector.trainer import VideoDetectorTrainer

logger = logging.getLogger("video_detector.retrain")


def count_pending_corrections() -> Dict[str, int]:
    """Returns {"ai_generated": N, "real": M, "total": N+M} corrections queued for the next retrain."""
    return count_pending(CORRECTIONS_DIR)


def run_retrain(min_new: int = 1, epochs: int = 5) -> RetrainResult:
    """
    Folds every pending correction into the training set, fine-tunes the current
    checkpoint (or trains a fresh VideoTemporalTransitionModel if none exists yet),
    and promotes the result only if it validates at least as well as what's live.
    """
    pending = count_pending_corrections()
    if pending["total"] < min_new:
        return RetrainResult(
            promoted=False,
            new_accuracy=0.0,
            samples_folded_in=0,
            cumulative_samples=read_cumulative_samples(CHECKPOINT_LOG_FILE),
            message=f"Nothing to train on: {pending['total']} correction(s) queued, need at least {min_new}.",
        )

    extra_samples: List[Tuple[Path, int]] = []
    extra_samples += [(p, 0) for p in list_pending(CORRECTIONS_DIR, "ai_generated")]
    extra_samples += [(p, 1) for p in list_pending(CORRECTIONS_DIR, "real")]

    trainer = VideoDetectorTrainer(checkpoint_path=DEFAULT_VIDEO_CHECKPOINT)
    if DEFAULT_VIDEO_CHECKPOINT.is_file():
        checkpoint = torch.load(DEFAULT_VIDEO_CHECKPOINT, map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.info("Continuing fine-tuning from existing checkpoint at %s", DEFAULT_VIDEO_CHECKPOINT)
    else:
        logger.info("No existing checkpoint found -- training a fresh temporal transition model.")

    pairs = trainer.prepare_data_from_videos(DATASET_DIR, extra_video_samples=extra_samples)

    candidate_path = DEFAULT_VIDEO_CHECKPOINT.with_name(DEFAULT_VIDEO_CHECKPOINT.stem + ".candidate" + DEFAULT_VIDEO_CHECKPOINT.suffix)
    trainer.checkpoint_path = candidate_path
    history = trainer.train(pairs, epochs=epochs, val_split=0.2)

    new_accuracy = history["val_accuracy"][-1] if history.get("val_accuracy") else 0.0
    train_loss = history["loss"][-1] if history.get("loss") else 0.0
    old_accuracy = read_last_checkpoint_accuracy(CHECKPOINT_LOG_FILE)

    if old_accuracy is not None and new_accuracy < old_accuracy:
        candidate_path.unlink(missing_ok=True)
        return RetrainResult(
            promoted=False,
            old_accuracy=old_accuracy,
            new_accuracy=new_accuracy,
            samples_folded_in=0,
            cumulative_samples=read_cumulative_samples(CHECKPOINT_LOG_FILE),
            message=(
                f"Candidate checkpoint rolled back: validation accuracy {new_accuracy:.4f} "
                f"is below the current checkpoint's {old_accuracy:.4f}. "
                f"{pending['total']} correction(s) remain queued for the next attempt."
            ),
        )

    candidate_path.replace(DEFAULT_VIDEO_CHECKPOINT)
    promoted_shas = [p.stem for p in list_pending(CORRECTIONS_DIR, "ai_generated")]
    promoted_shas += [p.stem for p in list_pending(CORRECTIONS_DIR, "real")]
    promote_pending_to_archive(CORRECTIONS_DIR, promoted_shas)

    version = next_checkpoint_version(CHECKPOINT_LOG_FILE)
    cumulative = read_cumulative_samples(CHECKPOINT_LOG_FILE) + pending["total"]
    append_checkpoint_log(
        CHECKPOINT_LOG_FILE,
        version=version,
        date=date.today().isoformat(),
        train_loss=train_loss,
        val_acc=new_accuracy,
        new_samples=pending["total"],
        cumulative_samples=cumulative,
    )

    return RetrainResult(
        promoted=True,
        old_accuracy=old_accuracy,
        new_accuracy=new_accuracy,
        samples_folded_in=pending["total"],
        cumulative_samples=cumulative,
        message=(
            f"Promoted checkpoint v{version}: validation accuracy {new_accuracy:.4f} "
            f"({'up from ' + format(old_accuracy, '.4f') if old_accuracy is not None else 'first checkpoint'}). "
            f"{pending['total']} correction(s) folded in."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the video AI detector on queued user corrections")
    parser.add_argument("--min-new", type=int, default=1, help="Minimum queued corrections required to run")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    args = parser.parse_args()

    result = run_retrain(min_new=args.min_new, epochs=args.epochs)
    print(result.message)
    if result.promoted:
        print(f"New checkpoint accuracy: {result.new_accuracy:.4f} (cumulative samples: {result.cumulative_samples})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest video_detector/tests/test_retrain.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full suite to confirm no regressions, then commit**

Run: `pytest`
Expected: all tests PASS.

```bash
git add video_detector/retrain.py video_detector/tests/test_retrain.py
git commit -m "feat(video_detector): add retrain.py -- fine-tune, validate, promote-or-rollback"
```

---

### Task 11: Git LFS — track `models/*.pt` instead of ignoring it

**Files:**
- Create: `.gitattributes`
- Modify: `.gitignore` (append after the `*.pt`/`*.pth` block)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a repo state where `models/*.pt` is trackable (via LFS) in later `git add` calls — no code depends on this, it's pure repo configuration, verified by direct `git check-ignore`/`git lfs` commands, not pytest.

- [ ] **Step 1: Verify Git LFS is installed**

Run: `git lfs version`
Expected: a version string (e.g. `git-lfs/3.x.x`). If this errors with "not found," install Git LFS first (https://git-lfs.com) before continuing — the rest of this task requires it.

- [ ] **Step 2: Initialize LFS for this repository**

Run: `git lfs install`
Expected: `Updated git hooks. Git LFS initialized.` (or `Git LFS initialized.` if hooks already exist)

- [ ] **Step 3: Create `.gitattributes`**

Create `.gitattributes` at the repository root:

```
*.pt filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 4: Un-ignore `models/*.pt` specifically**

In `.gitignore`, find:

```
# Trained model files & weights
*.pkl
*.joblib
*.h5
*.pt
*.pth
*.onnx
*.bin
*.weights
*.ckpt
```

Replace it with:

```
# Trained model files & weights
*.pkl
*.joblib
*.h5
*.pt
*.pth
*.onnx
*.bin
*.weights
*.ckpt

# Exception: trained checkpoints are intentionally tracked via Git LFS (see
# .gitattributes) so a fresh clone reproduces the exact same detection results
# without ever needing the personal training data that produced them.
!**/models/*.pt
```

- [ ] **Step 5: Verify the negation works and the existing checkpoint is now trackable**

Run: `git check-ignore -v image_detector/models/ai_detector.pt`
Expected: **no output and a non-zero exit code** — meaning the path is no longer ignored. (Compare to before this task, which printed `.gitignore:48:*.pt    image_detector/models/ai_detector.pt`.)

Run: `git lfs track "*.pt"` is **not** needed as a separate step — `.gitattributes` created in Step 3 already declares this; confirm it's recognized:

Run: `git lfs track`
Expected: output includes a line listing `*.pt` as a tracked pattern.

- [ ] **Step 6: Stage and commit the existing checkpoint through LFS, plus the config changes**

```bash
git add .gitattributes .gitignore
git add image_detector/models/ai_detector.pt
git status
```

Expected in `git status`: `image_detector/models/ai_detector.pt` appears as a new file staged for commit (not ignored). If it still shows as ignored, re-check Step 4's exact `.gitignore` edit before proceeding.

```bash
git commit -m "chore: track models/*.pt via Git LFS instead of gitignoring it

A fresh clone + 'git lfs pull' now reproduces the exact same checkpoint
(and therefore the exact same detection results) the training machine
produced, without ever receiving the personal media that trained it."
```

- [ ] **Step 7: Confirm the full test suite still passes**

Run: `pytest`
Expected: all tests PASS (this task touches no Python code, only git configuration — a full-suite run here is a sanity check that nothing in the working tree was accidentally disturbed).

---

### Task 12: UI integration — "Retrain Now" in the Continuous Learning dashboard

**Files:**
- Modify: `ui/feedback_ui.py` (imports near the top, and `render_learning_dashboard` around line 402)

**Interfaces:**
- Consumes: `image_detector.config.RETRAIN_SUGGEST_THRESHOLD`, `image_detector.retrain.{count_pending_corrections, run_retrain}` (Tasks 2 & 4); the same from `audio_detector`/`video_detector` (Tasks 5–7, 8–10).

This task has no automated test (it's a Streamlit rendering change — the project's existing convention, confirmed across every other `render_*` function in this file, is that these are verified by manually running the app, not unit-tested). Verification is a manual Streamlit check in Step 3.

- [ ] **Step 1: Add the new imports**

In `ui/feedback_ui.py`, find the existing imports block:

```python
from audio_detector import AudioProfiler, AudioSelfImprover
from core.decision import generate_final_decision, normalize_percentages
from image_detector import ImageProfiler, ImageSelfImprover
from image_detector.explain import build_nine_dimensions_dossier, generate_newbie_explanation
from video_detector import VideoProfiler, VideoSelfImprover
```

Replace it with:

```python
import audio_detector.config as audio_config
import audio_detector.retrain as audio_retrain
from audio_detector import AudioProfiler, AudioSelfImprover
from core.decision import generate_final_decision, normalize_percentages
import image_detector.config as image_config
import image_detector.retrain as image_retrain
from image_detector import ImageProfiler, ImageSelfImprover
from image_detector.explain import build_nine_dimensions_dossier, generate_newbie_explanation
import video_detector.config as video_config
import video_detector.retrain as video_retrain
from video_detector import VideoProfiler, VideoSelfImprover
```

- [ ] **Step 2: Add the "Retrain on Corrections" section to the dashboard**

In `render_learning_dashboard`, find:

```python
    ai_count = sum(1 for r in all_records if r.get("user_label") == "AI")
    real_count = sum(1 for r in all_records if r.get("user_label") == "REAL")
    m3.metric("Verified AI Samples", ai_count)
    m4.metric("Verified Real Samples", real_count)

    st.markdown("---")

    # Dynamic Weights and Calibration Offsets
```

Replace it with:

```python
    ai_count = sum(1 for r in all_records if r.get("user_label") == "AI")
    real_count = sum(1 for r in all_records if r.get("user_label") == "REAL")
    m3.metric("Verified AI Samples", ai_count)
    m4.metric("Verified Real Samples", real_count)

    st.markdown("---")

    # Retraining -- queued corrections per modality
    st.subheader("🔁 Retrain on Corrections")
    st.caption(
        "Corrections you submit below get queued, not trained on immediately. "
        "Retraining fine-tunes the checkpoint and only replaces it if the result "
        "validates at least as well as what's live -- see README.md."
    )
    for modality_name, retrain_module, threshold in (
        ("Image", image_retrain, image_config.RETRAIN_SUGGEST_THRESHOLD),
        ("Video", video_retrain, video_config.RETRAIN_SUGGEST_THRESHOLD),
        ("Audio", audio_retrain, audio_config.RETRAIN_SUGGEST_THRESHOLD),
    ):
        pending = retrain_module.count_pending_corrections()
        if pending["total"] == 0:
            continue
        col_info, col_btn = st.columns([4, 1])
        ready_note = "Ready to retrain." if pending["total"] >= threshold else f"({threshold} suggested before retraining.)"
        col_info.info(
            f"**{modality_name}**: {pending['total']} correction(s) queued "
            f"({pending['ai_generated']} AI, {pending['real']} Real). {ready_note}"
        )
        if col_btn.button(f"Retrain {modality_name} Now", key=f"retrain_{modality_name.lower()}"):
            with st.spinner(f"Retraining {modality_name.lower()} detector..."):
                result = retrain_module.run_retrain(min_new=1)
            if result.promoted:
                st.success(result.message)
            else:
                st.warning(result.message)

    st.markdown("---")

    # Dynamic Weights and Calibration Offsets
```

- [ ] **Step 3: Manually verify in the running app**

Run: `streamlit run app.py`

In the browser: go to the "🧠 Continuous Learning & Memory" tab.
Expected: if no corrections are queued for any modality (the common case right after this plan is first implemented), the new "🔁 Retrain on Corrections" subheader and caption render but no per-modality banner appears (every `pending["total"]` is 0) — confirm the page doesn't error.

Then: go to the Image tab, analyze any image, and submit feedback via the rating panel with a ground-truth label that differs from the detector's verdict (forcing a correction). Return to the Continuous Learning tab.
Expected: an info banner now reads "**Image**: 1 correction(s) queued (... AI, ... Real). (15 suggested before retraining.)" with a "Retrain Image Now" button. Click it.
Expected: a spinner appears, then either a green success message (checkpoint promoted) or a yellow warning (rolled back) — both are correct outcomes depending on whether the 1-sample fine-tune happens to validate at or above the existing checkpoint's logged accuracy.

Stop the Streamlit server (Ctrl+C) once verified.

- [ ] **Step 4: Run the full suite, then commit**

Run: `pytest`
Expected: all tests PASS (this task adds no new automated tests, but must not break existing ones — `ui/feedback_ui.py` isn't directly imported by the pytest suite's collected tests, but confirm nothing else regressed).

```bash
git add ui/feedback_ui.py
git commit -m "feat(ui): surface queued corrections and a Retrain Now button in the Continuous Learning dashboard"
```

---

### Task 13: README updates + final end-to-end verification

**Files:**
- Modify: `README.md`

**Interfaces:** None — documentation only.

- [ ] **Step 1: Update the "Data, models, and what's actually on disk vs. git" section**

In `README.md`, find the bullet:

```
- `image_detector/models/ai_detector.pt` (~42.8MB trained ResNet18 checkpoint) exists on disk and is loaded by `ImageAIDetector.load()`, but `.gitignore` excludes `*.pt` files — it is **not tracked in git**. If you clone this repo fresh, this file will be missing and the image detector will silently fall back to pure heuristic/statistical mode (the `load()` method catches the exception and logs a warning rather than failing).
```

Replace it with:

```
- `image_detector/models/ai_detector.pt` (~42.8MB trained ResNet18 checkpoint) exists on disk and is loaded by `ImageAIDetector.load()`. It **is tracked in git, via Git LFS** (`.gitattributes` declares `*.pt filter=lfs diff=lfs merge=lfs -text`). Run `git lfs install` once, then `git clone`/`git lfs pull` as normal — you'll get the exact same checkpoint bytes, and therefore the exact same detection results, without ever receiving the personal media that trained it. If you clone without Git LFS set up, the `.pt` file arrives as a small pointer stub instead of real weights; `ImageAIDetector.load()` will fail to load it, log a warning, and fall back to pure heuristic/statistical mode rather than crashing — run `git lfs pull` to fix this.
```

- [ ] **Step 2: Add a new "Feedback-driven retraining" section**

In `README.md`, immediately after the "What this project actually is (and isn't)" section's closing bullet list (before the `## \`scripts/\`` section), insert:

```markdown
---

## Feedback-driven retraining

Every modality's feedback widget (the rating panel under each result) now does two things when you submit a correction:

1. **Always** (unchanged): nudges a handful of heuristic scoring-weight constants, as described above.
2. **New**: copies the corrected file into `<modality>_detector/data/corrections/pending/<ai_generated|real>/<sha256>.<ext>` and records it in a manifest. This queue is gitignored (covered by the repo's existing blanket `data/` rule) — your corrected files never leave your machine.

Nothing retrains automatically. Once enough corrections are queued (15 by default — see `RETRAIN_SUGGEST_THRESHOLD` in each package's `config.py`), the "🧠 Continuous Learning & Memory" tab shows a banner with a **Retrain Now** button per modality. You can also trigger it from the command line:

```bash
python -m image_detector.retrain   # or audio_detector.retrain / video_detector.retrain
```

What happens on retrain (`<modality>_detector/retrain.py`):

1. Every queued correction is folded into the existing `dataset/train`+`dataset/val` split (or, for `audio_detector`/`video_detector`, which currently ship with zero training data, the corrections alone become the first-ever training set).
2. The existing checkpoint is fine-tuned for a few epochs (or trained from pretrained/random initialization if no checkpoint exists yet).
3. The result is validated on a held-out split and compared against the current checkpoint's last-logged accuracy (from `models/CHECKPOINT_LOG.md`).
4. **Promoted** only if the new accuracy is `>=` the old one (or unconditionally for the very first checkpoint): the live `.pt` file is replaced, the folded-in corrections move from `pending/` to `archive/`, and a row is appended to `CHECKPOINT_LOG.md` — version, date, training loss, validation accuracy, and sample counts. **Never** filenames, paths, or image content.
5. **Rolled back** otherwise: the candidate is discarded, every correction stays queued untouched, and you can gather more corrections before trying again.

`CHECKPOINT_LOG.md` is git-tracked per modality, so anyone reading the repo can see exactly how the live checkpoint's accuracy evolved over time without you ever exposing what was actually in your personal training set.
```

- [ ] **Step 3: Final full end-to-end verification**

Run: `pytest -v`
Expected: every test across `core/tests`, `image_detector/tests` (including the new `test_corrections_store.py`-adjacent and `test_retrain.py` tests), `audio_detector/tests`, `video_detector/tests`, and `tests/` PASSES.

Run: `git status`
Expected: only `README.md` is modified and unstaged at this point (everything else was committed task-by-task).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document the feedback-driven retraining pipeline and Git LFS checkpoint distribution"
```

---

## Self-Review Notes (completed during plan writing, not a separate pass)

- **Spec coverage:** every architecture piece from the spec (corrections store, per-modality `retrain.py`, Git LFS, `CHECKPOINT_LOG.md`, UI banner, README) maps to a task above. The spec's illustrative `CHECKPOINT_LOG.md` column "Train Acc" was changed to "Train Loss" during planning — none of the three trainers currently compute a true training-set accuracy (only training *loss*, and separately a *validation* accuracy), and fabricating a fake accuracy number from loss would violate the "no fabrications" constraint. This is a refinement of the spec's illustrative sketch, not a scope change.
- **Placeholder scan:** no TBD/TODO; every step has complete, runnable code.
- **Type consistency:** `RetrainResult`, `count_pending`/`count_pending_corrections`, `list_pending`, `queue_correction`, `promote_pending_to_archive`, `append_checkpoint_log`, `read_last_checkpoint_accuracy`, `next_checkpoint_version`, `read_cumulative_samples` are defined once in Task 1 and used with identical signatures in every later task.
- **Video's missing validation split** was identified as a blocking gap during planning (its `train()` had no held-out accuracy at all) and folded into Task 8 rather than left as a silent limitation — necessary for Task 10's promote/rollback decision to be real rather than fabricated.
