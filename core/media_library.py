"""
core.media_library: Content-addressed, copy-free registry of labeled media.

Training data stays exactly where it already lives on disk. This registry only records,
per file: its SHA-256 content hash (the identity), its label, and a path *hint* used to
find the bytes again. Nothing is copied, nothing is deleted, and file names are never part
of any training signal -- renaming or moving a file never changes what the model sees.

Pure path/file logic with no imports from any detector package (core/ never depends on
image_detector/audio_detector/video_detector).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from core.atomic_io import atomic_read_json, atomic_write_json

logger = logging.getLogger("core.media_library")

LABELS = ("ai_generated", "real")
_VAL_DENOMINATOR = 5  # sha-derived bucket: 1 in 5 files (~20%) is validation, always.


def compute_file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Streams the file once and returns its hex SHA-256 (constant memory, any file size)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def partition(sha256: str) -> str:
    """
    Deterministic, content-derived split: "val" or "train". The same file is always in the
    same split regardless of name, location, or when it was added, and adding new files
    never reshuffles existing ones -- so validation data can never leak into training.
    """
    return "val" if int(sha256[:8], 16) % _VAL_DENOMINATOR == 0 else "train"


def _stat_key(path: Path) -> Tuple[int, int]:
    st = path.stat()
    return st.st_size, st.st_mtime_ns


class MediaLibrary:
    """Registry of labeled media, persisted atomically as a single local (gitignored) JSON file."""

    def __init__(self, manifest_path: str | Path):
        self.path = Path(manifest_path)
        data = atomic_read_json(self.path, default=None)
        entries = data.get("entries", {}) if isinstance(data, dict) else {}
        self._entries: Dict[str, Dict[str, Any]] = entries if isinstance(entries, dict) else {}
        self._by_path: Dict[str, str] = {
            e["path_hint"]: sha for sha, e in self._entries.items() if e.get("path_hint")
        }

    # ------------------------------------------------------------------ persistence
    def _save(self) -> None:
        atomic_write_json(self.path, {"version": 1, "entries": self._entries}, indent=1)

    # ------------------------------------------------------------------ adding
    def _hash_with_cache(self, p: Path) -> Tuple[str, int, int]:
        size, mtime_ns = _stat_key(p)
        known = self._by_path.get(str(p))
        if known is not None:
            e = self._entries.get(known, {})
            if e.get("size") == size and e.get("mtime_ns") == mtime_ns:
                return known, size, mtime_ns  # unchanged since last hash: skip re-reading the file
        return compute_file_sha256(p), size, mtime_ns

    def _add_no_save(self, path: str | Path, label: str) -> bool:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {label!r}")
        p = Path(path).resolve()
        sha, size, mtime_ns = self._hash_with_cache(p)
        existing = self._entries.get(sha)
        if existing is None:
            self._entries[sha] = {
                "label": label,
                "path_hint": str(p),
                "size": size,
                "mtime_ns": mtime_ns,
                "status": "new",
            }
            self._by_path[str(p)] = sha
            return True
        # Same content already registered: refresh the hint only if the old one is dead.
        hint = existing.get("path_hint")
        if not hint or not Path(hint).is_file():
            existing.update(path_hint=str(p), size=size, mtime_ns=mtime_ns)
            self._by_path[str(p)] = sha
        if existing["label"] != label:
            existing["label"] = label
            existing["status"] = "new"  # a changed label is a correction worth (re)training on
            return True
        return False

    def add(self, path: str | Path, label: str) -> bool:
        """Registers one file. Returns True if the registry changed (new content or relabel)."""
        changed = self._add_no_save(path, label)
        self._save()
        return changed

    def add_many(self, paths: Iterable[str | Path], label: str) -> int:
        """Registers many files with a single manifest write. Returns how many changed the registry."""
        changed = 0
        for p in paths:
            try:
                changed += 1 if self._add_no_save(p, label) else 0
            except (OSError, ValueError) as exc:
                if isinstance(exc, ValueError):
                    raise
                logger.warning("Skipping unreadable file %s: %s", p, exc)
        self._save()
        return changed

    def add_tree(self, root: str | Path, label: str, extensions: Set[str]) -> int:
        """Recursively registers every file under `root` whose extension is in `extensions`."""
        files = [
            Path(dirpath) / name
            for dirpath, _, names in os.walk(root)
            for name in names
            if Path(name).suffix.lower() in extensions
        ]
        return self.add_many(files, label)

    # ------------------------------------------------------------------ lookup
    def resolve(self, sha256: str) -> Optional[Path]:
        """Returns the current on-disk path for this content hash, or None if it can't be found."""
        e = self._entries.get(sha256)
        if not e or not e.get("path_hint"):
            return None
        p = Path(e["path_hint"])
        try:
            size, mtime_ns = _stat_key(p)
        except OSError:
            return None
        if size == e.get("size") and mtime_ns == e.get("mtime_ns"):
            return p
        # Same path but changed metadata: only trust it if the bytes still hash to this identity.
        if compute_file_sha256(p) == sha256:
            e.update(size=size, mtime_ns=mtime_ns)
            return p
        return None

    def missing(self) -> List[str]:
        """Content hashes whose files can no longer be found at their recorded path."""
        return [sha for sha in self._entries if self.resolve(sha) is None]

    def rescan(self, roots: Iterable[str | Path]) -> int:
        """
        Re-links entries whose path went stale by walking `roots`. Only files whose size matches
        a missing entry are hashed, so a rescan over a huge folder stays cheap.
        Returns the number of entries relinked.
        """
        missing = set(self.missing())
        if not missing:
            return 0
        wanted_sizes: Dict[int, Set[str]] = {}
        for sha in missing:
            wanted_sizes.setdefault(self._entries[sha]["size"], set()).add(sha)
        relinked = 0
        for root in roots:
            for dirpath, _, names in os.walk(root):
                for name in names:
                    p = Path(dirpath) / name
                    try:
                        size, mtime_ns = _stat_key(p)
                    except OSError:
                        continue
                    if size not in wanted_sizes:
                        continue
                    sha = compute_file_sha256(p)
                    if sha in missing:
                        e = self._entries[sha]
                        e.update(path_hint=str(p.resolve()), size=size, mtime_ns=mtime_ns)
                        self._by_path[str(p.resolve())] = sha
                        missing.discard(sha)
                        relinked += 1
        if relinked:
            self._save()
        return relinked

    # ------------------------------------------------------------------ queries
    def entries(self) -> List[Dict[str, Any]]:
        return [dict(e, sha256=sha) for sha, e in self._entries.items()]

    def samples(
        self,
        label: Optional[str] = None,
        split: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Tuple[str, Path, str]]:
        """Resolvable (sha256, path, label) triples, optionally filtered. Unresolvable files are skipped."""
        out: List[Tuple[str, Path, str]] = []
        for sha, e in self._entries.items():
            if label and e["label"] != label:
                continue
            if status and e.get("status") != status:
                continue
            if split and partition(sha) != split:
                continue
            p = self.resolve(sha)
            if p is not None:
                out.append((sha, p, e["label"]))
        return out

    def select_for_retrain(
        self, replay_ratio: int = 4, min_replay: int = 50, seed: int = 42
    ) -> Tuple[List[Tuple[str, Path, str]], List[Tuple[str, Path, str]]]:
        """
        Picks (train, val) samples for one fine-tuning round without touching more media than needed.
        train = every 'new' file in the train partition + a seeded random replay sample of previously
        trained ('seen') train-partition files (replay_ratio x the new count, at least min_replay) so
        the model doesn't forget older data. val = every validation-partition file, new or seen --
        validation files are never trained on.
        """
        import random

        new_train = self.samples(split="train", status="new")
        seen_train = self.samples(split="train", status="seen")
        k = min(len(seen_train), max(min_replay, replay_ratio * len(new_train)))
        replay = random.Random(seed).sample(seen_train, k) if k else []
        return new_train + replay, self.samples(split="val")

    def counts(self) -> Dict[str, int]:
        ai = sum(1 for e in self._entries.values() if e["label"] == "ai_generated")
        real = sum(1 for e in self._entries.values() if e["label"] == "real")
        new = sum(1 for e in self._entries.values() if e.get("status") == "new")
        return {"ai_generated": ai, "real": real, "total": ai + real, "new": new, "missing": len(self.missing())}

    def mark_seen(self, shas: Iterable[str]) -> None:
        """Marks entries as folded into a promoted checkpoint (no longer 'new')."""
        for sha in shas:
            if sha in self._entries:
                self._entries[sha]["status"] = "seen"
        self._save()


def register_feedback(library: Optional[MediaLibrary], path: str | Path, user_label: str) -> bool:
    """
    Queues a user-verified file for the next retrain, by reference (no copy). 'AI' -> ai_generated,
    anything else -> real. Returns False (and never raises) if there is no library or the file
    is no longer readable, so feedback recording itself can't be broken by training bookkeeping.
    """
    if library is None:
        return False
    label = "ai_generated" if str(user_label).strip().upper() == "AI" else "real"
    try:
        return library.add(path, label)
    except OSError as exc:
        logger.warning("Feedback not queued for retraining (file unreadable): %s", exc)
        return False


# ---------------------------------------------------------------------- CLI
_MODALITIES = ("image", "audio", "video")


def library_for(modality: str) -> MediaLibrary:
    """The default registry for a modality: <repo>/<modality>_detector/data/library.json (gitignored)."""
    if modality not in _MODALITIES:
        raise ValueError(f"modality must be one of {_MODALITIES}")
    repo_root = Path(__file__).resolve().parent.parent
    return MediaLibrary(repo_root / f"{modality}_detector" / "data" / "library.json")


def _extensions_for(modality: str) -> Set[str]:
    import importlib
    return set(importlib.import_module(f"{modality}_detector.config").SUPPORTED_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register labeled media IN PLACE (no copies) for training, by content hash."
    )
    parser.add_argument("modality", choices=_MODALITIES)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_add = sub.add_parser("add", help="register files/folders with a label")
    p_add.add_argument("--label", required=True, choices=LABELS)
    p_add.add_argument("paths", nargs="+")
    p_scan = sub.add_parser("rescan", help="re-link entries whose files moved or were renamed")
    p_scan.add_argument("roots", nargs="+")
    sub.add_parser("stats", help="show counts")
    args = parser.parse_args()

    lib = library_for(args.modality)
    if args.cmd == "add":
        exts = _extensions_for(args.modality)
        total = 0
        for raw in args.paths:
            p = Path(raw)
            total += lib.add_tree(p, args.label, exts) if p.is_dir() else (1 if lib.add(p, args.label) else 0)
        print(f"Registered {total} new/changed item(s) as '{args.label}'. {lib.counts()}")
    elif args.cmd == "rescan":
        print(f"Re-linked {lib.rescan(args.roots)} entr(ies). {lib.counts()}")
    else:
        print(lib.counts())


if __name__ == "__main__":
    main()
