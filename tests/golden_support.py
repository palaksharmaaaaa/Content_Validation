"""Helpers for golden-regression tests that depend on learned calibration state."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLANK = "BLANK"


def fingerprint(rel: str) -> str:
    """BLANK for an untouched project (file absent, or a calibration file that has learned nothing); else a content hash."""
    path = ROOT / rel
    if not path.is_file():
        return BLANK
    if path.suffix == ".json":
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("samples_processed", 0) == 0:
                return BLANK
        except ValueError:
            pass
    return hashlib.sha256(path.read_bytes()).hexdigest()


def skip_unless_state_matches(meta_name: str, files: list) -> None:
    """Goldens are recorded on a blank project. Skip (with a reason) if any listed learned-state file is no longer blank
    and differs from what the baseline was recorded with (e.g. after feedback or retraining)."""
    current = {rel: fingerprint(rel) for rel in files}
    if all(v == BLANK for v in current.values()):
        return
    meta_path = ROOT / "tests" / "data" / meta_name
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) == current:
        return
    pytest.skip("learned state present (feedback/retraining/checkpoint); goldens are recorded on a blank project")
