"""Helpers for golden-regression tests that depend on learned calibration state."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def skip_unless_state_matches(meta_name: str, files: list) -> None:
    """Skips the calling test if any of ``files`` changed since the golden baseline was recorded (e.g. after feedback)."""
    meta_path = ROOT / "tests" / "data" / meta_name
    missing = [rel for rel in files if not (ROOT / rel).is_file()]
    if missing:
        pytest.skip(f"learned state not present on this machine ({missing[0]}); golden baseline does not apply")
    current = {rel: _sha(rel) for rel in files}
    if not meta_path.exists():
        meta_path.write_text(json.dumps(current, indent=1), encoding="utf-8")
        return
    if json.loads(meta_path.read_text(encoding="utf-8")) != current:
        pytest.skip("calibration/checkpoint changed since the golden baseline was recorded; regenerate it deliberately")
