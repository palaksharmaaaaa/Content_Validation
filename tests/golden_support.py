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


# ---------------------------------------------------------------------------------------------------------------
# Golden baselines are stored as per-case SHA-256 digests (tiny, reviewable in diffs). Regenerate deliberately with
#     UPDATE_GOLDEN=1 python -m pytest <golden test>
# and review what changed (the failure message names the differing cases).
# ---------------------------------------------------------------------------------------------------------------
import os

DATA = ROOT / "tests" / "data"


def _canonical(obj) -> str:
    return json.dumps(json.loads(json.dumps(obj, sort_keys=True, default=str)), sort_keys=True)


def _digest(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()[:20]


def _digests(actual) -> dict:
    if isinstance(actual, dict):
        return {"kind": "dict", "items": {str(k): _digest(v) for k, v in actual.items()}}
    if isinstance(actual, (list, tuple)):
        return {"kind": "list", "items": [_digest(v) for v in actual]}
    return {"kind": "scalar", "items": [_digest(actual)]}


def assert_golden(name: str, actual) -> None:
    """Compare ``actual`` with the stored per-case digests ``tests/data/<name>.digests.json``."""
    path = DATA / f"{name}.digests.json"
    current = _digests(actual)
    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.write_text(json.dumps(current, indent=1, sort_keys=True), encoding="utf-8")
        return
    assert path.exists(), f"missing golden {path.name}; create it with UPDATE_GOLDEN=1"
    expected = json.loads(path.read_text(encoding="utf-8"))
    if current == expected:
        return
    if expected["kind"] == "dict":
        keys = sorted(set(expected["items"]) | set(current["items"]))
        bad = [k for k in keys if expected["items"].get(k) != current["items"].get(k)]
    else:
        n = max(len(expected["items"]), len(current["items"]))
        bad = [i for i in range(n) if (expected["items"][i:i + 1] or [None]) != (current["items"][i:i + 1] or [None])]
    raise AssertionError(
        f"golden '{name}' differs in {len(bad)} case(s): {bad[:8]}{' ...' if len(bad) > 8 else ''}. "
        f"If the change is intended run: UPDATE_GOLDEN=1 python -m pytest -k {name.split('_')[0]}"
    )
