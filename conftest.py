"""
Repository-wide pytest guard: the test suite must never modify live models or live learner data.

Before the session starts, every trained checkpoint (``*/models/*.pt``) and every live data file
(``*/data/*.json``, ``*/data/*.npz``) is fingerprinted. If any of them changed, was created, or was deleted by
the time the session ends, the whole run fails with the list of offenders. This turns "a test silently overwrote
the production model" into an immediate, visible failure.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

import pytest

ROOT = Path(__file__).resolve().parent
PROTECTED_GLOBS = (
    "image_detector/models/*.pt",
    "audio_detector/models/*.pt",
    "video_detector/models/*.pt",
    "image_detector/data/*.json",
    "audio_detector/data/*.json",
    "video_detector/data/*.json",
    "image_detector/data/*.npz",
    "audio_detector/data/*.npz",
    "video_detector/data/*.npz",
    "core/data/*",
)

_before: Dict[str, str] = {}


def _fingerprint() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pattern in PROTECTED_GLOBS:
        for p in sorted(ROOT.glob(pattern)):
            if p.is_file():
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                out[str(p.relative_to(ROOT)).replace("\\", "/")] = h.hexdigest()
    return out


def pytest_sessionstart(session: pytest.Session) -> None:
    _before.clear()
    _before.update(_fingerprint())


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    after = _fingerprint()
    changed = sorted(k for k in set(_before) | set(after) if _before.get(k) != after.get(k))
    if changed:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        msg = ("TEST-ISOLATION VIOLATION: the test run modified live models/data (use tmp paths in tests): "
               + ", ".join(changed))
        if reporter is not None:
            reporter.write_line(msg, red=True, bold=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
