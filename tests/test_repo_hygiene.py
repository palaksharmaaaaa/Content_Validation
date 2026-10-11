"""The repository ships code and trained models only: no photographs, audio or video, and every model the code loads from the
repository is tracked, so a fresh clone gives the same results as this machine without any training data."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MEDIA = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".jfif", ".mp4", ".avi", ".mov", ".mkv", ".webm",
         ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
REQUIRED_MODELS = [
    "core/models/face_detection_yunet_2023mar.onnx",
    "core/models/face_recognition_sface_2021dec.onnx",
    "core/models/facial_expression_recognition_mobilefacenet_2022july.onnx",
    "core/models/mivolo_v2/model.safetensors",
    "core/perception/vocab_embeddings.npz",
]


def _tracked():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return out.splitlines()


def test_no_media_file_is_tracked():
    media = [f for f in _tracked() if Path(f).suffix.lower() in MEDIA]
    assert not media, f"media files must never be committed: {media}"


def test_the_project_ships_blank_nothing_trained_and_nothing_learned():
    """Only pretrained models are tracked: no checkpoint trained here, no calibration, feedback, library, fitted statistics or log."""
    learned = []
    for f in _tracked():
        if "tests/" in f or f.startswith("tests"):
            continue                                   # test files may have such words in their names; they hold no learned state
        name = Path(f).name.lower()
        if name.endswith((".pt", ".pth", ".ckpt")) or name in {"checkpoint_log.md", "library.json", "ood_stats.npz"}:
            learned.append(f)
        elif Path(f).suffix.lower() == ".jsonl" or any(k in name for k in ("_calibration", "_feedback", "_memory", "feedback_media")):
            learned.append(f)
    assert not learned, f"these look trained or learned and must not be tracked: {learned}"


@pytest.mark.parametrize("path", REQUIRED_MODELS)
def test_required_model_is_tracked(path):
    assert path in _tracked(), f"{path} is not tracked, so a fresh clone would not have it"


# --- no machine-specific paths -------------------------------------------------------------------------------------------
import re

_TEXT = {".py", ".md", ".json", ".txt", ".yml", ".yaml", ".ini", ".toml", ".cfg"}
_MACHINE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?!/)|/Users/[^/\s]|/home/[a-z][^/\s]*/|AppData|OneDrive|/mnt/[a-z]/|\\\\[A-Za-z0-9_.-]+\\")
# Deliberate inputs of the security tests (a hostile file name / URL), not paths the product uses.
_ALLOWED = {
    "tests/test_repo_hygiene.py",
    "core/tests/test_enterprise_hardening.py",
}


def test_no_machine_specific_path_is_tracked():
    hits = []
    for rel in _tracked():
        if Path(rel).suffix.lower() not in _TEXT or rel in _ALLOWED or rel.startswith("tests/data/"):
            continue
        for n, line in enumerate((ROOT / rel).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if _MACHINE_PATH.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:100]}")
    assert not hits, "paths that only exist on one machine/OS:\n" + "\n".join(hits)
