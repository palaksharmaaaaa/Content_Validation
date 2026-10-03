"""
services.fetch_models: download the large pretrained models into the Hugging Face cache (one-off, about 1.6 GB).

    python -m services.fetch_models            # download what is missing
    python -m services.fetch_models --check    # only report what is present

The small ONNX models (face detection, expression, face recognition) are bundled in ``core/models`` and need no download.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict

from core.perception.detector import MODEL_ID as DETECTOR_ID
from core.perception.recognizer import MODEL_ID as RECOGNIZER_ID

MODELS: Dict[str, str] = {
    "Object detector (RF-DETR Small, Apache-2.0)": DETECTOR_ID,
    "Scene / species / vehicle recognizer (SigLIP 2 Base, Apache-2.0)": RECOGNIZER_ID,
}


def present(repo_id: str) -> bool:
    """True if the model is already in the local cache."""
    from core.perception.hub import is_cached

    return is_cached(repo_id)


def main(argv=None) -> int:
    """Entry point; returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--check", action="store_true", help="report only, download nothing")
    args = ap.parse_args(argv)
    missing = 0
    for label, repo in MODELS.items():
        have = present(repo)
        print(f"{'present' if have else 'MISSING':8} {label} [{repo}]")
        if not have and not args.check:
            from huggingface_hub import snapshot_download

            snapshot_download(repo)
            print(f"downloaded {repo}")
        elif not have:
            missing += 1
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
