"""audio_detector.benchmarks: accuracy of the audio detector on a labelled folder (``real/`` and ``ai_generated/``); see core.benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from core.benchmark import evaluate_folder, write_report
from audio_detector.config import SUPPORTED_EXTENSIONS
from audio_detector.detector import AudioAIDetector


class AudioBenchmarkSuite:
    """Scores a detector on a labelled folder."""

    def __init__(self, detector: Optional[AudioAIDetector] = None):
        self.detector = detector or AudioAIDetector()
        self.detector.load()

    def evaluate_dataset(self, dataset_dir: Path | str, sensitivity: str = "balanced") -> Dict[str, Any]:
        report = evaluate_folder(dataset_dir, SUPPORTED_EXTENSIONS, lambda p: self.detector.predict(p, sensitivity=sensitivity))
        report["sensitivity"] = sensitivity
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score the audio detector on a labelled folder")
    parser.add_argument("--dataset", required=True, help="folder with real/ and ai_generated/")
    parser.add_argument("--sensitivity", default="balanced", choices=["balanced", "high", "aggressive"])
    parser.add_argument("--output", help="also write the report to this JSON file")
    args = parser.parse_args(argv)
    report = AudioBenchmarkSuite().evaluate_dataset(args.dataset, sensitivity=args.sensitivity)
    print(json.dumps(report, indent=2))
    if args.output:
        write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
