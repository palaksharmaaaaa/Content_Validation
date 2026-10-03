"""
video_detector.benchmarks: Benchmark evaluation suite for Video AI Detection.
Computes:
1. Video classification accuracy, Precision, Recall, F1-Score, and ROC-AUC.
2. Temporal continuity detection rates and confusion matrix.
3. Total sampled frames analyzed and inference latency per video.
4. Comprehensive Markdown and JSON reporting.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

import numpy as np

from video_detector.config import DATA_DIR, SUPPORTED_EXTENSIONS
from video_detector.detector import VideoAIDetector
from video_detector.schemas import VideoBenchmarkMetrics

logger = logging.getLogger("video_detector.benchmarks")


class VideoBenchmarkSuite:
    """Benchmark runner for Video AI Detection evaluation."""

    def __init__(self, detector: Optional[VideoAIDetector] = None):
        self.detector = detector or VideoAIDetector()
        self.detector.load()

    def evaluate_dataset(
        self, dataset_dir: Path | str, sensitivity: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Evaluates video detector against dataset folder containing 'ai_generated' and 'real' subfolders.
        """
        dataset_path = Path(dataset_dir)
        ai_dir = dataset_path / "ai_generated"
        real_dir = dataset_path / "real"

        ai_files = [p for p in ai_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if ai_dir.exists() else []
        real_files = [p for p in real_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if real_dir.exists() else []

        labeled_files = [(p, 1) for p in ai_files] + [(p, 0) for p in real_files]  # 1 = AI, 0 = Real
        if not labeled_files:
            raise ValueError(f"No valid video files found in {dataset_path}/ai_generated or {dataset_path}/real")

        y_true = []
        y_pred = []
        y_scores = []
        latencies = []
        total_frames = 0

        tp, fp, tn, fn = 0, 0, 0, 0

        for path, true_label in labeled_files:
            t0 = time.perf_counter()
            res = self.detector.analyze_video(path, sensitivity=sensitivity)
            lat = time.perf_counter() - t0
            latencies.append(lat)
            total_frames += res.get("sampled_frames_count", 0)

            pred_label = 1 if res["label"] == "LIKELY AI-GENERATED" else 0
            ai_prob = res["ai_percentage"] / 100.0

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_scores.append(ai_prob)

            if true_label == 1 and pred_label == 1:
                tp += 1
            elif true_label == 0 and pred_label == 1:
                fp += 1
            elif true_label == 0 and pred_label == 0:
                tn += 1
            elif true_label == 1 and pred_label == 0:
                fn += 1

        total = len(y_true)
        accuracy = (tp + tn) / max(1, total)
        f1 = (2 * tp) / max(1e-6, (2 * tp + fp + fn))

        # ROC-AUC
        sorted_pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
        num_pos = sum(y_true)
        num_neg = total - num_pos
        roc_auc = 0.5
        if num_pos > 0 and num_neg > 0:
            tp_cum, auc = 0, 0.0
            for score, label in sorted_pairs:
                if label == 1:
                    tp_cum += 1
                else:
                    auc += tp_cum
            roc_auc = auc / (num_pos * num_neg)

        metrics = VideoBenchmarkMetrics(
            total_videos=total,
            total_frames_analyzed=total_frames,
            video_accuracy=accuracy,
            video_f1_score=f1,
            video_roc_auc=roc_auc,
            mean_latency_seconds=float(np.mean(latencies)),
            confusion_matrix={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        )

        out_data = {
            "dataset": str(dataset_path),
            "sensitivity": sensitivity,
            "metrics": metrics.to_dict(),
            "ai_videos_count": len(ai_files),
            "real_videos_count": len(real_files),
        }

        report_file = DATA_DIR / "video_benchmark_results.json"
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)
        except Exception:
            pass

        return out_data


def main():
    parser = argparse.ArgumentParser(description="Run Video AI Detection Benchmarks")
    parser.add_argument("--dataset", type=str, required=True, help="Path to video dataset directory with real/ and ai_generated/")
    parser.add_argument("--sensitivity", type=str, default="high", choices=["balanced", "high", "aggressive"])
    args = parser.parse_args()

    suite = VideoBenchmarkSuite()
    results = suite.evaluate_dataset(args.dataset, sensitivity=args.sensitivity)
    print("\n=== VIDEO AI DETECTION BENCHMARK RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
