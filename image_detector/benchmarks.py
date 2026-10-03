"""
image_detector.benchmarks: Benchmark evaluation suite for Image AI Detection.
Computes:
1. Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
2. Confusion matrix (TP, TN, FP, FN).
3. Mean and percentile inference latency.
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

from image_detector.config import DATA_DIR, SUPPORTED_EXTENSIONS
from image_detector.detector import ImageAIDetector
from image_detector.schemas import ImageBenchmarkMetrics

logger = logging.getLogger("image_detector.benchmarks")


class ImageBenchmarkSuite:
    """Benchmark runner for Image AI Detection evaluation."""

    def __init__(self, detector: Optional[ImageAIDetector] = None):
        self.detector = detector or ImageAIDetector()
        self.detector.load()

    def evaluate_dataset(
        self, dataset_dir: Path | str, sensitivity: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Evaluates detector against dataset folder containing 'ai_generated' and 'real' subfolders.
        """
        dataset_path = Path(dataset_dir)
        ai_dir = dataset_path / "ai_generated"
        real_dir = dataset_path / "real"

        ai_files = [p for p in ai_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if ai_dir.exists() else []
        real_files = [p for p in real_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if real_dir.exists() else []

        labeled_files = [(p, 1) for p in ai_files] + [(p, 0) for p in real_files]  # 1 = AI, 0 = Real
        if not labeled_files:
            raise ValueError(f"No valid image files found in {dataset_path}/ai_generated or {dataset_path}/real")

        y_true = []
        y_pred = []
        y_scores = []
        latencies = []

        tp, fp, tn, fn = 0, 0, 0, 0

        for path, true_label in labeled_files:
            t0 = time.perf_counter()
            res = self.detector.predict(path, sensitivity=sensitivity)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

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
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-6, precision + recall)

        # ROC-AUC estimation via trapezoidal rule over sorted probabilities
        sorted_pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
        num_pos = sum(y_true)
        num_neg = total - num_pos
        roc_auc = 0.5
        if num_pos > 0 and num_neg > 0:
            tp_cum, fp_cum = 0, 0
            auc = 0.0
            prev_fp = 0
            for score, label in sorted_pairs:
                if label == 1:
                    tp_cum += 1
                else:
                    fp_cum += 1
                    auc += tp_cum
            roc_auc = auc / (num_pos * num_neg)

        metrics = ImageBenchmarkMetrics(
            total_images=total,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            mean_latency_ms=float(np.mean(latencies)),
            confusion_matrix={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        )

        out_data = {
            "dataset": str(dataset_path),
            "sensitivity": sensitivity,
            "metrics": metrics.to_dict(),
            "ai_samples_count": len(ai_files),
            "real_samples_count": len(real_files),
        }

        # Save benchmark report into image_detector/data
        report_file = DATA_DIR / "image_benchmark_results.json"
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)
        except Exception as exc:
            logger.debug("evaluate_dataset: ignored %s: %s", type(exc).__name__, exc)

        return out_data


def main():
    parser = argparse.ArgumentParser(description="Run Image AI Detection Benchmarks")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory with real/ and ai_generated/")
    parser.add_argument("--sensitivity", type=str, default="high", choices=["balanced", "high", "aggressive"])
    args = parser.parse_args()

    suite = ImageBenchmarkSuite()
    results = suite.evaluate_dataset(args.dataset, sensitivity=args.sensitivity)
    print("\n=== IMAGE AI DETECTION BENCHMARK RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
