"""Evaluate known/unknown recognition performance across similarity thresholds."""
from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib.pyplot as plt

from src.recognizer import FaceRecognizer, RecognitionResult
from src.reporting import REPORTS_DIR, classification_metrics, save_evaluation_report
from src.utils import EMBEDDINGS_DIR, TEST_IMAGES_DIR, ensure_dir, iter_image_files, read_image


THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
LOGGER = logging.getLogger(__name__)


def expected_label(path: Path) -> str:
    """Return a gallery identity for known images and UNKNOWN for unseen identities."""
    return "UNKNOWN" if "unknown" in path.parts else path.parent.name


def first_result(results: list[RecognitionResult], recognizer: FaceRecognizer, threshold: float) -> str:
    if not results:
        return "UNKNOWN"
    previous = recognizer.threshold
    recognizer.threshold = threshold
    identity, _ = recognizer.recognize_embedding(results[0].embedding)
    recognizer.threshold = previous
    return identity


def run_evaluation(test_dir: Path = TEST_IMAGES_DIR) -> list[dict[str, float]]:
    """Run each test image once, then score its embeddings at every threshold."""
    recognizer = FaceRecognizer(EMBEDDINGS_DIR)
    samples: list[tuple[str, list[RecognitionResult]]] = []
    for path in iter_image_files(test_dir):
        _, results = recognizer.recognize_image(read_image(path))
        samples.append((expected_label(path), results))
    true_labels = [actual for actual, _ in samples]
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        predictions = [first_result(results, recognizer, threshold) for _, results in samples]
        metrics, _, _ = classification_metrics(true_labels, predictions)
        rows.append({"threshold": threshold, **{key: float(value) for key, value in metrics.items() if key != "classification_report"}})
        if threshold == 0.35:
            save_evaluation_report(true_labels, predictions)
    directory = ensure_dir(REPORTS_DIR)
    with (directory / "threshold_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    best = max(rows, key=lambda row: (row["f1_score"], row["accuracy"]))
    (directory / "best_threshold.txt").write_text(f"Best threshold: {best['threshold']:.2f}\nF1 score: {best['f1_score']:.4f}\nAccuracy: {best['accuracy']:.4f}\n", encoding="utf-8")
    plt.figure(figsize=(8, 5))
    plt.plot([row["threshold"] for row in rows], [row["accuracy"] for row in rows], marker="o", label="Accuracy")
    plt.plot([row["threshold"] for row in rows], [row["f1_score"] for row in rows], marker="s", label="F1 score")
    plt.xlabel("Cosine similarity threshold"); plt.ylabel("Score"); plt.title("Threshold Evaluation"); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(directory / "threshold_vs_accuracy.png", dpi=160); plt.close()
    LOGGER.info("Best threshold %.2f (F1=%.4f, accuracy=%.4f)", best["threshold"], best["f1_score"], best["accuracy"])
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_evaluation()
